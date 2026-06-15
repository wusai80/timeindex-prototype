"""Online index construction for the TimeIndex prototype."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from importlib import import_module
from typing import Any

from .candidate_index import SkipCandidateIndex
from .config import ConstructionConfig, TimeIndexConfig
from .event import ChainSummary, DecisionIntent, Event, EventRecord, OrdinaryLink, SkipLink
from .extractors import EventRepresentationExtractor
from .scoring import PrototypeScorer
from .stores import ChainStore, EdgeStore, EventStore, KeyDirectory, SkipLinkStore


class IndexConstructor:
    """Legacy stub kept for compatibility with the original skeleton tests."""

    def __init__(
        self,
        config: ConstructionConfig,
        event_store: EventStore,
        key_directory: KeyDirectory,
        edge_store: EdgeStore,
        chain_store: ChainStore,
        skip_candidate_index: SkipCandidateIndex,
        skip_link_store: SkipLinkStore,
    ) -> None:
        self.config = config
        self.event_store = event_store
        self.key_directory = key_directory
        self.edge_store = edge_store
        self.chain_store = chain_store
        self.skip_candidate_index = skip_candidate_index
        self.skip_link_store = skip_link_store

    def add_event(self, event: Event, intent: DecisionIntent | None = None) -> None:
        raise NotImplementedError("Online index construction is not implemented yet.")


class TimeIndex:
    """Online temporal evidence index orchestrator."""

    def __init__(self, config: TimeIndexConfig, intent: DecisionIntent | None = None) -> None:
        self.config = config
        self.default_intent = intent or DecisionIntent()
        self.extractor = self._build_extractor(config)
        self.scorer = self._build_scorer(config)
        self.event_store = self._build_component(EventStore, config)
        self.key_directory = self._build_component(KeyDirectory, config)
        self.edge_store = self._build_component(EdgeStore, config)
        self.chain_store = self._build_component(ChainStore, config)
        self.skip_candidate_index = self._build_component(SkipCandidateIndex, config)
        self.skip_link_store = self._build_component(SkipLinkStore, config)
        self._insertion_order = 0

    def insert(self, event: Event) -> EventRecord:
        record = self._featurize_event(event)
        ordinary_candidates = self._local_candidate_records(record)
        ordinary_links = self._select_ordinary_links(record, ordinary_candidates)
        for link in ordinary_links:
            self._store_ordinary_link(link)

        new_chains = self._extend_chain_summaries(record, ordinary_links)
        skip_candidates = self._skip_candidates(record, self.default_intent, ordinary_links)
        skip_links = self._select_skip_links(record, skip_candidates, ordinary_links)
        for link in skip_links:
            self._store_skip_link(link)

        self._insert_event_record(record)
        self._insert_keys(record)
        self._update_anchor_indexes(record, new_chains)
        self._expire_if_needed()
        return record

    def get_event(self, event_id: str) -> EventRecord | None:
        return self._call_first(self.event_store, ("get",), event_id)

    def ordinary_links(self, event_id: str) -> list[OrdinaryLink]:
        links = self._call_first(self.edge_store, ("incoming", "incoming_links"), event_id)
        return list(links or [])

    def skip_links(self, event_id: str) -> list[SkipLink]:
        links = self._call_first(self.skip_link_store, ("incoming", "incoming_links"), event_id)
        return list(links or [])

    def chains(self, event_id: str) -> list[ChainSummary]:
        chains = self._call_first(
            self.chain_store,
            ("get_for_tail", "for_tail", "tail_summaries"),
            event_id,
        )
        return list(chains or [])

    def _build_extractor(self, config: TimeIndexConfig) -> Any:
        try:
            return EventRepresentationExtractor(config.extractor)
        except TypeError:
            return EventRepresentationExtractor()

    def _build_scorer(self, config: TimeIndexConfig) -> Any:
        try:
            return PrototypeScorer(config.scoring)
        except TypeError:
            return PrototypeScorer()

    def _build_component(self, cls: type[Any], config: TimeIndexConfig) -> Any:
        constructor_args = (
            config.stores,
            config,
        )
        for arg in constructor_args:
            try:
                return cls(arg)
            except TypeError:
                continue
        return cls()

    def _featurize_event(self, event: Event) -> EventRecord:
        extractor_module = import_module("timeindex.extractors")
        featurize_event = getattr(extractor_module, "featurize_event", None)
        if callable(featurize_event):
            record = featurize_event(event, self.config.extractor)
        else:
            try:
                record = self.extractor.extract(event)
            except NotImplementedError:
                record = EventRecord(event=event)
        if record.metadata.insertion_order is None:
            record.metadata.insertion_order = self._insertion_order
        self._insertion_order += 1
        return record

    def _local_candidate_records(self, record: EventRecord) -> list[EventRecord]:
        candidate_ids: list[str] = []
        lookup = self._call_first(self.key_directory, ("lookup_keys",), record.lookup_keys)
        if lookup is None:
            for key in sorted(record.lookup_keys):
                matches = self._call_first(self.key_directory, ("lookup",), key)
                candidate_ids.extend(str(event_id) for event_id in (matches or []))
        else:
            candidate_ids.extend(str(event_id) for event_id in lookup)

        seen: set[str] = set()
        candidates: list[EventRecord] = []
        for candidate_id in candidate_ids:
            if candidate_id in seen or candidate_id == record.event.event_id:
                continue
            seen.add(candidate_id)
            candidate = self.get_event(candidate_id)
            if candidate is not None and self._is_valid_event(candidate):
                candidates.append(candidate)
        return candidates

    def _select_ordinary_links(
        self,
        record: EventRecord,
        candidates: Sequence[EventRecord],
    ) -> list[OrdinaryLink]:
        threshold = self.config.scoring.local_dependency_threshold
        scored: list[tuple[float, EventRecord]] = []
        for candidate in candidates:
            score = self._dependency_score(candidate, record)
            if score >= threshold:
                scored.append((score, candidate))

        scored.sort(key=lambda item: (-item[0], self._event_order(item[1]), item[1].event.event_id))
        limit = self.config.stores.ordinary_fan_in
        return [
            OrdinaryLink(
                predecessor_id=candidate.event.event_id,
                successor_id=record.event.event_id,
                score=score,
            )
            for score, candidate in scored[:limit]
        ]

    def _extend_chain_summaries(
        self,
        record: EventRecord,
        ordinary_links: Sequence[OrdinaryLink],
    ) -> list[ChainSummary]:
        new_chains: list[ChainSummary] = []
        limit = self.config.stores.chain_summaries_per_family
        for position, link in enumerate(ordinary_links):
            predecessor = self.get_event(link.predecessor_id)
            if predecessor is None:
                continue
            family = self._chain_family(record, predecessor)
            representative_ids = [predecessor.event.event_id, record.event.event_id]
            summary = ChainSummary(
                chain_id=f"{link.predecessor_id}->{record.event.event_id}:{position}",
                family=family,
                head_id=predecessor.event.event_id,
                tail_id=record.event.event_id,
                representative_event_ids=representative_ids,
                aspects=set(predecessor.aspects) | set(record.aspects),
                dependency_confidence=link.score,
                summary=f"{family} chain from {predecessor.event.event_id} to {record.event.event_id}",
                cost=float(len(representative_ids)),
            )
            self._call_first(self.chain_store, ("add", "add_summary"), summary)
            new_chains.append(summary)
            if len(new_chains) >= limit:
                break
        return new_chains

    def _skip_candidates(
        self,
        record: EventRecord,
        intent: DecisionIntent,
        ordinary_links: Sequence[OrdinaryLink],
    ) -> list[Any]:
        ordinary_predecessors = [link.predecessor_id for link in ordinary_links]
        result = self._call_first(
            self.skip_candidate_index,
            ("get_skip_candidates",),
            record,
            intent,
            ordinary_predecessors,
        )
        if result is None:
            result = self._call_first(
                self.skip_candidate_index,
                ("retrieve",),
                record,
                intent,
            )
        return list(result or [])

    def _select_skip_links(
        self,
        record: EventRecord,
        candidates: Sequence[Any],
        ordinary_links: Sequence[OrdinaryLink],
    ) -> list[SkipLink]:
        threshold = self.config.scoring.skip_threshold
        scored: list[tuple[float, Any]] = []
        ordinary_predecessors = [link.predecessor_id for link in ordinary_links]
        for candidate in candidates:
            if self._candidate_id(candidate) == record.event.event_id:
                continue
            score = self._skip_score(candidate, record, self.default_intent, ordinary_predecessors)
            if score >= threshold:
                scored.append((score, candidate))

        scored.sort(key=lambda item: (-item[0], self._candidate_order(item[1]), self._candidate_id(item[1])))
        limit = self.config.stores.skip_fan_in
        links: list[SkipLink] = []
        for score, candidate in scored[:limit]:
            links.append(
                SkipLink(
                    from_id=self._candidate_id(candidate),
                    to_id=record.event.event_id,
                    skip_value=score,
                    segment_confidence=score,
                    aspects=set(self._candidate_aspects(candidate)),
                    summary=self._candidate_summary(candidate),
                    representative_event_ids=self._candidate_event_ids(candidate),
                    cost=float(max(1, len(self._candidate_event_ids(candidate)))),
                )
            )
        return links

    def _insert_event_record(self, record: EventRecord) -> None:
        self._call_first(self.event_store, ("insert", "add"), record)

    def _insert_keys(self, record: EventRecord) -> None:
        self._call_first(self.key_directory, ("add_event",), record.event.event_id, record.lookup_keys)
        self._call_first(self.key_directory, ("add",), record.event.event_id, sorted(record.lookup_keys))

    def _update_anchor_indexes(
        self,
        record: EventRecord,
        new_chains: Sequence[ChainSummary],
    ) -> None:
        if self._anchor_score(record) >= self.config.scoring.anchor_threshold:
            self._call_first(
                self.skip_candidate_index,
                ("add_event_anchor", "add"),
                record,
                self.default_intent,
            )
        for chain in new_chains:
            if self._anchor_score(chain) >= self.config.scoring.anchor_threshold:
                self._call_first(
                    self.skip_candidate_index,
                    ("add_chain_anchor",),
                    chain,
                    self.default_intent,
                )

    def _expire_if_needed(self) -> None:
        if not self.config.construction.expire_stale_items:
            return
        active_history_size = self.config.stores.active_history_size
        self._call_first(self.event_store, ("expire",), active_history_size)
        self._call_first(self.key_directory, ("expire",), active_history_size)

    def _dependency_score(self, predecessor: EventRecord, current: EventRecord) -> float:
        scoring_module = import_module("timeindex.scoring")
        function = getattr(scoring_module, "dependency_score", None)
        if callable(function):
            return float(function(predecessor, current, self.config.scoring))
        try:
            return float(self.scorer.score_local_dependency(predecessor, current))
        except NotImplementedError:
            return self._fallback_dependency_score(predecessor, current)

    def _skip_score(
        self,
        candidate: Any,
        target: EventRecord,
        intent: DecisionIntent,
        ordinary_predecessors: Sequence[str],
    ) -> float:
        scoring_module = import_module("timeindex.scoring")
        function = getattr(scoring_module, "skip_score", None)
        if callable(function):
            return float(function(candidate, target, intent, ordinary_predecessors, self.config.scoring))
        try:
            return float(self.scorer.score_skip(candidate, target, intent))
        except NotImplementedError:
            base = self._fallback_dependency_score(self._candidate_record(candidate), target)
            bonus = 0.1 if self._candidate_id(candidate) not in set(ordinary_predecessors) else 0.0
            return min(1.0, base + bonus)

    def _anchor_score(self, candidate: EventRecord | ChainSummary) -> float:
        scoring_module = import_module("timeindex.scoring")
        function = getattr(scoring_module, "anchor_score", None)
        if callable(function):
            return float(function(candidate, self.default_intent, []))
        try:
            return float(self.scorer.score_anchor(candidate, self.default_intent))
        except NotImplementedError:
            return self._fallback_anchor_score(candidate)

    def _fallback_dependency_score(self, predecessor: EventRecord, current: EventRecord) -> float:
        overlap = len(set(predecessor.lookup_keys) & set(current.lookup_keys))
        total = len(set(predecessor.lookup_keys) | set(current.lookup_keys))
        if total == 0:
            return 0.0
        return overlap / total

    def _fallback_anchor_score(self, candidate: EventRecord | ChainSummary) -> float:
        if isinstance(candidate, EventRecord):
            return min(1.0, 0.2 + 0.2 * len(candidate.aspects))
        return min(1.0, candidate.dependency_confidence + 0.1 * len(candidate.aspects))

    def _candidate_record(self, candidate: Any) -> EventRecord:
        if isinstance(candidate, EventRecord):
            return candidate
        if isinstance(candidate, ChainSummary):
            event_id = candidate.tail_id
            record = self.get_event(event_id)
            if record is not None:
                return record
        record = self.get_event(self._candidate_id(candidate))
        if record is not None:
            return record
        return EventRecord(event=Event(event_id=self._candidate_id(candidate), time=0, event_type="candidate"))

    def _candidate_id(self, candidate: Any) -> str:
        for attribute in ("event_id", "tail_id", "chain_id", "object_id"):
            value = getattr(candidate, attribute, None)
            if value is not None:
                return str(value)
        if isinstance(candidate, EventRecord):
            return candidate.event.event_id
        return str(candidate)

    def _candidate_order(self, candidate: Any) -> int:
        if isinstance(candidate, EventRecord):
            return self._event_order(candidate)
        if isinstance(candidate, ChainSummary):
            record = self.get_event(candidate.tail_id)
            return self._event_order(record) if record is not None else 0
        return 0

    def _candidate_aspects(self, candidate: Any) -> Iterable[str]:
        if isinstance(candidate, EventRecord):
            return candidate.aspects
        return getattr(candidate, "aspects", set())

    def _candidate_summary(self, candidate: Any) -> str:
        if isinstance(candidate, EventRecord):
            return f"Event anchor {candidate.event.event_id}"
        return str(getattr(candidate, "summary", ""))

    def _candidate_event_ids(self, candidate: Any) -> list[str]:
        if isinstance(candidate, EventRecord):
            return [candidate.event.event_id]
        event_ids = getattr(candidate, "representative_event_ids", None)
        if event_ids:
            return [str(event_id) for event_id in event_ids]
        if isinstance(candidate, ChainSummary):
            return [candidate.head_id, candidate.tail_id]
        return [self._candidate_id(candidate)]

    def _chain_family(self, current: EventRecord, predecessor: EventRecord) -> str:
        shared = sorted(current.aspects & predecessor.aspects)
        if shared:
            return shared[0]
        if current.event.event_type == predecessor.event.event_type:
            return current.event.event_type
        return "generic"

    def _event_order(self, record: EventRecord | None) -> int:
        if record is None:
            return 0
        insertion_order = record.metadata.insertion_order
        return int(insertion_order) if insertion_order is not None else 0

    def _is_valid_event(self, record: EventRecord) -> bool:
        result = self._call_first(self.event_store, ("is_valid",), record.event.event_id)
        if result is None:
            return not record.metadata.expired
        return bool(result)

    def _store_ordinary_link(self, link: OrdinaryLink) -> None:
        self._call_first(self.edge_store, ("insert", "add"), link)

    def _store_skip_link(self, link: SkipLink) -> None:
        self._call_first(self.skip_link_store, ("insert", "add"), link)

    def _call_first(self, target: Any, names: Sequence[str], *args: Any) -> Any:
        for name in names:
            method = getattr(target, name, None)
            if callable(method):
                return method(*args)
        return None
