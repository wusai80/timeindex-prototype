"""Online index construction for the TimeIndex prototype."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
import heapq
from importlib import import_module
from typing import Any

from .candidate_index import SkipCandidateIndex
from .config import ConstructionConfig, TimeIndexConfig
from .event import ChainSummary, DecisionIntent, Event, EventRecord, OrdinaryLink, SkipLink
from .extractors import EventRepresentationExtractor
from .scoring import PrototypeScorer
from .stores import ChainStore, EdgeStore, EntityDirectory, EventStore, KeyDirectory, SkipLinkStore


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
        self._apply_construction_flags()
        self.default_intent = intent or DecisionIntent()
        self.extractor = self._build_extractor(config)
        self.scorer = self._build_scorer(config)
        self.event_store = self._build_component(EventStore, config)
        self.key_directory = self._build_component(KeyDirectory, config)
        self.entity_directory = self._build_component(EntityDirectory, config)
        self.edge_store = self._build_component(EdgeStore, config)
        self.chain_store = self._build_component(ChainStore, config)
        self.skip_candidate_index = self._build_component(SkipCandidateIndex, config)
        self.skip_link_store = self._build_component(SkipLinkStore, config)
        self._featurize_event_impl = self._bind_module_function("timeindex.extractors", "featurize_event")
        self._dependency_score_impl = self._bind_module_function("timeindex.scoring", "dependency_score")
        self._skip_score_impl = self._bind_module_function("timeindex.scoring", "skip_score")
        self._anchor_score_impl = self._bind_module_function("timeindex.scoring", "anchor_score")
        self._rarity_score_impl = self._bind_module_function("timeindex.scoring", "rarity_score")
        self._event_get = self._bind_first(self.event_store, ("get",))
        self._event_insert = self._bind_first(self.event_store, ("insert", "add"))
        self._event_is_valid = self._bind_first(self.event_store, ("is_valid",))
        self._event_expire = self._bind_first(self.event_store, ("expire",))
        self._key_lookup_keys = self._bind_first(self.key_directory, ("lookup_keys",))
        self._key_lookup = self._bind_first(self.key_directory, ("lookup",))
        self._key_add_event = self._bind_first(self.key_directory, ("add_event",))
        self._key_add = self._bind_first(self.key_directory, ("add",))
        self._key_expire = self._bind_first(self.key_directory, ("expire",))
        self._entity_add_event = self._bind_first(self.entity_directory, ("add_event",))
        self._entity_recent_sources = self._bind_first(self.entity_directory, ("recent_sources",))
        self._entity_recent_destinations = self._bind_first(self.entity_directory, ("recent_destinations",))
        self._entity_recent_participants = self._bind_first(self.entity_directory, ("recent_participants",))
        self._entity_recent_flow_pair = self._bind_first(self.entity_directory, ("recent_flow_pair",))
        self._entity_expire = self._bind_first(self.entity_directory, ("expire",))
        self._edge_incoming = self._bind_first(self.edge_store, ("incoming", "incoming_links"))
        self._edge_insert = self._bind_first(self.edge_store, ("insert", "add"))
        self._chain_get_for_tail = self._bind_first(self.chain_store, ("get_for_tail", "for_tail", "tail_summaries"))
        self._chain_add = self._bind_first(self.chain_store, ("add", "add_summary"))
        self._skip_candidate_get = self._bind_first(self.skip_candidate_index, ("get_skip_candidates",))
        self._skip_candidate_retrieve = self._bind_first(self.skip_candidate_index, ("retrieve",))
        self._skip_candidate_get_object = self._bind_first(self.skip_candidate_index, ("get_object",))
        self._skip_candidate_add_event_anchor = self._bind_first(self.skip_candidate_index, ("add_event_anchor", "add"))
        self._skip_candidate_add_chain_anchor = self._bind_first(self.skip_candidate_index, ("add_chain_anchor",))
        self._skip_link_incoming = self._bind_first(self.skip_link_store, ("incoming", "incoming_links"))
        self._skip_link_insert = self._bind_first(self.skip_link_store, ("insert", "add"))
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
        if self._event_get is None:
            return None
        return self._event_get(event_id)

    def ordinary_links(self, event_id: str) -> list[OrdinaryLink]:
        links = self._edge_incoming(event_id) if self._edge_incoming is not None else None
        return list(links or [])

    def skip_links(self, event_id: str) -> list[SkipLink]:
        links = self._skip_link_incoming(event_id) if self._skip_link_incoming is not None else None
        return list(links or [])

    def chains(self, event_id: str) -> list[ChainSummary]:
        chains = self._chain_get_for_tail(event_id) if self._chain_get_for_tail is not None else None
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
        keyword_variants: tuple[dict[str, Any], ...] = ()
        if cls is KeyDirectory:
            keyword_variants = ({"posting_list_size": config.stores.posting_list_size},)
        elif cls is EntityDirectory:
            keyword_variants = ({"posting_list_size": config.stores.posting_list_size},)
        elif cls is EdgeStore:
            keyword_variants = (
                {
                    "fan_in": config.stores.ordinary_fan_in,
                    "maintain_outgoing_links": config.construction.maintain_outgoing_links,
                },
                {"fan_in": config.stores.ordinary_fan_in},
            )
        elif cls is ChainStore:
            keyword_variants = ({"summaries_per_family": config.stores.chain_summaries_per_family},)
        elif cls is SkipLinkStore:
            keyword_variants = (
                {
                    "fan_in": config.stores.skip_fan_in,
                    "maintain_outgoing_links": config.construction.maintain_outgoing_links,
                },
                {"fan_in": config.stores.skip_fan_in},
            )

        for kwargs in keyword_variants:
            try:
                return cls(**kwargs)
            except TypeError:
                continue

        constructor_args = (config.stores, config)
        for arg in constructor_args:
            try:
                return cls(arg)
            except TypeError:
                continue
        return cls()

    def _featurize_event(self, event: Event) -> EventRecord:
        if self._featurize_event_impl is not None:
            record = self._featurize_event_impl(event, self.config.extractor)
        else:
            try:
                record = self.extractor.extract(event)
            except NotImplementedError:
                record = EventRecord(event=event)
        if record.metadata.insertion_order is None:
            record.metadata.insertion_order = self._insertion_order
        self._populate_record_rarity(record)
        self._insertion_order += 1
        return record

    def _local_candidate_records(self, record: EventRecord) -> list[EventRecord]:
        entity_candidates = self._entity_candidate_records(record)
        if entity_candidates:
            return self._prune_local_candidates(record, entity_candidates)

        lookup_keys = self._candidate_lookup_keys(record)
        candidate_ids: list[str] = []
        lookup = self._key_lookup_keys(lookup_keys) if self._key_lookup_keys is not None else None
        if lookup is None:
            for key in sorted(lookup_keys):
                matches = self._key_lookup(key) if self._key_lookup is not None else None
                candidate_ids.extend(str(event_id) for event_id in (matches or []))
        else:
            candidate_ids.extend(str(event_id) for event_id in lookup)

        seen: set[str] = set()
        candidates: list[EventRecord] = []
        fallback_latest: EventRecord | None = None
        for candidate_id in candidate_ids:
            if candidate_id in seen or candidate_id == record.event.event_id:
                continue
            seen.add(candidate_id)
            candidate = self.get_event(candidate_id)
            if candidate is not None and self._is_valid_event(candidate) and self._is_causal_predecessor(candidate, record):
                if fallback_latest is None or self._event_order(candidate) > self._event_order(fallback_latest):
                    fallback_latest = candidate
                if self._passes_ordinary_candidate_prefilter(record, candidate):
                    candidates.append(candidate)
        if not candidates and fallback_latest is not None:
            candidates.append(fallback_latest)
        return self._prune_local_candidates(record, candidates)

    def _entity_candidate_records(self, record: EventRecord) -> list[EventRecord]:
        candidate_ids: list[str] = []
        current_source = sorted(record.source_entities)
        current_destination = sorted(record.destination_entities)

        if self._entity_recent_sources is not None:
            for entity in current_source:
                candidate_ids.extend(self._entity_recent_sources(entity))
        if self._entity_recent_destinations is not None:
            for entity in current_destination:
                candidate_ids.extend(self._entity_recent_destinations(entity))
            for entity in current_source:
                candidate_ids.extend(self._entity_recent_destinations(entity))
        if self._entity_recent_sources is not None:
            for entity in current_destination:
                candidate_ids.extend(self._entity_recent_sources(entity))
        if self._entity_recent_flow_pair is not None:
            for source in current_source:
                for destination in current_destination:
                    candidate_ids.extend(self._entity_recent_flow_pair(source, destination))
        if self._entity_recent_participants is not None:
            for entity in sorted(record.participant_entities):
                candidate_ids.extend(self._entity_recent_participants(entity))

        seen: set[str] = set()
        candidates: list[EventRecord] = []
        fallback_latest: EventRecord | None = None
        for candidate_id in candidate_ids:
            if candidate_id in seen or candidate_id == record.event.event_id:
                continue
            seen.add(candidate_id)
            candidate = self.get_event(candidate_id)
            if candidate is None or not self._is_valid_event(candidate) or not self._is_causal_predecessor(candidate, record):
                continue
            if fallback_latest is None or self._event_order(candidate) > self._event_order(fallback_latest):
                fallback_latest = candidate
            if self._passes_ordinary_candidate_prefilter(record, candidate):
                candidates.append(candidate)
            if len(candidates) >= max(8, self.config.stores.ordinary_fan_in * 2):
                break

        if not candidates and fallback_latest is not None:
            return [fallback_latest]
        return candidates

    def _candidate_lookup_keys(self, record: EventRecord) -> list[str]:
        if record.participant_entities:
            transaction_keys = [
                key
                for key in record.lookup_keys
                if key.startswith(("participant:", "flow_src:", "flow_dst:", "flow_pair:"))
            ]
            if transaction_keys:
                return sorted(transaction_keys)
        entity_keys = [
            key
            for key in record.lookup_keys
            if key.startswith(("entity:", "participant:", "flow_src:", "flow_dst:", "flow_pair:"))
        ]
        if entity_keys:
            return sorted(entity_keys)
        return sorted(record.lookup_keys)

    def _prune_local_candidates(
        self,
        record: EventRecord,
        candidates: Sequence[EventRecord],
    ) -> list[EventRecord]:
        current_source = record.source_entities
        current_destination = record.destination_entities
        same_entity_latest: dict[str, EventRecord] = {}
        bridge_latest: dict[str, EventRecord] = {}
        fallback_latest: EventRecord | None = None

        for candidate in candidates:
            if fallback_latest is None or self._event_order(candidate) > self._event_order(fallback_latest):
                fallback_latest = candidate

            candidate_source = candidate.source_entities
            candidate_destination = candidate.destination_entities

            same_entities = (candidate_source & current_source) | (candidate_destination & current_destination)
            for entity in same_entities:
                existing = same_entity_latest.get(entity)
                if existing is None or self._event_order(candidate) > self._event_order(existing):
                    same_entity_latest[entity] = candidate

            bridge_entities = (candidate_destination & current_source) | (candidate_source & current_destination)
            for entity in bridge_entities:
                existing = bridge_latest.get(entity)
                if existing is None or self._event_order(candidate) > self._event_order(existing):
                    bridge_latest[entity] = candidate

        selected: dict[str, EventRecord] = {}
        for candidate in same_entity_latest.values():
            selected[candidate.event.event_id] = candidate
        for candidate in bridge_latest.values():
            selected[candidate.event.event_id] = candidate
        if not selected and fallback_latest is not None:
            selected[fallback_latest.event.event_id] = fallback_latest

        return sorted(
            selected.values(),
            key=lambda item: (-self._event_order(item), item.event.event_id),
        )

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

        limit = self.config.stores.ordinary_fan_in
        top_scored = heapq.nsmallest(
            limit,
            scored,
            key=lambda item: (-item[0], -self._event_order(item[1]), item[1].event.event_id),
        )
        return [
            OrdinaryLink(
                predecessor_id=candidate.event.event_id,
                successor_id=record.event.event_id,
                score=score,
            )
            for score, candidate in top_scored
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
            inherited = self._inherited_chain_summary(predecessor, family)
            if inherited is not None and not self._should_inherit_chain(inherited, predecessor, record):
                inherited = None
            representative_ids = self._chain_representative_ids(inherited, predecessor)
            source_entities = set(predecessor.source_entities)
            destination_entities = set(predecessor.destination_entities)
            inherited_aspects: set[str] = set()
            dependency_confidence = link.score
            if inherited is not None:
                source_entities.update(inherited.source_entities)
                destination_entities.update(inherited.destination_entities)
                inherited_aspects.update(inherited.aspects)
                dependency_confidence = min(link.score, float(inherited.dependency_confidence or link.score))
            summary = ChainSummary(
                chain_id=f"{link.predecessor_id}->{record.event.event_id}:{position}",
                family=family,
                head_id=inherited.head_id if inherited is not None else predecessor.event.event_id,
                tail_id=record.event.event_id,
                representative_event_ids=representative_ids,
                source_entities=source_entities,
                destination_entities=destination_entities,
                aspects=inherited_aspects | set(predecessor.aspects) | set(record.aspects),
                dependency_confidence=dependency_confidence,
                summary=f"{family} chain from {(inherited.head_id if inherited is not None else predecessor.event.event_id)} to {record.event.event_id}",
                cost=float(max(1, len(representative_ids))),
            )
            if self._chain_add is not None:
                self._chain_add(summary)
            new_chains.append(summary)
            if len(new_chains) >= limit:
                break
        return new_chains

    def _inherited_chain_summary(self, predecessor: EventRecord, family: str) -> ChainSummary | None:
        if self._chain_get_for_tail is None:
            return None
        summaries = list(self._chain_get_for_tail(predecessor.event.event_id) or [])
        compatible = [summary for summary in summaries if summary.family == family]
        if not compatible:
            compatible = summaries
        if not compatible:
            return None
        compatible.sort(key=lambda item: (-item.dependency_confidence, item.chain_id))
        return compatible[0]

    def _chain_representative_ids(
        self,
        inherited: ChainSummary | None,
        predecessor: EventRecord,
    ) -> list[str]:
        event_ids: list[str] = []
        if inherited is not None:
            event_ids.extend(str(event_id) for event_id in inherited.representative_event_ids)
            if inherited.tail_id != predecessor.event.event_id:
                event_ids.append(inherited.tail_id)
        event_ids.append(predecessor.event.event_id)

        seen: set[str] = set()
        compact: list[str] = []
        for event_id in event_ids:
            if event_id in seen:
                continue
            seen.add(event_id)
            compact.append(event_id)
        limit = max(1, self.config.construction.skip_summary_event_limit)
        return compact[-limit:]

    def _should_inherit_chain(
        self,
        inherited: ChainSummary,
        predecessor: EventRecord,
        current: EventRecord,
    ) -> bool:
        if inherited.tail_id != predecessor.event.event_id:
            return False

        local_entities = (
            set(predecessor.source_entities)
            | set(predecessor.destination_entities)
            | set(current.source_entities)
            | set(current.destination_entities)
        )
        inherited_entities = set(inherited.source_entities) | set(inherited.destination_entities)
        new_entities = inherited_entities - local_entities

        local_aspects = set(predecessor.aspects) | set(current.aspects)
        new_aspects = set(inherited.aspects) - local_aspects

        if new_entities or new_aspects:
            return True

        limit = max(1, self.config.construction.skip_summary_event_limit)
        return len(inherited.representative_event_ids) < limit and inherited.head_id != predecessor.event.event_id

    def _skip_candidates(
        self,
        record: EventRecord,
        intent: DecisionIntent,
        ordinary_links: Sequence[OrdinaryLink],
    ) -> list[Any]:
        if not self.config.construction.enable_skip_links or self.config.stores.skip_fan_in <= 0:
            return []
        ordinary_predecessors = [link.predecessor_id for link in ordinary_links]
        result = None
        if self._skip_candidate_get is not None:
            result = self._skip_candidate_get(record, intent, ordinary_predecessors)
        if result is None:
            if self._skip_candidate_retrieve is not None:
                result = self._skip_candidate_retrieve(record, intent)
        candidates: list[Any] = []
        ordinary_predecessor_records = [
            predecessor
            for predecessor in (self.get_event(link.predecessor_id) for link in ordinary_links)
            if predecessor is not None
        ]
        for candidate in list(result or []):
            resolved = self._resolve_candidate(candidate)
            if resolved is not None:
                if self._forward_validates_skip_candidate(resolved, record, ordinary_predecessor_records):
                    candidates.append(resolved)
        limit = max(1, self.config.stores.skip_fan_in * self.config.construction.skip_candidate_pool_factor)
        return self._dedupe_skip_candidates(candidates)[:limit]

    def _select_skip_links(
        self,
        record: EventRecord,
        candidates: Sequence[Any],
        ordinary_links: Sequence[OrdinaryLink],
    ) -> list[SkipLink]:
        threshold = self.config.scoring.skip_threshold
        if not self.config.construction.enable_skip_links or self.config.stores.skip_fan_in <= 0:
            return []
        scored: list[tuple[float, Any]] = []
        ordinary_predecessors = [
            predecessor
            for predecessor in (self.get_event(link.predecessor_id) for link in ordinary_links)
            if predecessor is not None
        ]
        score_predecessors: Sequence[Any] = (
            ordinary_predecessors if self.config.construction.enable_bridge_score else ()
        )
        for candidate in candidates:
            if self._candidate_id(candidate) == record.event.event_id:
                continue
            if not self._candidate_is_causal(candidate, record):
                continue
            score = self._skip_score(candidate, record, self.default_intent, score_predecessors)
            if score >= threshold:
                scored.append((score, candidate))

        limit = self.config.stores.skip_fan_in
        top_scored = heapq.nsmallest(
            limit,
            scored,
            key=lambda item: (-item[0], -self._candidate_order(item[1]), self._candidate_id(item[1])),
        )
        links: list[SkipLink] = []
        for score, candidate in top_scored:
            candidate_event_ids = self._candidate_event_ids(candidate)
            candidate_event_ids = self._bounded_skip_event_ids(candidate_event_ids, record)
            segment_confidence = self._candidate_segment_confidence(candidate, record)
            links.append(
                SkipLink(
                    from_id=self._candidate_id(candidate),
                    to_id=record.event.event_id,
                    skip_value=score,
                    segment_confidence=segment_confidence,
                    source_entities=set(self._candidate_source_entities(candidate)),
                    destination_entities=set(self._candidate_destination_entities(candidate)),
                    aspects=set(self._candidate_aspects(candidate)),
                    summary=self._skip_link_summary(candidate, record, score),
                    representative_event_ids=candidate_event_ids,
                    cost=float(1.0 + 0.25 * max(0, len(candidate_event_ids) - 1)),
                )
            )
        return links

    def _insert_event_record(self, record: EventRecord) -> None:
        if self._event_insert is not None:
            self._event_insert(record)

    def _insert_keys(self, record: EventRecord) -> None:
        if self._key_add_event is not None:
            self._key_add_event(record.event.event_id, record.lookup_keys)
        elif self._key_add is not None:
            self._key_add(record.event.event_id, sorted(record.lookup_keys))
        if self._entity_add_event is not None:
            self._entity_add_event(
                record.event.event_id,
                sorted(record.source_entities),
                sorted(record.destination_entities),
            )

    def _update_anchor_indexes(
        self,
        record: EventRecord,
        new_chains: Sequence[ChainSummary],
    ) -> None:
        existing_anchors = self._existing_anchor_objects()
        if not self.config.construction.enable_skip_links:
            return
        if self._anchor_score(record, existing_anchors) >= self.config.scoring.anchor_threshold:
            if self._skip_candidate_add_event_anchor is not None:
                self._skip_candidate_add_event_anchor(record, self.default_intent)
                existing_anchors = [*existing_anchors, record]
        for chain in new_chains:
            if self._anchor_score(chain, existing_anchors) >= self.config.scoring.anchor_threshold:
                if self._skip_candidate_add_chain_anchor is not None:
                    self._skip_candidate_add_chain_anchor(chain, self.default_intent)
                    existing_anchors = [*existing_anchors, chain]

    def _expire_if_needed(self) -> None:
        if not self.config.construction.expire_stale_items:
            return
        active_history_size = self.config.stores.active_history_size
        if self._active_event_count() <= active_history_size:
            return
        expired_ids: list[str] = []
        if self._event_expire is not None:
            try:
                expired_ids = list(self._event_expire(max_size=active_history_size) or [])
            except TypeError:
                expired_ids = list(self._event_expire(active_history_size) or [])

        if self._key_expire is not None:
            try:
                self._key_expire(expired_ids)
            except TypeError:
                if expired_ids:
                    self._key_expire(tuple(expired_ids))
        if self._entity_expire is not None:
            self._entity_expire(expired_ids)

        for store in (self.edge_store, self.chain_store, self.skip_link_store, self.skip_candidate_index):
            expire_method = getattr(store, "expire", None)
            if callable(expire_method):
                try:
                    expire_method(expired_ids)
                except TypeError:
                    continue

    def _dependency_score(self, predecessor: EventRecord, current: EventRecord) -> float:
        if self._dependency_score_impl is not None:
            return float(self._dependency_score_impl(predecessor, current, self.config.scoring))
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
        if self._skip_score_impl is not None:
            return float(self._skip_score_impl(candidate, target, intent, ordinary_predecessors, self.config.scoring))
        try:
            return float(self.scorer.score_skip(candidate, target, intent))
        except NotImplementedError:
            base = self._fallback_dependency_score(self._candidate_record(candidate), target)
            predecessor_ids = {self._candidate_id(item) for item in ordinary_predecessors}
            bonus = 0.1 if self._candidate_id(candidate) not in predecessor_ids else 0.0
            return min(1.0, base + bonus)

    def _anchor_score(
        self,
        candidate: EventRecord | ChainSummary,
        existing_anchors: Sequence[EventRecord | ChainSummary],
    ) -> float:
        if self._anchor_score_impl is not None:
            return float(self._anchor_score_impl(candidate, self.default_intent, existing_anchors))
        try:
            return float(self.scorer.score_anchor(candidate, self.default_intent))
        except NotImplementedError:
            return self._fallback_anchor_score(candidate)

    def _fallback_dependency_score(self, predecessor: EventRecord, current: EventRecord) -> float:
        exact_overlap = len(set(predecessor.lookup_keys) & set(current.lookup_keys))
        total = len(set(predecessor.lookup_keys) | set(current.lookup_keys))
        overlap_score = 0.0 if total == 0 else exact_overlap / total
        predecessor_destination = predecessor.destination_entities
        predecessor_source = predecessor.source_entities
        current_source = current.source_entities
        current_destination = current.destination_entities
        continuity = 0.0
        if predecessor_destination & current_source:
            continuity = 1.0
        elif predecessor_source & current_source:
            continuity = 0.75
        elif predecessor_destination & current_destination:
            continuity = 0.60
        return max(overlap_score, continuity)

    def _fallback_anchor_score(self, candidate: EventRecord | ChainSummary) -> float:
        if isinstance(candidate, EventRecord):
            return min(1.0, 0.2 + 0.2 * len(candidate.aspects))
        return min(1.0, candidate.dependency_confidence + 0.1 * len(candidate.aspects))

    def _candidate_record(self, candidate: Any) -> EventRecord:
        resolved = self._resolve_candidate(candidate)
        if resolved is not candidate:
            return self._candidate_record(resolved)
        if isinstance(candidate, EventRecord):
            return candidate
        if isinstance(candidate, ChainSummary):
            event_id = candidate.representative_event_ids[0] if candidate.representative_event_ids else candidate.head_id
            record = self.get_event(event_id)
            if record is not None:
                return record
        record = self.get_event(self._candidate_id(candidate))
        if record is not None:
            return record
        return EventRecord(event=Event(event_id=self._candidate_id(candidate), time=0, event_type="candidate"))

    def _candidate_id(self, candidate: Any) -> str:
        resolved = self._resolve_candidate(candidate)
        if resolved is not candidate:
            return self._candidate_id(resolved)
        for attribute in ("event_id", "tail_id", "chain_id", "object_id"):
            value = getattr(candidate, attribute, None)
            if value is not None:
                return str(value)
        if isinstance(candidate, EventRecord):
            return candidate.event.event_id
        return str(candidate)

    def _candidate_order(self, candidate: Any) -> int:
        resolved = self._resolve_candidate(candidate)
        if resolved is not candidate:
            return self._candidate_order(resolved)
        if isinstance(candidate, EventRecord):
            return self._event_order(candidate)
        if isinstance(candidate, ChainSummary):
            record = self.get_event(candidate.tail_id)
            return self._event_order(record) if record is not None else 0
        return 0

    def _candidate_aspects(self, candidate: Any) -> Iterable[str]:
        resolved = self._resolve_candidate(candidate)
        if resolved is not candidate:
            return self._candidate_aspects(resolved)
        if isinstance(candidate, EventRecord):
            return candidate.aspects
        return getattr(candidate, "aspects", set())

    def _candidate_source_entities(self, candidate: Any) -> Iterable[str]:
        resolved = self._resolve_candidate(candidate)
        if resolved is not candidate:
            return self._candidate_source_entities(resolved)
        if isinstance(candidate, EventRecord):
            return candidate.source_entities
        return getattr(candidate, "source_entities", set())

    def _candidate_destination_entities(self, candidate: Any) -> Iterable[str]:
        resolved = self._resolve_candidate(candidate)
        if resolved is not candidate:
            return self._candidate_destination_entities(resolved)
        if isinstance(candidate, EventRecord):
            return candidate.destination_entities
        return getattr(candidate, "destination_entities", set())

    def _candidate_summary(self, candidate: Any) -> str:
        resolved = self._resolve_candidate(candidate)
        if resolved is not candidate:
            return self._candidate_summary(resolved)
        if isinstance(candidate, EventRecord):
            return f"Event anchor {candidate.event.event_id}"
        return str(getattr(candidate, "summary", ""))

    def _candidate_segment_confidence(self, candidate: Any, target: EventRecord) -> float:
        resolved = self._resolve_candidate(candidate)
        if resolved is not candidate:
            return self._candidate_segment_confidence(resolved, target)
        if isinstance(candidate, EventRecord):
            return self._dependency_score(candidate, target)
        if isinstance(candidate, ChainSummary):
            return float(candidate.dependency_confidence)
        return 0.0

    def _candidate_event_ids(self, candidate: Any) -> list[str]:
        resolved = self._resolve_candidate(candidate)
        if resolved is not candidate:
            return self._candidate_event_ids(resolved)
        if isinstance(candidate, EventRecord):
            return [candidate.event.event_id]
        event_ids = getattr(candidate, "representative_event_ids", None)
        if event_ids:
            return [str(event_id) for event_id in event_ids]
        if isinstance(candidate, ChainSummary):
            return [candidate.head_id, candidate.tail_id]
        return [self._candidate_id(candidate)]

    def _forward_validates_skip_candidate(
        self,
        candidate: Any,
        target: EventRecord,
        ordinary_predecessors: Sequence[EventRecord],
    ) -> bool:
        candidate_source = set(self._candidate_source_entities(candidate))
        candidate_destination = set(self._candidate_destination_entities(candidate))
        if self._entities_have_transaction_continuity(candidate_source, candidate_destination, target):
            return True
        for predecessor in ordinary_predecessors:
            if self._entities_have_transaction_continuity(candidate_source, candidate_destination, predecessor):
                return True
        candidate_aspects = set(self._candidate_aspects(candidate))
        if candidate_aspects & target.aspects:
            return True
        return False

    def _dedupe_skip_candidates(self, candidates: Sequence[Any]) -> list[Any]:
        deduped: dict[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], Any] = {}
        for candidate in candidates:
            signature = (
                tuple(sorted(str(value) for value in self._candidate_source_entities(candidate))),
                tuple(sorted(str(value) for value in self._candidate_destination_entities(candidate))),
                tuple(sorted(str(value) for value in self._candidate_aspects(candidate))),
            )
            existing = deduped.get(signature)
            if existing is None or self._candidate_order(candidate) > self._candidate_order(existing):
                deduped[signature] = candidate
        return sorted(
            deduped.values(),
            key=lambda item: (-self._candidate_order(item), self._candidate_id(item)),
        )

    def _bounded_skip_event_ids(self, candidate_event_ids: Sequence[str], target: EventRecord) -> list[str]:
        bounded = [
            event_id
            for event_id in candidate_event_ids
            if event_id != target.event.event_id
        ]
        limit = max(1, self.config.construction.skip_summary_event_limit)
        if len(bounded) <= limit:
            return bounded
        return bounded[:limit]

    def _skip_link_summary(self, candidate: Any, target: EventRecord, score: float) -> str:
        base = self._candidate_summary(candidate).strip()
        if not base:
            base = f"Bridge evidence from {self._candidate_id(candidate)} into {target.event.event_id}"
        if "bridge" not in base.lower():
            base = f"{base} [bridge={score:.2f}]"
        return base

    def _entities_have_transaction_continuity(
        self,
        predecessor_source: set[str],
        predecessor_destination: set[str],
        current: EventRecord,
    ) -> bool:
        return bool(
            (predecessor_destination & current.source_entities)
            or (predecessor_source & current.source_entities)
            or (predecessor_destination & current.destination_entities)
            or (predecessor_source & current.destination_entities)
        )

    def _resolve_candidate(self, candidate: Any) -> Any:
        if isinstance(candidate, (EventRecord, ChainSummary)):
            return candidate
        if isinstance(candidate, str):
            if self._skip_candidate_get_object is not None:
                resolved = self._skip_candidate_get_object(candidate)
                if resolved is not None:
                    return resolved
            record = self.get_event(candidate)
            if record is not None:
                return record
        return candidate

    def _chain_family(self, current: EventRecord, predecessor: EventRecord) -> str:
        predecessor_destination = predecessor.destination_entities
        current_source = current.source_entities
        if predecessor_destination & current_source:
            return "transaction_flow"
        predecessor_source = predecessor.source_entities
        if predecessor_source & current_source:
            return "account_sequence"
        shared = sorted(current.aspects & predecessor.aspects)
        if shared:
            return shared[0]
        if current.event.event_type == predecessor.event.event_type:
            return current.event.event_type
        return "generic"

    def _event_entities(self, record: EventRecord, role: str) -> set[str]:
        if role == "source":
            return set(record.source_entities)
        return set(record.destination_entities)

    def _event_order(self, record: EventRecord | None) -> int:
        if record is None:
            return 0
        insertion_order = record.metadata.insertion_order
        return int(insertion_order) if insertion_order is not None else 0

    def _is_valid_event(self, record: EventRecord) -> bool:
        result = self._event_is_valid(record.event.event_id) if self._event_is_valid is not None else None
        if result is None:
            return not record.metadata.expired
        return bool(result)

    def _is_causal_predecessor(self, predecessor: EventRecord, current: EventRecord) -> bool:
        return self._time_sort_key(predecessor.event.time) <= self._time_sort_key(current.event.time)

    def _candidate_is_causal(self, candidate: Any, target: EventRecord) -> bool:
        resolved = self._resolve_candidate(candidate)
        if isinstance(resolved, EventRecord):
            return self._is_causal_predecessor(resolved, target)
        if isinstance(resolved, ChainSummary):
            event_ids = list(resolved.representative_event_ids) or [resolved.head_id, resolved.tail_id]
            observed = False
            for event_id in event_ids:
                record = self.get_event(str(event_id))
                if record is None:
                    continue
                observed = True
                if not self._is_causal_predecessor(record, target):
                    return False
            return observed
        return True

    def _store_ordinary_link(self, link: OrdinaryLink) -> None:
        if self._edge_insert is not None:
            self._edge_insert(link)

    def _store_skip_link(self, link: SkipLink) -> None:
        if self._skip_link_insert is not None:
            self._skip_link_insert(link)

    def _passes_ordinary_candidate_prefilter(self, record: EventRecord, candidate: EventRecord) -> bool:
        if record.participant_entities and candidate.participant_entities:
            return self._has_transaction_continuity(candidate, record)
        if candidate.entity_keys & record.entity_keys:
            return True
        if candidate.attribute_keys & record.attribute_keys:
            return True
        if candidate.context_keys & record.context_keys:
            return True
        return bool(candidate.aspects & record.aspects)

    def _has_transaction_continuity(self, predecessor: EventRecord, current: EventRecord) -> bool:
        return bool(
            (predecessor.destination_entities & current.source_entities)
            or (predecessor.source_entities & current.source_entities)
            or (predecessor.destination_entities & current.destination_entities)
            or (predecessor.source_entities & current.destination_entities)
        )

    def _existing_anchor_objects(self) -> list[EventRecord | ChainSummary]:
        recent_method = getattr(getattr(self.skip_candidate_index, "anchor_table", None), "recent", None)
        if not callable(recent_method):
            return []
        objects: list[EventRecord | ChainSummary] = []
        for entry in recent_method():
            obj = getattr(entry, "obj", None)
            if isinstance(obj, (EventRecord, ChainSummary)):
                objects.append(obj)
        return objects

    def _active_event_count(self) -> int:
        try:
            return len(self.event_store)
        except TypeError:
            records = getattr(self.event_store, "_records", None)
            if isinstance(records, dict):
                return len(records)
            order = getattr(self.event_store, "order", None)
            if order is not None:
                return len(order)
        return 0

    def _bind_module_function(self, module_name: str, attribute: str) -> Any:
        module = import_module(module_name)
        function = getattr(module, attribute, None)
        return function if callable(function) else None

    def _bind_first(self, target: Any, names: Sequence[str]) -> Any:
        for name in names:
            method = getattr(target, name, None)
            if callable(method):
                return method
        return None

    def _call_first(self, target: Any, names: Sequence[str], *args: Any) -> Any:
        for name in names:
            method = getattr(target, name, None)
            if callable(method):
                return method(*args)
        return None

    def _time_sort_key(self, value: Any) -> tuple[int, float | str]:
        if isinstance(value, (int, float)):
            return (0, float(value))
        text = str(value).strip()
        if not text:
            return (1, "")
        try:
            return (0, float(text))
        except ValueError:
            pass
        for candidate in (text, text.replace("Z", "+00:00"), text.replace("/", "-")):
            try:
                return (0, datetime.fromisoformat(candidate).timestamp())
            except ValueError:
                continue
        return (1, text)

    def _apply_construction_flags(self) -> None:
        if not self.config.construction.enable_skip_links:
            self.config.stores.skip_fan_in = 0
        if not self.config.construction.enable_aspect_candidates:
            self.config.stores.aspect_candidates = 0
        if not self.config.construction.enable_rare_candidates:
            self.config.stores.rarity_candidates = 0
        if not self.config.construction.enable_correlation_candidates:
            self.config.stores.correlation_candidates = 0

    def _populate_record_rarity(self, record: EventRecord) -> None:
        if self._rarity_score_impl is None or not record.lookup_keys:
            return
        key_frequencies: dict[str, int] = {}
        for key in record.lookup_keys:
            matches = self._key_lookup(key) if self._key_lookup is not None else ()
            key_frequencies[key] = len(list(matches or ()))
        history_size = max(self._active_event_count(), 1)
        record.metadata.rarity = float(self._rarity_score_impl(record, key_frequencies, history_size))
