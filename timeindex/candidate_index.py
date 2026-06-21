"""Bounded candidate index for long-range evidence anchors."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .config import StoreConfig
from .event import ChainSummary, DecisionIntent, EventRecord


@dataclass(slots=True)
class AnchorEntry:
    """Anchor object tracked across the skip-candidate sub-indexes."""

    anchor_id: str
    kind: str
    obj: EventRecord | ChainSummary
    aspects: set[str]
    vector: np.ndarray | None
    rarity: float
    intent_name: str | None
    insertion_order: int


class AnchorTable:
    """Recent bounded table of anchor entries."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._entries: list[AnchorEntry] = []

    def add(self, entry: AnchorEntry) -> None:
        self._entries = [item for item in self._entries if item.anchor_id != entry.anchor_id]
        self._entries.append(entry)
        self._entries.sort(key=lambda item: (-item.insertion_order, item.anchor_id))
        self._entries = self._entries[: self.limit]

    def recent(self) -> list[AnchorEntry]:
        return list(self._entries)

    def expire(self, expired_anchor_ids: Iterable[str]) -> None:
        expired = set(expired_anchor_ids)
        if not expired:
            return
        self._entries = [entry for entry in self._entries if entry.anchor_id not in expired]


class CorrIndex:
    """Vector-similarity index backed by a bounded list."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._entries: list[AnchorEntry] = []

    def add(self, entry: AnchorEntry) -> None:
        if entry.vector is None:
            return
        self._entries = [item for item in self._entries if item.anchor_id != entry.anchor_id]
        self._entries.append(entry)
        self._entries.sort(key=lambda item: (-item.insertion_order, item.anchor_id))
        self._entries = self._entries[: self.limit]

    def query(self, vector: np.ndarray | None, limit: int) -> list[tuple[AnchorEntry, float]]:
        if vector is None:
            return []
        scored = [
            (entry, _cosine_similarity(vector, entry.vector))
            for entry in self._entries
            if entry.vector is not None
        ]
        scored.sort(key=lambda item: (-item[1], -item[0].insertion_order, item[0].anchor_id))
        return scored[:limit]

    def expire(self, expired_anchor_ids: Iterable[str]) -> None:
        expired = set(expired_anchor_ids)
        if not expired:
            return
        self._entries = [entry for entry in self._entries if entry.anchor_id not in expired]


class RarityIndex:
    """Rarity-ranked bounded anchor list."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._entries: list[AnchorEntry] = []

    def add(self, entry: AnchorEntry) -> None:
        self._entries = [item for item in self._entries if item.anchor_id != entry.anchor_id]
        self._entries.append(entry)
        self._entries.sort(key=lambda item: (-item.rarity, -item.insertion_order, item.anchor_id))
        self._entries = self._entries[: self.limit]

    def top(self) -> list[AnchorEntry]:
        return list(self._entries)

    def expire(self, expired_anchor_ids: Iterable[str]) -> None:
        expired = set(expired_anchor_ids)
        if not expired:
            return
        self._entries = [entry for entry in self._entries if entry.anchor_id not in expired]


class IntentIndex:
    """Intent-aware bounded anchor postings."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._entries: list[AnchorEntry] = []

    def add(self, entry: AnchorEntry) -> None:
        self._entries = [item for item in self._entries if item.anchor_id != entry.anchor_id]
        self._entries.append(entry)
        self._entries.sort(key=lambda item: (-item.insertion_order, item.anchor_id))
        self._entries = self._entries[: self.limit]

    def query(self, intent: DecisionIntent | None) -> list[tuple[AnchorEntry, float]]:
        if intent is None:
            return []

        scored: list[tuple[AnchorEntry, float]] = []
        for entry in self._entries:
            overlap = len(intent.aspects & entry.aspects)
            aspect_score = overlap / max(1, len(intent.aspects))
            name_score = 1.0 if intent.name and entry.intent_name == intent.name else 0.0
            total = 0.75 * aspect_score + 0.25 * name_score
            if total > 0.0:
                scored.append((entry, total))

        scored.sort(key=lambda item: (-item[1], -item[0].insertion_order, item[0].anchor_id))
        return scored[: self.limit]

    def expire(self, expired_anchor_ids: Iterable[str]) -> None:
        expired = set(expired_anchor_ids)
        if not expired:
            return
        self._entries = [entry for entry in self._entries if entry.anchor_id not in expired]


class AspectIndex:
    """Aspect-overlap bounded anchor postings."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._entries: list[AnchorEntry] = []

    def add(self, entry: AnchorEntry) -> None:
        self._entries = [item for item in self._entries if item.anchor_id != entry.anchor_id]
        self._entries.append(entry)
        self._entries.sort(key=lambda item: (-item.insertion_order, item.anchor_id))
        self._entries = self._entries[: self.limit]

    def query(self, aspects: set[str]) -> list[tuple[AnchorEntry, float]]:
        if not aspects:
            return []
        scored: list[tuple[AnchorEntry, float]] = []
        for entry in self._entries:
            overlap = len(aspects & entry.aspects)
            if overlap == 0:
                continue
            score = overlap / max(1, len(aspects))
            scored.append((entry, score))
        scored.sort(key=lambda item: (-item[1], -item[0].insertion_order, item[0].anchor_id))
        return scored[: self.limit]

    def expire(self, expired_anchor_ids: Iterable[str]) -> None:
        expired = set(expired_anchor_ids)
        if not expired:
            return
        self._entries = [entry for entry in self._entries if entry.anchor_id not in expired]


class SkipCandidateIndex:
    """Bounded multi-view candidate index for skip-link anchors."""

    def __init__(self, config: StoreConfig | None = None) -> None:
        self.config = config or StoreConfig()
        self.anchor_table = AnchorTable(self.config.anchor_candidates)
        self.corr_index = CorrIndex(self.config.correlation_candidates)
        self.rarity_index = RarityIndex(self.config.rarity_candidates)
        self.intent_index = IntentIndex(self.config.intent_candidates)
        self.aspect_index = AspectIndex(self.config.aspect_candidates)
        self._objects: dict[str, EventRecord | ChainSummary] = {}
        self._next_order = 0

    def add_event_anchor(self, record: EventRecord, intent: DecisionIntent | None = None) -> None:
        entry = AnchorEntry(
            anchor_id=record.event.event_id,
            kind="event",
            obj=record,
            aspects=set(record.aspects),
            vector=record.sketch,
            rarity=float(record.metadata.rarity),
            intent_name=intent.name if intent is not None else None,
            insertion_order=self._next_insertion_order(),
        )
        self._add_entry(entry)

    def add_chain_anchor(self, summary: ChainSummary, intent: DecisionIntent | None = None) -> None:
        entry = AnchorEntry(
            anchor_id=summary.chain_id,
            kind="chain",
            obj=summary,
            aspects=set(summary.aspects),
            vector=None,
            rarity=float(summary.dependency_confidence),
            intent_name=intent.name if intent is not None else None,
            insertion_order=self._next_insertion_order(),
        )
        self._add_entry(entry)

    def retrieve(self, record: EventRecord, intent: DecisionIntent | None = None) -> Sequence[str]:
        return self.get_skip_candidates(record, intent, ordinary_predecessors=())

    def get_skip_candidates(
        self,
        event: EventRecord,
        intent: DecisionIntent | None,
        ordinary_predecessors: Sequence[str],
    ) -> list[str]:
        excluded_ids = set(ordinary_predecessors)
        excluded_ids.add(event.event.event_id)

        aggregate_scores: dict[str, float] = {}
        insertion_orders: dict[str, int] = {}

        def add_scored(entries: list[tuple[AnchorEntry, float]]) -> None:
            for entry, score in entries:
                if entry.anchor_id in excluded_ids:
                    continue
                participant_score = _participant_relevance(entry.obj, event)
                inflow_score = _inflow_relevance(entry.obj, event)
                generic_penalty = _generic_penalty(entry.obj)
                total_score = score + 0.35 * participant_score + 0.35 * inflow_score - generic_penalty
                if total_score <= 0.0:
                    continue
                aggregate_scores[entry.anchor_id] = aggregate_scores.get(entry.anchor_id, 0.0) + total_score
                insertion_orders[entry.anchor_id] = entry.insertion_order

        add_scored([(entry, 0.05) for entry in self.anchor_table.recent()])
        add_scored(self.corr_index.query(event.sketch, self.config.correlation_candidates))
        add_scored([(entry, entry.rarity) for entry in self.rarity_index.top()])
        add_scored(self.intent_index.query(intent))
        add_scored(self.aspect_index.query(set(event.aspects)))

        limit = max(
            1,
            self.config.anchor_candidates,
            self.config.correlation_candidates,
            self.config.rarity_candidates,
            self.config.intent_candidates,
            self.config.aspect_candidates,
        )
        ranked = sorted(
            aggregate_scores.items(),
            key=lambda item: (-item[1], -insertion_orders[item[0]], item[0]),
        )

        diversified: list[str] = []
        seen_signatures: set[tuple[str, ...]] = set()
        for anchor_id, _score in ranked:
            obj = self._objects.get(anchor_id)
            if obj is None:
                continue
            signature = _bridge_signature(obj)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            diversified.append(anchor_id)
            if len(diversified) >= limit:
                break
        return diversified

    def get_object(self, anchor_id: str) -> EventRecord | ChainSummary | None:
        return self._objects.get(anchor_id)

    def expire(self, expired_event_ids: Iterable[str]) -> None:
        expired = set(str(event_id) for event_id in expired_event_ids)
        if not expired:
            return

        expired_anchor_ids: set[str] = set()
        for anchor_id, obj in list(self._objects.items()):
            if isinstance(obj, EventRecord):
                if obj.event.event_id in expired:
                    expired_anchor_ids.add(anchor_id)
                    del self._objects[anchor_id]
            elif isinstance(obj, ChainSummary):
                if (
                    obj.head_id in expired
                    or obj.tail_id in expired
                    or set(obj.representative_event_ids) & expired
                ):
                    expired_anchor_ids.add(anchor_id)
                    del self._objects[anchor_id]

        if not expired_anchor_ids:
            return
        self.anchor_table.expire(expired_anchor_ids)
        self.corr_index.expire(expired_anchor_ids)
        self.rarity_index.expire(expired_anchor_ids)
        self.intent_index.expire(expired_anchor_ids)
        self.aspect_index.expire(expired_anchor_ids)
        self._refresh_objects()

    def _add_entry(self, entry: AnchorEntry) -> None:
        self._objects[entry.anchor_id] = entry.obj
        self.anchor_table.add(entry)
        self.corr_index.add(entry)
        self.rarity_index.add(entry)
        self.intent_index.add(entry)
        self.aspect_index.add(entry)
        self._refresh_objects()

    def _next_insertion_order(self) -> int:
        order = self._next_order
        self._next_order += 1
        return order

    def _refresh_objects(self) -> None:
        retained_ids = self._retained_anchor_ids()
        if not retained_ids:
            self._objects.clear()
            return
        self._objects = {
            anchor_id: obj
            for anchor_id, obj in self._objects.items()
            if anchor_id in retained_ids
        }

    def _retained_anchor_ids(self) -> set[str]:
        retained_ids: set[str] = set()
        for entries in (
            self.anchor_table.recent(),
            self.corr_index._entries,
            self.rarity_index._entries,
            self.intent_index._entries,
            self.aspect_index._entries,
        ):
            retained_ids.update(entry.anchor_id for entry in entries)
        return retained_ids


def _cosine_similarity(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float:
    if vec_a is None or vec_b is None:
        return 0.0
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _bridge_signature(obj: EventRecord | ChainSummary) -> tuple[str, ...]:
    if isinstance(obj, EventRecord):
        source = sorted(obj.source_entities)
        destination = sorted(obj.destination_entities)
        if source or destination:
            return ("event", *source, "|", *destination)
        return ("event", obj.event.event_type, *sorted(obj.aspects))

    source = sorted(str(value) for value in getattr(obj, "source_entities", ()))
    destination = sorted(str(value) for value in getattr(obj, "destination_entities", ()))
    if source or destination:
        return ("chain", *source, "|", *destination)
    return ("chain", str(obj.family), *sorted(str(aspect) for aspect in obj.aspects))


def _participant_relevance(anchor: EventRecord | ChainSummary, event: EventRecord) -> float:
    anchor_source, anchor_destination = _anchor_entities(anchor)
    if not (anchor_source or anchor_destination):
        return 0.0
    target_source = set(event.source_entities)
    target_destination = set(event.destination_entities)
    if anchor_destination & target_source:
        return 1.0
    if anchor_source & target_source:
        return 0.75
    if anchor_destination & target_destination:
        return 0.60
    if anchor_source & target_destination:
        return 0.50
    if (anchor_source | anchor_destination) & (target_source | target_destination):
        return 0.40
    return 0.0


def _inflow_relevance(anchor: EventRecord | ChainSummary, event: EventRecord) -> float:
    anchor_source, anchor_destination = _anchor_entities(anchor)
    target_source = set(event.source_entities)
    if not target_source:
        return 0.0
    if anchor_destination & target_source:
        return 1.0
    return 0.0


def _generic_penalty(anchor: EventRecord | ChainSummary) -> float:
    aspects = set(getattr(anchor, "aspects", ()))
    if aspects == {"generic_evidence"}:
        return 0.15
    return 0.0


def _anchor_entities(anchor: EventRecord | ChainSummary) -> tuple[set[str], set[str]]:
    if isinstance(anchor, EventRecord):
        return set(anchor.source_entities), set(anchor.destination_entities)
    return set(getattr(anchor, "source_entities", ())), set(getattr(anchor, "destination_entities", ()))
