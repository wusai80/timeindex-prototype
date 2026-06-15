"""Bounded candidate index for long-range evidence anchors."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

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
                aggregate_scores[entry.anchor_id] = aggregate_scores.get(entry.anchor_id, 0.0) + score
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
        return [anchor_id for anchor_id, _score in ranked[:limit]]

    def get_object(self, anchor_id: str) -> EventRecord | ChainSummary | None:
        return self._objects.get(anchor_id)

    def _add_entry(self, entry: AnchorEntry) -> None:
        self._objects[entry.anchor_id] = entry.obj
        self.anchor_table.add(entry)
        self.corr_index.add(entry)
        self.rarity_index.add(entry)
        self.intent_index.add(entry)
        self.aspect_index.add(entry)

    def _next_insertion_order(self) -> int:
        order = self._next_order
        self._next_order += 1
        return order


def _cosine_similarity(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float:
    if vec_a is None or vec_b is None:
        return 0.0
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)
