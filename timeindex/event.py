"""Shared dataclasses for the TimeIndex prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose
from types import MappingProxyType
from typing import Any

import numpy as np


@dataclass(slots=True)
class Event:
    """Raw event schema from the prototype specification."""

    event_id: str
    time: str | int | float
    event_type: str
    attrs: dict[str, Any] = field(default_factory=dict)
    ctx: dict[str, Any] = field(default_factory=dict)
    text: str | None = None
    label: str | None = None

    def __getstate__(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "time": self.time,
            "event_type": self.event_type,
            "attrs": dict(self.attrs),
            "ctx": dict(self.ctx),
            "text": self.text,
            "label": self.label,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.event_id = state["event_id"]
        self.time = state["time"]
        self.event_type = state["event_type"]
        self.attrs = MappingProxyType(dict(state["attrs"]))
        self.ctx = MappingProxyType(dict(state["ctx"]))
        self.text = state["text"]
        self.label = state["label"]


@dataclass(slots=True)
class EventMetadata:
    """Per-event metadata tracked by the online index."""

    rarity: float = 0.0
    surprise: float = 0.0
    insertion_order: int | None = None
    expired: bool = False
    labels: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EventRecord:
    """Event plus extracted retrieval representation."""

    event: Event
    lookup_keys: set[str] = field(default_factory=set)
    sketch: np.ndarray | None = None
    aspects: set[str] = field(default_factory=set)
    metadata: EventMetadata = field(default_factory=EventMetadata)
    entity_keys: frozenset[str] = field(init=False)
    attribute_keys: frozenset[str] = field(init=False)
    context_keys: frozenset[str] = field(init=False)
    source_entities: frozenset[str] = field(init=False)
    destination_entities: frozenset[str] = field(init=False)
    participant_entities: frozenset[str] = field(init=False)
    sketch_norm: float = field(init=False)
    sketch_is_normalized: bool = field(init=False)

    def __post_init__(self) -> None:
        self.lookup_keys = frozenset(self.lookup_keys)
        self.aspects = frozenset(self.aspects)
        self.event.attrs = MappingProxyType(dict(self.event.attrs))
        self.event.ctx = MappingProxyType(dict(self.event.ctx))

        self.entity_keys = frozenset(key for key in self.lookup_keys if key.startswith("entity:"))
        self.attribute_keys = frozenset(key for key in self.lookup_keys if key.startswith(("attr:", "attr_bin:")))
        self.context_keys = frozenset(key for key in self.lookup_keys if key.startswith(("ctx:", "type:", "time_block:")))

        self.source_entities = frozenset(_event_entities(self.event.attrs, role="source"))
        self.destination_entities = frozenset(_event_entities(self.event.attrs, role="destination"))
        self.participant_entities = self.source_entities | self.destination_entities

        if self.sketch is None:
            self.sketch_norm = 0.0
            self.sketch_is_normalized = False
            return

        self.sketch.setflags(write=False)
        self.sketch_norm = float(np.linalg.norm(self.sketch))
        self.sketch_is_normalized = isclose(self.sketch_norm, 1.0, rel_tol=1e-9, abs_tol=1e-9)


@dataclass(slots=True)
class DecisionIntent:
    """Intent definition used at construction and retrieval time."""

    aspects: set[str] = field(default_factory=set)
    aspect_weights: dict[str, float] = field(default_factory=dict)
    coverage_target: int | float | None = None
    name: str | None = None


@dataclass(slots=True)
class EventQuery:
    """Query event, intent, and budget."""

    event: Event
    intent: DecisionIntent = field(default_factory=DecisionIntent)
    budget: int | float = 10


@dataclass(slots=True)
class EvidenceObject:
    """Returned evidence unit for the downstream agent."""

    object_id: str
    event_ids: list[str] = field(default_factory=list)
    aspects: set[str] = field(default_factory=set)
    summary: str = ""
    cost: float = 0.0


@dataclass(slots=True)
class OrdinaryLink:
    """Locally relevant predecessor edge."""

    predecessor_id: str
    successor_id: str
    score: float


@dataclass(slots=True)
class ChainSummary:
    """Compact summary of a temporal evidence chain."""

    chain_id: str
    family: str
    head_id: str
    tail_id: str
    representative_event_ids: list[str] = field(default_factory=list)
    source_entities: set[str] = field(default_factory=set)
    destination_entities: set[str] = field(default_factory=set)
    aspects: set[str] = field(default_factory=set)
    dependency_confidence: float = 0.0
    summary: str = ""
    cost: float = 0.0


@dataclass(slots=True)
class SkipLink:
    """Value-aware shortcut to a distant evidence anchor."""

    from_id: str
    to_id: str
    skip_value: float
    segment_confidence: float = 0.0
    source_entities: set[str] = field(default_factory=set)
    destination_entities: set[str] = field(default_factory=set)
    aspects: set[str] = field(default_factory=set)
    summary: str = ""
    representative_event_ids: list[str] = field(default_factory=list)
    cost: float = 0.0


_SOURCE_ENTITY_PREFIXES = ("src_", "source_", "from_", "origin_", "sender_")
_DESTINATION_ENTITY_PREFIXES = ("dst_", "destination_", "to_", "beneficiary_", "counterparty_", "receiver_", "recipient_", "target_")
_SOURCE_ENTITY_FIELDS = {
    "account_id",
    "src_account",
    "source_account",
    "from_account",
    "origin_account",
    "sender_account",
    "src_user",
    "source_user",
}
_DESTINATION_ENTITY_FIELDS = {
    "dst_account",
    "destination_account",
    "to_account",
    "beneficiary_account",
    "beneficiary_id",
    "counterparty_account",
    "recipient_account",
    "receiver_account",
    "target_account",
    "dst_user",
    "target_user",
}


def _event_entities(attrs: dict[str, Any], role: str) -> set[str]:
    values: set[str] = set()
    known_fields = _SOURCE_ENTITY_FIELDS if role == "source" else _DESTINATION_ENTITY_FIELDS
    for field_name, value in attrs.items():
        if value in (None, ""):
            continue
        normalized_name = field_name.lower()
        if normalized_name in known_fields:
            values.add(str(value).strip().lower())
    return values
