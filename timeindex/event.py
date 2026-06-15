"""Shared dataclasses for the TimeIndex prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    aspects: set[str] = field(default_factory=set)
    summary: str = ""
    representative_event_ids: list[str] = field(default_factory=list)
    cost: float = 0.0
