"""Shared protocols for TimeIndex components."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .event import ChainSummary, DecisionIntent, Event, EventQuery, EventRecord, EvidenceObject, SkipLink


class EventRepresentationExtractor(Protocol):
    """Extract a retrieval representation for a raw event."""

    def extract(self, event: Event) -> EventRecord:
        """Build lookup keys, sketch, and aspects for an event."""


class DependencyScorer(Protocol):
    """Score local predecessor links."""

    def score_local_dependency(self, predecessor: EventRecord, current: EventRecord) -> float:
        """Return the local dependency score for an event pair."""


class SkipScorer(Protocol):
    """Score value-aware long-range anchors."""

    def score_skip(self, anchor: EventRecord | ChainSummary, query: EventRecord, intent: DecisionIntent) -> float:
        """Return the skip value for an anchor and target event."""


class RetrievalScorer(Protocol):
    """Score evidence objects during dual-frontier retrieval."""

    def marginal_utility(
        self,
        candidate: EvidenceObject | SkipLink | ChainSummary,
        selected: Sequence[EvidenceObject],
        query: EventQuery,
    ) -> float:
        """Return the estimated marginal utility of a candidate."""


class Retriever(Protocol):
    """Protocol for query-time evidence retrieval."""

    def retrieve(self, query: EventQuery) -> Sequence[EvidenceObject]:
        """Return budgeted evidence objects for a query."""
