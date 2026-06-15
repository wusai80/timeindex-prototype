"""Scoring stubs for the TimeIndex prototype."""

from __future__ import annotations

from collections.abc import Sequence

from .config import ScoringConfig
from .event import ChainSummary, DecisionIntent, EventQuery, EventRecord, EvidenceObject, SkipLink


class PrototypeScorer:
    """Stub scorer for local links, anchors, skips, and retrieval utility."""

    def __init__(self, config: ScoringConfig) -> None:
        self.config = config

    def score_local_dependency(self, predecessor: EventRecord, current: EventRecord) -> float:
        raise NotImplementedError("Local dependency scoring is not implemented yet.")

    def score_anchor(self, candidate: EventRecord | ChainSummary, intent: DecisionIntent) -> float:
        raise NotImplementedError("Anchor scoring is not implemented yet.")

    def score_skip(self, anchor: EventRecord | ChainSummary, query: EventRecord, intent: DecisionIntent) -> float:
        raise NotImplementedError("Skip scoring is not implemented yet.")

    def marginal_utility(
        self,
        candidate: EvidenceObject | SkipLink | ChainSummary,
        selected: Sequence[EvidenceObject],
        query: EventQuery,
    ) -> float:
        raise NotImplementedError("Retrieval utility scoring is not implemented yet.")
