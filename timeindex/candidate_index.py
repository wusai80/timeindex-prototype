"""Candidate index stubs for the TimeIndex prototype."""

from __future__ import annotations

from collections.abc import Sequence

from .event import ChainSummary, DecisionIntent, EventRecord


class SkipCandidateIndex:
    """Stub bounded candidate index for long-range evidence anchors."""

    def add_event_anchor(self, record: EventRecord, intent: DecisionIntent | None = None) -> None:
        raise NotImplementedError("Anchor indexing is not implemented yet.")

    def add_chain_anchor(self, summary: ChainSummary, intent: DecisionIntent | None = None) -> None:
        raise NotImplementedError("Anchor indexing is not implemented yet.")

    def retrieve(self, record: EventRecord, intent: DecisionIntent | None = None) -> Sequence[str]:
        raise NotImplementedError("Skip-candidate retrieval is not implemented yet.")
