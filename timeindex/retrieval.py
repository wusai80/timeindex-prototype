"""Retrieval stubs for the TimeIndex prototype."""

from __future__ import annotations

from collections.abc import Sequence

from .config import RetrievalConfig
from .event import EventQuery, EvidenceObject
from .stores import ChainStore, EdgeStore, EventStore, SkipLinkStore


class DualFrontierRetriever:
    """Stub dual-frontier retriever following Algorithm 2."""

    def __init__(
        self,
        event_store: EventStore,
        edge_store: EdgeStore,
        chain_store: ChainStore,
        skip_link_store: SkipLinkStore,
        config: RetrievalConfig,
    ) -> None:
        self.event_store = event_store
        self.edge_store = edge_store
        self.chain_store = chain_store
        self.skip_link_store = skip_link_store
        self.config = config

    def retrieve(self, query: EventQuery) -> Sequence[EvidenceObject]:
        raise NotImplementedError("Dual-frontier retrieval is not implemented yet.")
