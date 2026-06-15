"""Construction stubs for the TimeIndex prototype."""

from __future__ import annotations

from .candidate_index import SkipCandidateIndex
from .config import ConstructionConfig
from .event import DecisionIntent, Event
from .stores import ChainStore, EdgeStore, EventStore, KeyDirectory, SkipLinkStore


class IndexConstructor:
    """Stub online construction orchestrator following Algorithm 1."""

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
