"""In-memory store stubs for the TimeIndex prototype."""

from __future__ import annotations

from collections.abc import Sequence

from .event import ChainSummary, EventRecord, OrdinaryLink, SkipLink


class EventStore:
    """Stores event records keyed by event id."""

    def add(self, record: EventRecord) -> None:
        raise NotImplementedError("Event storage is not implemented yet.")

    def get(self, event_id: str) -> EventRecord | None:
        raise NotImplementedError("Event storage is not implemented yet.")

    def list(self) -> Sequence[EventRecord]:
        raise NotImplementedError("Event storage is not implemented yet.")


class KeyDirectory:
    """Maps lookup keys to bounded recent event ids."""

    def add(self, event_id: str, keys: Sequence[str]) -> None:
        raise NotImplementedError("Key directory maintenance is not implemented yet.")

    def lookup(self, key: str) -> Sequence[str]:
        raise NotImplementedError("Key directory lookup is not implemented yet.")


class EdgeStore:
    """Stores ordinary dependency links."""

    def add(self, link: OrdinaryLink) -> None:
        raise NotImplementedError("Ordinary link storage is not implemented yet.")

    def incoming(self, event_id: str) -> Sequence[OrdinaryLink]:
        raise NotImplementedError("Ordinary link lookup is not implemented yet.")

    def outgoing(self, event_id: str) -> Sequence[OrdinaryLink]:
        raise NotImplementedError("Ordinary link lookup is not implemented yet.")


class ChainStore:
    """Stores bounded chain summaries per tail event and family."""

    def add(self, summary: ChainSummary) -> None:
        raise NotImplementedError("Chain summary storage is not implemented yet.")

    def get_for_tail(self, event_id: str) -> Sequence[ChainSummary]:
        raise NotImplementedError("Chain summary lookup is not implemented yet.")


class SkipLinkStore:
    """Stores incoming and outgoing skip links."""

    def add(self, link: SkipLink) -> None:
        raise NotImplementedError("Skip-link storage is not implemented yet.")

    def incoming(self, event_id: str) -> Sequence[SkipLink]:
        raise NotImplementedError("Skip-link lookup is not implemented yet.")

    def outgoing(self, event_id: str) -> Sequence[SkipLink]:
        raise NotImplementedError("Skip-link lookup is not implemented yet.")
