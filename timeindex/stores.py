"""In-memory stores for the TimeIndex prototype."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import replace

from .event import ChainSummary, EventRecord, OrdinaryLink, SkipLink


class EventStore:
    """Stores event records keyed by event id."""

    def __init__(self) -> None:
        self._records: dict[str, EventRecord] = {}
        self._insertion_order: list[str] = []
        self._next_order = 0

    def insert(self, record: EventRecord) -> None:
        """Insert or replace a record while preserving insertion order."""

        event_id = record.event.event_id
        metadata = record.metadata
        if metadata.insertion_order is None:
            metadata = replace(metadata, insertion_order=self._next_order)
            record = replace(record, metadata=metadata)
            self._next_order += 1

        if event_id not in self._records:
            self._insertion_order.append(event_id)
        self._records[event_id] = record

    def add(self, record: EventRecord) -> None:
        self.insert(record)

    def get(self, event_id: str) -> EventRecord | None:
        record = self._records.get(event_id)
        if record is None or record.metadata.expired:
            return None
        return record

    def contains(self, event_id: str) -> bool:
        return self.get(event_id) is not None

    def is_valid(self, event_id: str) -> bool:
        return self.contains(event_id)

    def expire(self, event_ids: Iterable[str] | None = None, max_size: int | None = None) -> list[str]:
        """Mark records as expired by id or by active-size budget."""

        expired_ids: list[str] = []
        pending = set(event_ids or ())

        if max_size is not None and max_size >= 0:
            active_ids = [event_id for event_id in self._insertion_order if self.is_valid(event_id)]
            overflow = max(0, len(active_ids) - max_size)
            pending.update(active_ids[:overflow])

        for event_id in self._insertion_order:
            if event_id not in pending:
                continue
            record = self._records.get(event_id)
            if record is None or record.metadata.expired:
                continue
            record.metadata.expired = True
            expired_ids.append(event_id)

        return expired_ids

    def list(self) -> Sequence[EventRecord]:
        return [self._records[event_id] for event_id in self._insertion_order if self.is_valid(event_id)]


class KeyDirectory:
    """Maps lookup keys to bounded recent event ids."""

    def __init__(self, posting_list_size: int = 100) -> None:
        self.posting_list_size = posting_list_size
        self._postings: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=self.posting_list_size))

    def add_event(self, event_id: str, keys: Sequence[str]) -> None:
        for key in sorted(set(keys)):
            postings = self._postings[key]
            if event_id in postings:
                postings.remove(event_id)
            postings.append(event_id)

    def add(self, event_id: str, keys: Sequence[str]) -> None:
        self.add_event(event_id, keys)

    def lookup(self, key: str) -> Sequence[str]:
        return list(reversed(self._postings.get(key, ())))

    def lookup_keys(self, keys: Sequence[str]) -> Sequence[str]:
        seen: set[str] = set()
        ranked_ids: list[str] = []
        for key in sorted(set(keys)):
            for event_id in reversed(self._postings.get(key, ())):
                if event_id in seen:
                    continue
                seen.add(event_id)
                ranked_ids.append(event_id)
        return ranked_ids

    def expire(self, expired_event_ids: Iterable[str]) -> None:
        expired = set(expired_event_ids)
        if not expired:
            return
        for key, postings in list(self._postings.items()):
            filtered = [event_id for event_id in postings if event_id not in expired]
            if not filtered:
                del self._postings[key]
                continue
            self._postings[key] = deque(filtered, maxlen=self.posting_list_size)


class EdgeStore:
    """Stores ordinary dependency links."""

    def __init__(self, fan_in: int = 5) -> None:
        self.fan_in = fan_in
        self._incoming: dict[str, list[OrdinaryLink]] = defaultdict(list)
        self._outgoing: dict[str, list[OrdinaryLink]] = defaultdict(list)

    def add(self, link: OrdinaryLink) -> None:
        if link.predecessor_id == link.successor_id:
            return

        incoming = [item for item in self._incoming[link.successor_id] if item.predecessor_id != link.predecessor_id]
        incoming.append(link)
        incoming.sort(key=lambda item: (-item.score, item.predecessor_id, item.successor_id))
        kept = incoming[: self.fan_in]
        removed = incoming[self.fan_in :]
        self._incoming[link.successor_id] = kept

        self._rebuild_outgoing_for_successor(link.successor_id)
        for removed_link in removed:
            self._remove_from_outgoing(removed_link)

    def incoming(self, event_id: str) -> Sequence[OrdinaryLink]:
        return list(self._incoming.get(event_id, ()))

    def outgoing(self, event_id: str) -> Sequence[OrdinaryLink]:
        return list(self._outgoing.get(event_id, ()))

    def _rebuild_outgoing_for_successor(self, successor_id: str) -> None:
        current_links = self._incoming.get(successor_id, ())
        predecessor_ids = {link.predecessor_id for link in current_links}
        for predecessor_id in predecessor_ids:
            outgoing = [link for link in self._outgoing.get(predecessor_id, ()) if link.successor_id != successor_id]
            outgoing.extend(
                link for link in current_links if link.predecessor_id == predecessor_id and link.successor_id == successor_id
            )
            outgoing.sort(key=lambda item: (item.successor_id, -item.score, item.predecessor_id))
            self._outgoing[predecessor_id] = outgoing

    def _remove_from_outgoing(self, link: OrdinaryLink) -> None:
        outgoing = [item for item in self._outgoing.get(link.predecessor_id, ()) if item.successor_id != link.successor_id]
        if outgoing:
            self._outgoing[link.predecessor_id] = outgoing
        elif link.predecessor_id in self._outgoing:
            del self._outgoing[link.predecessor_id]


class ChainStore:
    """Stores bounded chain summaries per tail event and family."""

    def __init__(self, summaries_per_family: int = 5) -> None:
        self.summaries_per_family = summaries_per_family
        self._by_tail_family: dict[tuple[str, str], list[ChainSummary]] = defaultdict(list)

    def add(self, summary: ChainSummary) -> None:
        key = (summary.tail_id, summary.family)
        summaries = [item for item in self._by_tail_family[key] if item.chain_id != summary.chain_id]
        summaries.append(summary)
        summaries.sort(key=lambda item: (-item.dependency_confidence, item.chain_id))
        self._by_tail_family[key] = summaries[: self.summaries_per_family]

    def get_for_tail(self, event_id: str) -> Sequence[ChainSummary]:
        summaries: list[ChainSummary] = []
        for (tail_id, _family), items in sorted(self._by_tail_family.items()):
            if tail_id == event_id:
                summaries.extend(items)
        summaries.sort(key=lambda item: (item.family, -item.dependency_confidence, item.chain_id))
        return summaries


class SkipLinkStore:
    """Stores incoming and outgoing skip links."""

    def __init__(self, fan_in: int = 3) -> None:
        self.fan_in = fan_in
        self._incoming: dict[str, list[SkipLink]] = defaultdict(list)
        self._outgoing: dict[str, list[SkipLink]] = defaultdict(list)

    def add(self, link: SkipLink) -> None:
        if link.from_id == link.to_id:
            return

        incoming = [item for item in self._incoming[link.to_id] if item.from_id != link.from_id]
        incoming.append(link)
        incoming.sort(key=lambda item: (-item.skip_value, item.from_id, item.to_id))
        kept = incoming[: self.fan_in]
        removed = incoming[self.fan_in :]
        self._incoming[link.to_id] = kept

        self._rebuild_outgoing_for_target(link.to_id)
        for removed_link in removed:
            self._remove_from_outgoing(removed_link)

    def incoming(self, event_id: str) -> Sequence[SkipLink]:
        return list(self._incoming.get(event_id, ()))

    def outgoing(self, event_id: str) -> Sequence[SkipLink]:
        return list(self._outgoing.get(event_id, ()))

    def _rebuild_outgoing_for_target(self, target_id: str) -> None:
        current_links = self._incoming.get(target_id, ())
        source_ids = {link.from_id for link in current_links}
        for source_id in source_ids:
            outgoing = [link for link in self._outgoing.get(source_id, ()) if link.to_id != target_id]
            outgoing.extend(link for link in current_links if link.from_id == source_id and link.to_id == target_id)
            outgoing.sort(key=lambda item: (item.to_id, -item.skip_value, item.from_id))
            self._outgoing[source_id] = outgoing

    def _remove_from_outgoing(self, link: SkipLink) -> None:
        outgoing = [item for item in self._outgoing.get(link.from_id, ()) if item.to_id != link.to_id]
        if outgoing:
            self._outgoing[link.from_id] = outgoing
        elif link.from_id in self._outgoing:
            del self._outgoing[link.from_id]
