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
        self._active_start = 0
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

    def __len__(self) -> int:
        return len(self._records)

    def expire(self, event_ids: Iterable[str] | None = None, max_size: int | None = None) -> list[str]:
        """Expire records by id or active-size budget and purge them from memory."""

        expired_ids: list[str] = []
        expired_lookup: set[str] = set()
        pending = {str(event_id) for event_id in (event_ids or ())}

        if max_size is not None and max_size >= 0:
            overflow = max(0, len(self._records) - max_size)
            while overflow > 0 and self._active_start < len(self._insertion_order):
                event_id = self._insertion_order[self._active_start]
                self._active_start += 1
                record = self._records.get(event_id)
                if record is None or record.metadata.expired:
                    continue
                record.metadata.expired = True
                expired_ids.append(event_id)
                expired_lookup.add(event_id)
                del self._records[event_id]
                overflow -= 1

        for event_id in pending:
            if event_id in expired_lookup:
                continue
            record = self._records.get(event_id)
            if record is None or record.metadata.expired:
                continue
            record.metadata.expired = True
            expired_ids.append(event_id)
            expired_lookup.add(event_id)
            del self._records[event_id]

        self._compact_insertion_order()
        return expired_ids

    def list(self) -> Sequence[EventRecord]:
        return [
            record
            for event_id in self._insertion_order[self._active_start :]
            for record in [self._records.get(event_id)]
            if record is not None and not record.metadata.expired
        ]

    def _compact_insertion_order(self) -> None:
        if self._active_start <= 0:
            return
        remaining = len(self._insertion_order) - self._active_start
        if self._active_start < 1024 and self._active_start < remaining:
            return
        self._insertion_order = self._insertion_order[self._active_start :]
        self._active_start = 0


class KeyDirectory:
    """Maps lookup keys to bounded recent event ids."""

    def __init__(self, posting_list_size: int = 100) -> None:
        self.posting_list_size = posting_list_size
        self._postings: dict[str, deque[str]] = {}

    def add_event(self, event_id: str, keys: Sequence[str]) -> None:
        for key in sorted(set(keys)):
            postings = self._postings.get(key)
            if postings is None:
                postings = deque(maxlen=self.posting_list_size)
                self._postings[key] = postings
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


class EntityDirectory:
    """Specialized newest-first postings for entity and flow lookups."""

    def __init__(self, posting_list_size: int = 100) -> None:
        self.posting_list_size = posting_list_size
        self._source: dict[str, deque[str]] = {}
        self._destination: dict[str, deque[str]] = {}
        self._participant: dict[str, deque[str]] = {}
        self._flow_pair: dict[str, deque[str]] = {}

    def add_event(
        self,
        event_id: str,
        source_entities: Sequence[str],
        destination_entities: Sequence[str],
    ) -> None:
        source_values = sorted(set(source_entities))
        destination_values = sorted(set(destination_entities))
        participant_values = sorted(set(source_values) | set(destination_values))

        for entity in source_values:
            self._append(self._source, entity, event_id)
        for entity in destination_values:
            self._append(self._destination, entity, event_id)
        for entity in participant_values:
            self._append(self._participant, entity, event_id)
        for source in source_values:
            for destination in destination_values:
                if source == destination:
                    continue
                self._append(self._flow_pair, f"{source}->{destination}", event_id)

    def recent_sources(self, entity: str) -> Sequence[str]:
        return list(reversed(self._source.get(entity, ())))

    def recent_destinations(self, entity: str) -> Sequence[str]:
        return list(reversed(self._destination.get(entity, ())))

    def recent_participants(self, entity: str) -> Sequence[str]:
        return list(reversed(self._participant.get(entity, ())))

    def recent_flow_pair(self, source_entity: str, destination_entity: str) -> Sequence[str]:
        return list(reversed(self._flow_pair.get(f"{source_entity}->{destination_entity}", ())))

    def expire(self, expired_event_ids: Iterable[str]) -> None:
        expired = set(expired_event_ids)
        if not expired:
            return
        for mapping in (self._source, self._destination, self._participant, self._flow_pair):
            for key, postings in list(mapping.items()):
                filtered = [event_id for event_id in postings if event_id not in expired]
                if not filtered:
                    del mapping[key]
                    continue
                mapping[key] = deque(filtered, maxlen=self.posting_list_size)

    def _append(self, mapping: dict[str, deque[str]], key: str, event_id: str) -> None:
        postings = mapping.get(key)
        if postings is None:
            postings = deque(maxlen=self.posting_list_size)
            mapping[key] = postings
        postings.append(event_id)


class EdgeStore:
    """Stores ordinary dependency links."""

    def __init__(self, fan_in: int = 5, maintain_outgoing_links: bool = True) -> None:
        self.fan_in = fan_in
        self.maintain_outgoing_links = maintain_outgoing_links
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

        if self.maintain_outgoing_links:
            self._rebuild_outgoing_for_successor(link.successor_id)
            for removed_link in removed:
                self._remove_from_outgoing(removed_link)

    def incoming(self, event_id: str) -> Sequence[OrdinaryLink]:
        return list(self._incoming.get(event_id, ()))

    def outgoing(self, event_id: str) -> Sequence[OrdinaryLink]:
        return list(self._outgoing.get(event_id, ()))

    def expire(self, expired_event_ids: Iterable[str]) -> None:
        expired = set(expired_event_ids)
        if not expired:
            return

        for successor_id in list(self._incoming):
            if successor_id in expired:
                del self._incoming[successor_id]
                continue
            filtered = [
                link for link in self._incoming[successor_id]
                if link.predecessor_id not in expired and link.successor_id not in expired
            ]
            if filtered:
                self._incoming[successor_id] = filtered
            else:
                del self._incoming[successor_id]

        if self.maintain_outgoing_links:
            for predecessor_id in list(self._outgoing):
                if predecessor_id in expired:
                    del self._outgoing[predecessor_id]
                    continue
                filtered = [
                    link for link in self._outgoing[predecessor_id]
                    if link.predecessor_id not in expired and link.successor_id not in expired
                ]
                if filtered:
                    self._outgoing[predecessor_id] = filtered
                else:
                    del self._outgoing[predecessor_id]

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
        self._by_tail: dict[str, list[ChainSummary]] = defaultdict(list)
        self._families_by_tail: dict[str, set[str]] = defaultdict(set)

    def add(self, summary: ChainSummary) -> None:
        key = (summary.tail_id, summary.family)
        summaries = [item for item in self._by_tail_family[key] if item.chain_id != summary.chain_id]
        summaries.append(summary)
        summaries.sort(key=lambda item: (-item.dependency_confidence, item.chain_id))
        self._by_tail_family[key] = summaries[: self.summaries_per_family]
        self._families_by_tail[summary.tail_id].add(summary.family)
        self._refresh_tail(summary.tail_id)

    def get_for_tail(self, event_id: str) -> Sequence[ChainSummary]:
        by_tail = getattr(self, "_by_tail", None)
        if isinstance(by_tail, dict):
            if event_id in by_tail:
                return list(by_tail[event_id])
        families_by_tail = getattr(self, "_families_by_tail", None)
        if isinstance(families_by_tail, dict) and event_id not in families_by_tail:
            return []
        summaries: list[ChainSummary] = []
        for (tail_id, _family), items in self._by_tail_family.items():
            if tail_id == event_id:
                summaries.extend(items)
        summaries.sort(key=lambda item: (item.family, -item.dependency_confidence, item.chain_id))
        return summaries

    def expire(self, expired_event_ids: Iterable[str]) -> None:
        expired = set(expired_event_ids)
        if not expired:
            return
        for key in list(self._by_tail_family):
            tail_id, _family = key
            if tail_id in expired:
                del self._by_tail_family[key]
                continue
            filtered = [
                summary
                for summary in self._by_tail_family[key]
                if summary.head_id not in expired
                and summary.tail_id not in expired
                and not (set(summary.representative_event_ids) & expired)
            ]
            if filtered:
                self._by_tail_family[key] = filtered[: self.summaries_per_family]
                self._refresh_tail(tail_id)
            else:
                del self._by_tail_family[key]
                families = self._families_by_tail.get(tail_id)
                if families is not None:
                    families.discard(_family)
                    if not families:
                        self._families_by_tail.pop(tail_id, None)
                self._refresh_tail(tail_id)

    def _refresh_tail(self, tail_id: str) -> None:
        if not hasattr(self, "_by_tail"):
            self._by_tail = defaultdict(list)
        if not hasattr(self, "_families_by_tail"):
            self._families_by_tail = defaultdict(set)
        summaries: list[ChainSummary] = []
        for family in self._families_by_tail.get(tail_id, ()):
            summaries.extend(self._by_tail_family.get((tail_id, family), ()))
        summaries.sort(key=lambda item: (item.family, -item.dependency_confidence, item.chain_id))
        if summaries:
            self._by_tail[tail_id] = summaries
        else:
            self._by_tail.pop(tail_id, None)


class SkipLinkStore:
    """Stores incoming and outgoing skip links."""

    def __init__(self, fan_in: int = 3, maintain_outgoing_links: bool = True) -> None:
        self.fan_in = fan_in
        self.maintain_outgoing_links = maintain_outgoing_links
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

        if self.maintain_outgoing_links:
            self._rebuild_outgoing_for_target(link.to_id)
            for removed_link in removed:
                self._remove_from_outgoing(removed_link)

    def incoming(self, event_id: str) -> Sequence[SkipLink]:
        return list(self._incoming.get(event_id, ()))

    def outgoing(self, event_id: str) -> Sequence[SkipLink]:
        return list(self._outgoing.get(event_id, ()))

    def expire(self, expired_event_ids: Iterable[str]) -> None:
        expired = set(expired_event_ids)
        if not expired:
            return

        for target_id in list(self._incoming):
            if target_id in expired:
                del self._incoming[target_id]
                continue
            filtered = [
                link for link in self._incoming[target_id]
                if link.from_id not in expired
                and link.to_id not in expired
                and not (set(link.representative_event_ids) & expired)
            ]
            if filtered:
                self._incoming[target_id] = filtered
            else:
                del self._incoming[target_id]

        if self.maintain_outgoing_links:
            for source_id in list(self._outgoing):
                if source_id in expired:
                    del self._outgoing[source_id]
                    continue
                filtered = [
                    link for link in self._outgoing[source_id]
                    if link.from_id not in expired
                    and link.to_id not in expired
                    and not (set(link.representative_event_ids) & expired)
                ]
                if filtered:
                    self._outgoing[source_id] = filtered
                else:
                    del self._outgoing[source_id]

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
