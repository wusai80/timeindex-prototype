from __future__ import annotations

from collections import defaultdict, deque

from timeindex.config import TimeIndexConfig
from timeindex.construction import TimeIndex
from timeindex.event import ChainSummary, DecisionIntent, Event, EventRecord, OrdinaryLink, SkipLink
import timeindex.construction as construction_module
import timeindex.extractors as extractors_module
import timeindex.scoring as scoring_module


class FakeEventStore:
    def __init__(self, config=None) -> None:
        self.records: dict[str, EventRecord] = {}
        self.order: deque[str] = deque()
        self.limit = getattr(config, "active_history_size", 10_000)

    def insert(self, record: EventRecord) -> None:
        event_id = record.event.event_id
        self.records[event_id] = record
        self.order.append(event_id)

    def get(self, event_id: str) -> EventRecord | None:
        return self.records.get(event_id)

    def is_valid(self, event_id: str) -> bool:
        record = self.records.get(event_id)
        return record is not None and not record.metadata.expired

    def expire(self, active_history_size: int) -> None:
        self.limit = active_history_size
        while len(self.order) > active_history_size:
            oldest = self.order.popleft()
            record = self.records.get(oldest)
            if record is not None:
                record.metadata.expired = True


class FakeKeyDirectory:
    def __init__(self, config=None) -> None:
        self.limit = getattr(config, "posting_list_size", 100)
        self.postings: dict[str, deque[str]] = defaultdict(deque)

    def add_event(self, event_id: str, keys) -> None:
        for key in sorted(keys):
            posting = self.postings[key]
            posting.append(event_id)
            while len(posting) > self.limit:
                posting.popleft()

    def lookup_keys(self, keys) -> list[str]:
        results: list[str] = []
        for key in sorted(keys):
            results.extend(self.postings.get(key, ()))
        return results

    def expire(self, active_history_size: int) -> None:
        return None


class FakeEdgeStore:
    def __init__(self, config=None) -> None:
        self.fan_in = getattr(config, "ordinary_fan_in", 5)
        self.incoming_map: dict[str, list[OrdinaryLink]] = defaultdict(list)
        self.outgoing_map: dict[str, list[OrdinaryLink]] = defaultdict(list)

    def insert(self, link: OrdinaryLink) -> None:
        incoming = self.incoming_map[link.successor_id]
        outgoing = self.outgoing_map[link.predecessor_id]
        incoming.append(link)
        outgoing.append(link)
        incoming.sort(key=lambda item: (-item.score, item.predecessor_id))
        if len(incoming) > self.fan_in:
            removed = incoming.pop()
            self.outgoing_map[removed.predecessor_id] = [
                item for item in self.outgoing_map[removed.predecessor_id] if item != removed
            ]

    def incoming(self, event_id: str) -> list[OrdinaryLink]:
        return list(self.incoming_map.get(event_id, ()))


class FakeChainStore:
    def __init__(self, config=None) -> None:
        self.limit = getattr(config, "chain_summaries_per_family", 5)
        self.by_tail_family: dict[tuple[str, str], list[ChainSummary]] = defaultdict(list)

    def add(self, summary: ChainSummary) -> None:
        key = (summary.tail_id, summary.family)
        bucket = self.by_tail_family[key]
        bucket.append(summary)
        bucket.sort(key=lambda item: (-item.dependency_confidence, item.chain_id))
        del bucket[self.limit :]

    def get_for_tail(self, event_id: str) -> list[ChainSummary]:
        summaries: list[ChainSummary] = []
        for (tail_id, _family), items in self.by_tail_family.items():
            if tail_id == event_id:
                summaries.extend(items)
        summaries.sort(key=lambda item: (-item.dependency_confidence, item.chain_id))
        return summaries


class FakeSkipLinkStore:
    def __init__(self, config=None) -> None:
        self.fan_in = getattr(config, "skip_fan_in", 3)
        self.incoming_map: dict[str, list[SkipLink]] = defaultdict(list)
        self.outgoing_map: dict[str, list[SkipLink]] = defaultdict(list)

    def insert(self, link: SkipLink) -> None:
        incoming = self.incoming_map[link.to_id]
        outgoing = self.outgoing_map[link.from_id]
        incoming.append(link)
        outgoing.append(link)
        incoming.sort(key=lambda item: (-item.skip_value, item.from_id))
        if len(incoming) > self.fan_in:
            removed = incoming.pop()
            self.outgoing_map[removed.from_id] = [
                item for item in self.outgoing_map[removed.from_id] if item != removed
            ]

    def incoming(self, event_id: str) -> list[SkipLink]:
        return list(self.incoming_map.get(event_id, ()))


class FakeSkipCandidateIndex:
    def __init__(self, config=None) -> None:
        self.event_anchors: list[EventRecord] = []
        self.chain_anchors: list[ChainSummary] = []

    def add_event_anchor(self, record: EventRecord, intent: DecisionIntent | None = None) -> None:
        self.event_anchors.append(record)

    def add_chain_anchor(self, summary: ChainSummary, intent: DecisionIntent | None = None) -> None:
        self.chain_anchors.append(summary)

    def get_skip_candidates(
        self,
        event: EventRecord,
        intent: DecisionIntent | None,
        ordinary_predecessors: list[str],
    ) -> list[EventRecord | ChainSummary]:
        ordinary_ids = set(ordinary_predecessors)
        candidates: list[EventRecord | ChainSummary] = []
        for record in self.event_anchors:
            if record.event.event_id != event.event.event_id and record.event.event_id not in ordinary_ids:
                candidates.append(record)
        for summary in self.chain_anchors:
            if summary.tail_id != event.event.event_id and summary.tail_id not in ordinary_ids:
                candidates.append(summary)
        return candidates


def fake_featurize_event(event: Event, _config) -> EventRecord:
    account = str(event.attrs.get("account_id", ""))
    beneficiary = str(event.attrs.get("beneficiary_id", ""))
    lookup_keys = {
        f"type:{event.event_type}",
        f"account:{account}",
    }
    if beneficiary:
        lookup_keys.add(f"beneficiary:{beneficiary}")

    aspects = set(event.attrs.get("aspects", ()))
    if not aspects:
        aspects = {"generic_evidence"}

    return EventRecord(
        event=event,
        lookup_keys=lookup_keys,
        aspects=aspects,
    )


def fake_dependency_score(candidate: EventRecord, target: EventRecord, config) -> float:
    overlap = len(candidate.lookup_keys & target.lookup_keys)
    if overlap == 0:
        return 0.0
    base = 0.35 + 0.20 * min(overlap, 2)
    if candidate.event.attrs.get("role") == "routine":
        base -= 0.05
    recency_bonus = 0.01 * float(candidate.event.time)
    return min(1.0, base + recency_bonus)


def fake_skip_score(anchor, target: EventRecord, intent, ordinary_predecessors, config) -> float:
    anchor_id = getattr(anchor, "tail_id", None)
    if anchor_id is None and isinstance(anchor, EventRecord):
        anchor_id = anchor.event.event_id
    if anchor_id in ordinary_predecessors:
        return 0.0
    if isinstance(anchor, ChainSummary):
        return min(1.0, 0.55 + 0.10 * len(anchor.aspects))
    if isinstance(anchor, EventRecord) and anchor.event.attrs.get("anchor", False):
        return 0.80
    return 0.15


def fake_anchor_score(obj, intent, existing_anchors) -> float:
    if isinstance(obj, EventRecord):
        return 0.80 if obj.event.attrs.get("anchor", False) else 0.20
    return min(1.0, 0.50 + 0.10 * len(obj.aspects))


def install_test_doubles(monkeypatch) -> None:
    monkeypatch.setattr(construction_module, "EventStore", FakeEventStore)
    monkeypatch.setattr(construction_module, "KeyDirectory", FakeKeyDirectory)
    monkeypatch.setattr(construction_module, "EdgeStore", FakeEdgeStore)
    monkeypatch.setattr(construction_module, "ChainStore", FakeChainStore)
    monkeypatch.setattr(construction_module, "SkipLinkStore", FakeSkipLinkStore)
    monkeypatch.setattr(construction_module, "SkipCandidateIndex", FakeSkipCandidateIndex)
    monkeypatch.setattr(extractors_module, "featurize_event", fake_featurize_event, raising=False)
    monkeypatch.setattr(scoring_module, "dependency_score", fake_dependency_score, raising=False)
    monkeypatch.setattr(scoring_module, "skip_score", fake_skip_score, raising=False)
    monkeypatch.setattr(scoring_module, "anchor_score", fake_anchor_score, raising=False)


def make_config() -> TimeIndexConfig:
    config = TimeIndexConfig()
    config.stores.ordinary_fan_in = 2
    config.stores.skip_fan_in = 2
    config.stores.chain_summaries_per_family = 2
    config.stores.active_history_size = 50
    config.scoring.local_dependency_threshold = 0.35
    config.scoring.skip_threshold = 0.50
    config.scoring.anchor_threshold = 0.45
    return config


def make_event(event_id: str, time: int, account: str, **attrs) -> Event:
    data = {"account_id": account}
    data.update(attrs)
    return Event(event_id=event_id, time=time, event_type="transaction", attrs=data)


def build_index(monkeypatch) -> TimeIndex:
    install_test_doubles(monkeypatch)
    return TimeIndex(make_config())


def test_ordinary_fan_in_bound(monkeypatch) -> None:
    index = build_index(monkeypatch)
    for i in range(4):
        index.insert(make_event(f"p{i}", i, "A", anchor=True, aspects={"source_accumulation"}))

    target = make_event("q", 10, "A", beneficiary_id="B", aspects={"large_transfer"})
    index.insert(target)

    incoming = index.ordinary_links("q")
    assert len(incoming) == 2
    assert {link.successor_id for link in incoming} == {"q"}


def test_skip_fan_in_bound(monkeypatch) -> None:
    index = build_index(monkeypatch)
    for i in range(5):
        index.insert(make_event(f"a{i}", i, f"A{i}", anchor=True, aspects={"source_accumulation"}))

    index.insert(make_event("base", 20, "shared", anchor=False, aspects={"routine"}))
    target = make_event("query", 21, "shared", beneficiary_id="B", anchor=False, aspects={"full_balance_transfer"})
    index.insert(target)

    incoming = index.skip_links("query")
    assert len(incoming) == 2
    assert all(link.to_id == "query" for link in incoming)


def test_chain_bound(monkeypatch) -> None:
    index = build_index(monkeypatch)
    for i in range(5):
        index.insert(make_event(f"c{i}", i, "A", aspects={"source_accumulation"}))

    target = make_event("tail", 10, "A", beneficiary_id="B", aspects={"source_accumulation"})
    index.insert(target)

    chains = index.chains("tail")
    assert len(chains) == 2
    assert all(chain.tail_id == "tail" for chain in chains)


def test_no_self_predecessor(monkeypatch) -> None:
    index = build_index(monkeypatch)
    event = make_event("self", 1, "A", beneficiary_id="B", aspects={"large_transfer"})
    index.insert(event)
    index.insert(event)

    incoming = index.ordinary_links("self")
    assert all(link.predecessor_id != "self" for link in incoming)


def test_synthetic_transaction_sequence_creates_ordinary_and_skip_links(monkeypatch) -> None:
    index = build_index(monkeypatch)
    events = [
        make_event("e1", 1, "A", amount=200, anchor=True, aspects={"source_accumulation"}),
        make_event("e2", 2, "A", amount=150, role="routine", anchor=True, aspects={"source_accumulation"}),
        make_event("e3", 3, "A", amount=40, role="routine", anchor=True, aspects={"beneficiary_novelty"}),
        make_event("e4", 4, "A", amount=30, role="routine", aspects={"routine"}),
        make_event("e5", 5, "A", amount=400, anchor=True, aspects={"large_transfer"}),
        make_event("e6", 6, "A", amount=780, anchor=True, aspects={"full_balance_transfer"}),
        make_event("e7", 7, "A", beneficiary_id="B", amount=780, aspects={"full_balance_transfer"}),
    ]

    for event in events:
        index.insert(event)

    ordinary = index.ordinary_links("e7")
    skip = index.skip_links("e7")

    assert ordinary
    assert skip
    assert any(link.predecessor_id == "e6" for link in ordinary)
    assert any(link.from_id in {"e1", "e2", "e3", "e5"} for link in skip)
