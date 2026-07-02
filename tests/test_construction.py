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

    def expire(self, active_history_size: int | None = None, max_size: int | None = None) -> list[str]:
        limit = active_history_size if active_history_size is not None else max_size
        if limit is None:
            return []
        self.limit = limit
        expired: list[str] = []
        while len(self.order) > limit:
            oldest = self.order.popleft()
            record = self.records.get(oldest)
            if record is not None:
                record.metadata.expired = True
                expired.append(oldest)
        return expired


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

    def expire(self, expired_event_ids) -> None:
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
    assert len(incoming) == 1
    assert {link.successor_id for link in incoming} == {"q"}
    assert [link.predecessor_id for link in incoming] == ["p3"]


def test_skip_fan_in_bound(monkeypatch) -> None:
    index = build_index(monkeypatch)
    for i in range(5):
        index.insert(make_event(f"a{i}", i, f"A{i}", beneficiary_id="shared", anchor=True, aspects={"source_accumulation"}))

    index.insert(make_event("base", 20, "shared", anchor=False, aspects={"routine"}))
    target = make_event("query", 21, "shared", beneficiary_id="B", anchor=False, aspects={"full_balance_transfer"})
    index.insert(target)

    incoming = index.skip_links("query")
    assert len(incoming) == 2
    assert all(link.to_id == "query" for link in incoming)


def test_enable_skip_links_flag_disables_skip_insertion(monkeypatch) -> None:
    install_test_doubles(monkeypatch)
    config = make_config()
    config.construction.enable_skip_links = False
    index = TimeIndex(config)

    index.insert(make_event("p1", 1, "A", beneficiary_id="shared", anchor=True, aspects={"source_accumulation"}))
    index.insert(make_event("q1", 2, "A", beneficiary_id="B", aspects={"full_balance_transfer"}))

    assert index.skip_links("q1") == []


def test_trivial_single_event_skip_anchors_can_be_filtered(monkeypatch) -> None:
    install_test_doubles(monkeypatch)
    config = make_config()
    config.construction.skip_min_single_event_order_gap = 10
    index = TimeIndex(config)

    for i in range(5):
        index.insert(make_event(f"a{i}", i, f"A{i}", beneficiary_id="shared", anchor=True, aspects={"source_accumulation"}))

    index.insert(make_event("base", 20, "shared", anchor=False, aspects={"routine"}))
    index.insert(make_event("query", 21, "shared", beneficiary_id="B", anchor=False, aspects={"full_balance_transfer"}))

    skip_links = index.skip_links("query")

    assert skip_links
    assert all(len(link.representative_event_ids) >= 2 for link in skip_links)


def test_skip_funnel_report_tracks_stages(monkeypatch) -> None:
    index = build_index(monkeypatch)
    for i in range(3):
        index.insert(make_event(f"a{i}", i, f"A{i}", beneficiary_id="shared", anchor=True, aspects={"source_accumulation"}))

    index.insert(make_event("base", 20, "shared", anchor=False, aspects={"routine"}))
    index.insert(make_event("query", 21, "shared", beneficiary_id="B", anchor=False, aspects={"full_balance_transfer"}))

    report = index.skip_funnel_report()

    assert report["events_processed"] >= 5
    assert report["stages"]["raw_candidates"] >= report["stages"]["forward_validated_candidates"]
    assert report["stages"].get("selected_links", 0) >= 0
    assert "candidate_types" in report
    assert "score_means" in report


def test_chain_richness_report_tracks_created_and_anchored_chains(monkeypatch) -> None:
    index = build_index(monkeypatch)
    index.insert(make_event("p1", 1, "A", beneficiary_id="mid", anchor=True, aspects={"source_accumulation"}))
    index.insert(make_event("p2", 2, "mid", beneficiary_id="B", anchor=True, aspects={"beneficiary_novelty"}))
    index.insert(make_event("q", 3, "B", beneficiary_id="C", anchor=False, aspects={"large_transfer"}))

    report = index.chain_richness_report()

    assert report["created_chains"]["count"] >= 1
    assert report["created_chains"]["mean_hop_count"] >= 1.0
    assert report["created_chains"]["max_representative_event_count"] >= 1
    assert report["anchored_chains"]["count"] >= 1


def test_iso_time_strings_are_causal_in_construction(monkeypatch) -> None:
    install_test_doubles(monkeypatch)
    index = TimeIndex(make_config())

    early = fake_featurize_event(
        Event(
            event_id="e1",
            time="2025-01-01T00:00:00",
            event_type="transaction",
            attrs={"account_id": "A", "beneficiary_id": "B", "aspects": {"source_accumulation"}},
        ),
        None,
    )
    late = fake_featurize_event(
        Event(
            event_id="e2",
            time="2025-01-01T01:00:00",
            event_type="transaction",
            attrs={"account_id": "A", "beneficiary_id": "C", "aspects": {"large_transfer"}},
        ),
        None,
    )

    assert index._is_causal_predecessor(early, late) is True


def test_same_timestamp_is_not_causal_predecessor(monkeypatch) -> None:
    install_test_doubles(monkeypatch)
    index = TimeIndex(make_config())
    shared_time = fake_featurize_event(
        Event(event_id="e1", time=5, event_type="transaction", attrs={"account_id": "A"}),
        None,
    )
    same_time = fake_featurize_event(
        Event(event_id="e2", time=5, event_type="transaction", attrs={"account_id": "A"}),
        None,
    )

    assert index._is_causal_predecessor(shared_time, same_time) is False


def test_enable_bridge_score_gates_chain_anchor_registration(monkeypatch) -> None:
    config = make_config()
    config.construction.enable_bridge_score = False
    install_test_doubles(monkeypatch)
    index = TimeIndex(config)
    index.insert(make_event("p1", 1, "A", beneficiary_id="B", anchor=True, aspects={"source_accumulation"}))
    index.insert(make_event("q", 2, "B", beneficiary_id="C", aspects={"large_transfer"}))

    assert index.chain_richness_report()["created_chains"]["count"] >= 1
    assert index.chain_richness_report()["anchored_chains"]["count"] == 0


def test_chain_candidate_with_missing_supporting_events_is_not_causal(monkeypatch) -> None:
    index = build_index(monkeypatch)
    target = fake_featurize_event(make_event("q", 3, "A", beneficiary_id="B", aspects={"large_transfer"}), None)
    chain = ChainSummary(
        chain_id="c-missing",
        family="transaction_flow",
        head_id="missing-1",
        tail_id="missing-2",
        representative_event_ids=["missing-1"],
        source_entities={"x"},
        destination_entities={"a"},
        aspects={"generic_evidence"},
    )

    assert index._candidate_is_causal(chain, target) is False


def test_rarity_is_populated_during_indexing(monkeypatch) -> None:
    index = build_index(monkeypatch)
    index.insert(make_event("e1", 1, "A", beneficiary_id="B", aspects={"source_accumulation"}))
    record = index.insert(make_event("e2", 2, "A", beneficiary_id="B", aspects={"large_transfer"}))

    assert 0.0 <= record.metadata.rarity <= 1.0


def test_chain_bound(monkeypatch) -> None:
    index = build_index(monkeypatch)
    for i in range(5):
        index.insert(make_event(f"c{i}", i, "A", aspects={"source_accumulation"}))

    target = make_event("tail", 10, "A", beneficiary_id="B", aspects={"source_accumulation"})
    index.insert(target)

    chains = index.chains("tail")
    assert len(chains) == 1
    assert all(chain.tail_id == "tail" for chain in chains)
    assert [chain.head_id for chain in chains] == ["c4"]
    assert chains[0].representative_event_ids[-1] == "c4"


def test_no_self_predecessor(monkeypatch) -> None:
    index = build_index(monkeypatch)
    event = make_event("self", 1, "A", beneficiary_id="B", aspects={"large_transfer"})
    index.insert(event)
    index.insert(event)

    incoming = index.ordinary_links("self")
    assert all(link.predecessor_id != "self" for link in incoming)


def test_no_future_predecessor_by_timestamp(monkeypatch) -> None:
    index = build_index(monkeypatch)
    index.insert(make_event("future_first", 10, "A", anchor=True, aspects={"source_accumulation"}))
    index.insert(make_event("past_second", 4, "A", anchor=True, aspects={"source_accumulation"}))
    index.insert(make_event("query", 6, "A", beneficiary_id="B", aspects={"large_transfer"}))

    incoming = index.ordinary_links("query")

    assert incoming
    assert all(link.predecessor_id != "future_first" for link in incoming)
    assert any(link.predecessor_id == "past_second" for link in incoming)


def test_chain_summaries_store_supporting_history_only(monkeypatch) -> None:
    index = build_index(monkeypatch)
    index.insert(make_event("p1", 1, "A", anchor=True, aspects={"source_accumulation"}))
    index.insert(make_event("q1", 2, "A", beneficiary_id="B", aspects={"large_transfer"}))

    chains = index.chains("q1")

    assert chains
    assert all("q1" not in chain.representative_event_ids for chain in chains)
    assert all(chain.representative_event_ids == ["p1"] for chain in chains)
    assert all(chain.source_entities == {"a"} for chain in chains)
    assert all(chain.destination_entities == set() for chain in chains)


def test_forward_validation_uses_chain_supporting_event(monkeypatch) -> None:
    index = build_index(monkeypatch)
    source = make_event("p1", 1, "A", beneficiary_id="B", aspects={"source_accumulation"})
    target = make_event("q1", 2, "B", beneficiary_id="C", aspects={"large_transfer"})
    source_record = fake_featurize_event(source, None)
    target_record = fake_featurize_event(target, None)
    chain = ChainSummary(
        chain_id="c1",
        family="transaction_flow",
        head_id="p1",
        tail_id="missing-tail",
        representative_event_ids=["p1"],
        source_entities={"a"},
        destination_entities={"b"},
        aspects={"source_accumulation"},
        dependency_confidence=0.8,
        summary="transaction_flow chain from p1 to missing-tail",
    )
    index.event_store.insert(source_record)
    index.event_store.insert(target_record)

    assert index._forward_validates_skip_candidate(chain, target_record, []) is True


def test_skip_links_keep_distinct_chain_anchor_ids(monkeypatch) -> None:
    index = build_index(monkeypatch)

    root1 = fake_featurize_event(
        Event("r1", 1, "transaction", attrs={"account_id": "A", "beneficiary_id": "B", "aspects": {"source_accumulation"}}),
        None,
    )
    root2 = fake_featurize_event(
        Event("r2", 2, "transaction", attrs={"account_id": "A", "beneficiary_id": "B", "aspects": {"source_accumulation"}}),
        None,
    )
    tail = fake_featurize_event(
        Event("tail", 3, "transaction", attrs={"account_id": "B", "beneficiary_id": "C", "aspects": {"large_transfer"}}),
        None,
    )
    target = fake_featurize_event(
        Event("q", 4, "transaction", attrs={"account_id": "C", "beneficiary_id": "D", "aspects": {"large_transfer"}}),
        None,
    )
    for record in (root1, root2, tail, target):
        index.event_store.insert(record)

    chain1 = ChainSummary(
        chain_id="chain:r1:tail",
        family="transaction_flow",
        head_id="r1",
        tail_id="tail",
        representative_event_ids=["r1", "tail"],
        source_entities={"a"},
        destination_entities={"c"},
        aspects={"large_transfer"},
        dependency_confidence=0.9,
        summary="branch 1",
    )
    chain2 = ChainSummary(
        chain_id="chain:r2:tail",
        family="transaction_flow",
        head_id="r2",
        tail_id="tail",
        representative_event_ids=["r2", "tail"],
        source_entities={"a"},
        destination_entities={"c"},
        aspects={"large_transfer"},
        dependency_confidence=0.85,
        summary="branch 2",
    )

    skip_links = index._select_skip_links(target, [chain1, chain2], ordinary_links=[])

    assert len(skip_links) == 2
    assert {link.from_id for link in skip_links} == {"chain:r1:tail", "chain:r2:tail"}


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
    assert any(set(link.representative_event_ids) & {"e1", "e2", "e3", "e5"} for link in skip)


def test_real_index_links_across_interacting_accounts() -> None:
    config = TimeIndexConfig()
    config.stores.ordinary_fan_in = 3
    config.stores.skip_fan_in = 0
    config.stores.active_history_size = 100
    config.scoring.local_dependency_threshold = 0.30
    config.scoring.time_decay = 1_000_000.0

    index = TimeIndex(config)
    events = [
        Event(
            event_id="e1",
            time=1,
            event_type="transfer",
            attrs={
                "src_account": "X",
                "dst_account": "A",
                "amount": 500.0,
                "payment_format": "wire",
                "currency": "USD",
            },
        ),
        Event(
            event_id="e2",
            time=2,
            event_type="transfer",
            attrs={
                "src_account": "A",
                "dst_account": "B",
                "amount": 500.0,
                "payment_format": "wire",
                "currency": "USD",
            },
        ),
        Event(
            event_id="q",
            time=3,
            event_type="transfer",
            attrs={
                "src_account": "B",
                "dst_account": "C",
                "amount": 500.0,
                "payment_format": "wire",
                "currency": "USD",
            },
        ),
    ]

    for event in events:
        index.insert(event)

    incoming = index.ordinary_links("q")
    chains = index.chains("q")

    assert any(link.predecessor_id == "e2" for link in incoming)
    assert any(chain.family == "transaction_flow" for chain in chains)


def test_real_timeindex_insert_and_expire_runtime_path() -> None:
    config = TimeIndexConfig()
    config.stores.active_history_size = 2
    config.stores.posting_list_size = 10
    config.stores.ordinary_fan_in = 2
    config.stores.skip_fan_in = 2

    index = TimeIndex(config)
    index.insert(Event(event_id="e1", time=1, event_type="deposit", attrs={"account_id": "A", "amount": 100.0}))
    index.insert(Event(event_id="e2", time=2, event_type="deposit", attrs={"account_id": "A", "amount": 200.0}))
    index.insert(Event(event_id="e3", time=3, event_type="transfer", attrs={"account_id": "A", "amount": 200.0}))

    assert index.get_event("e1") is None
    assert index.get_event("e2") is not None
    assert index.get_event("e3") is not None


def test_real_local_candidates_keep_latest_same_entity_and_latest_bridge() -> None:
    config = TimeIndexConfig()
    config.stores.posting_list_size = 20
    config.stores.active_history_size = 100
    config.stores.skip_fan_in = 0

    index = TimeIndex(config)
    history = [
        Event(
            event_id="same_old",
            time=1,
            event_type="transfer",
            attrs={"src_account": "A", "dst_account": "M", "amount": 100.0, "payment_format": "wire", "currency": "USD"},
        ),
        Event(
            event_id="same_new",
            time=2,
            event_type="transfer",
            attrs={"src_account": "A", "dst_account": "N", "amount": 100.0, "payment_format": "wire", "currency": "USD"},
        ),
        Event(
            event_id="bridge_old",
            time=3,
            event_type="transfer",
            attrs={"src_account": "X", "dst_account": "A", "amount": 100.0, "payment_format": "wire", "currency": "USD"},
        ),
        Event(
            event_id="bridge_new",
            time=4,
            event_type="transfer",
            attrs={"src_account": "Y", "dst_account": "A", "amount": 100.0, "payment_format": "wire", "currency": "USD"},
        ),
    ]

    for event in history:
        index.insert(event)

    query = Event(
        event_id="q",
        time=5,
        event_type="transfer",
        attrs={"src_account": "A", "dst_account": "Z", "amount": 100.0, "payment_format": "wire", "currency": "USD"},
    )
    record = index._featurize_event(query)
    candidates = index._local_candidate_records(record)

    assert [candidate.event.event_id for candidate in candidates] == ["bridge_new", "same_new"]


def test_real_local_candidates_use_entity_directory_before_generic_lookup() -> None:
    config = TimeIndexConfig()
    config.stores.posting_list_size = 20
    config.stores.active_history_size = 100
    config.stores.skip_fan_in = 0

    index = TimeIndex(config)
    index.insert(
        Event(
            event_id="same_new",
            time=2,
            event_type="transfer",
            attrs={"src_account": "A", "dst_account": "N", "amount": 100.0, "payment_format": "wire", "currency": "USD"},
        )
    )
    index.insert(
        Event(
            event_id="bridge_new",
            time=4,
            event_type="transfer",
            attrs={"src_account": "Y", "dst_account": "A", "amount": 100.0, "payment_format": "wire", "currency": "USD"},
        )
    )

    query = Event(
        event_id="q",
        time=5,
        event_type="transfer",
        attrs={"src_account": "A", "dst_account": "Z", "amount": 100.0, "payment_format": "wire", "currency": "USD"},
    )
    record = index._featurize_event(query)
    candidates = index._entity_candidate_records(record)

    assert [candidate.event.event_id for candidate in candidates] == ["same_new", "bridge_new"]


def test_real_local_candidates_keep_latest_bridge_per_bridged_entity() -> None:
    config = TimeIndexConfig()
    config.stores.posting_list_size = 30
    config.stores.active_history_size = 100
    config.stores.skip_fan_in = 0

    index = TimeIndex(config)
    history = [
        Event(
            event_id="bridge_account",
            time=1,
            event_type="transfer",
            attrs={"src_account": "X", "dst_account": "A", "amount": 100.0, "payment_format": "wire", "currency": "USD"},
        ),
        Event(
            event_id="bridge_user",
            time=2,
            event_type="transfer",
            attrs={"src_account": "Y", "dst_user": "user_1", "amount": 100.0, "payment_format": "wire", "currency": "USD"},
        ),
    ]

    for event in history:
        index.insert(event)

    query = Event(
        event_id="q",
        time=3,
        event_type="transfer",
        attrs={
            "src_account": "A",
            "src_user": "user_1",
            "dst_account": "Z",
            "amount": 100.0,
            "payment_format": "wire",
            "currency": "USD",
        },
    )
    record = index._featurize_event(query)
    candidates = index._local_candidate_records(record)

    assert [candidate.event.event_id for candidate in candidates] == ["bridge_user", "bridge_account"]


def test_real_local_candidates_keep_latest_fallback_after_prefilter() -> None:
    config = TimeIndexConfig()
    config.stores.posting_list_size = 20
    config.stores.active_history_size = 100
    config.stores.skip_fan_in = 0
    config.extractor.time_bucket_width = 100

    index = TimeIndex(config)
    index.insert(Event(event_id="old", time=1, event_type="heartbeat", attrs={"status": "ok"}, text="steady"))
    index.insert(Event(event_id="new", time=2, event_type="heartbeat", attrs={"status": "warn"}, text="drift"))

    query = Event(
        event_id="q",
        time=3,
        event_type="heartbeat",
        attrs={"status": "critical"},
        text="outage",
    )
    record = index._featurize_event(query)
    candidates = index._local_candidate_records(record)

    assert [candidate.event.event_id for candidate in candidates] == ["new"]


def test_real_lanl_computer_handoff_links_predecessor() -> None:
    config = TimeIndexConfig()
    config.stores.ordinary_fan_in = 3
    config.stores.skip_fan_in = 0
    config.stores.active_history_size = 100
    config.scoring.local_dependency_threshold = 0.30
    config.scoring.time_decay = 1_000_000.0

    index = TimeIndex(config)
    history = [
        Event(
            event_id="e1",
            time=1,
            event_type="authentication",
            attrs={
                "src_user": "alice",
                "dst_user": "bob",
                "src_computer": "c1",
                "dst_computer": "c2",
                "auth_type": "Kerberos",
                "is_cross_host_auth": True,
            },
        ),
        Event(
            event_id="e2",
            time=2,
            event_type="authentication",
            attrs={
                "src_user": "alice",
                "dst_user": "bob",
                "src_computer": "c2",
                "dst_computer": "c3",
                "auth_type": "Kerberos",
                "is_cross_host_auth": True,
            },
        ),
        Event(
            event_id="q",
            time=3,
            event_type="authentication",
            attrs={
                "src_user": "alice",
                "dst_user": "admin",
                "src_computer": "c3",
                "dst_computer": "c4",
                "auth_type": "Kerberos",
                "is_cross_host_auth": True,
                "is_new_dst_for_user": True,
            },
        ),
    ]

    for event in history:
        index.insert(event)

    incoming = index.ordinary_links("q")
    assert any(link.predecessor_id == "e2" for link in incoming)


def test_real_ordinary_link_tie_break_prefers_more_recent_candidate() -> None:
    config = TimeIndexConfig()
    config.stores.ordinary_fan_in = 1
    config.stores.skip_fan_in = 0
    config.stores.active_history_size = 100
    config.scoring.local_dependency_threshold = 0.0
    config.scoring.time_decay = 1_000_000.0

    index = TimeIndex(config)
    older = Event(event_id="older", time=1, event_type="heartbeat", attrs={"status": "ok"}, text="steady")
    newer = Event(event_id="newer", time=2, event_type="heartbeat", attrs={"status": "ok"}, text="steady")
    query = Event(event_id="q", time=3, event_type="heartbeat", attrs={"status": "ok"}, text="steady")

    index.insert(older)
    index.insert(newer)
    index.insert(query)

    incoming = index.ordinary_links("q")
    assert [link.predecessor_id for link in incoming] == ["newer"]
