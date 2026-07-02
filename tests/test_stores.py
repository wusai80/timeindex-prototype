import numpy as np

from timeindex.candidate_index import SkipCandidateIndex
from timeindex.config import StoreConfig
from timeindex.event import ChainSummary, DecisionIntent, Event, EventMetadata, EventRecord, OrdinaryLink, SkipLink
from timeindex.stores import ChainStore, EdgeStore, EntityDirectory, EventStore, KeyDirectory, SkipLinkStore


def make_record(
    event_id: str,
    *,
    aspects: set[str] | None = None,
    rarity: float = 0.0,
    vector: list[float] | None = None,
    attrs: dict[str, object] | None = None,
) -> EventRecord:
    return EventRecord(
        event=Event(event_id=event_id, time=int(event_id.strip("e") or 0), event_type="transfer", attrs=attrs or {}),
        lookup_keys={f"entity:{event_id}"},
        sketch=np.array(vector if vector is not None else [1.0, 0.0, 0.0], dtype=float),
        aspects=aspects or set(),
        metadata=EventMetadata(rarity=rarity),
    )


def test_event_store_insert_contains_and_expire() -> None:
    store = EventStore()
    records = [make_record("e1"), make_record("e2"), make_record("e3")]

    for record in records:
        store.insert(record)

    assert store.contains("e2")
    assert store.is_valid("e3")
    assert [record.event.event_id for record in store.list()] == ["e1", "e2", "e3"]

    expired_ids = store.expire(max_size=2)
    assert expired_ids == ["e1"]
    assert not store.contains("e1")
    assert store.get("e1") is None
    assert [record.event.event_id for record in store.list()] == ["e2", "e3"]


def test_key_directory_bounded_postings_and_expire() -> None:
    directory = KeyDirectory(posting_list_size=2)
    directory.add_event("e1", ["entity:alice", "ctx:atm"])
    directory.add_event("e2", ["entity:alice"])
    directory.add_event("e3", ["entity:alice", "ctx:web"])

    assert list(directory.lookup("entity:alice")) == ["e3", "e2"]
    assert list(directory.lookup_keys(["entity:alice", "ctx:web"])) == ["e3", "e2"]

    directory.expire(["e2"])
    assert list(directory.lookup("entity:alice")) == ["e3"]


def test_key_directory_does_not_duplicate_when_alias_is_called_once() -> None:
    directory = KeyDirectory(posting_list_size=5)
    directory.add_event("e1", ["entity:alice"])
    directory.add("e2", ["entity:alice"])

    assert list(directory.lookup("entity:alice")) == ["e2", "e1"]


def test_entity_directory_returns_newest_first_for_same_and_bridge_lookups() -> None:
    directory = EntityDirectory(posting_list_size=5)
    directory.add_event("e1", ["a"], ["m"])
    directory.add_event("e2", ["a"], ["n"])
    directory.add_event("e3", ["x"], ["a"])

    assert list(directory.recent_sources("a")) == ["e2", "e1"]
    assert list(directory.recent_destinations("a")) == ["e3"]
    assert list(directory.recent_participants("a")) == ["e3", "e2", "e1"]
    assert list(directory.recent_flow_pair("a", "n")) == ["e2"]


def test_edge_store_enforces_incoming_fan_in_and_updates_outgoing() -> None:
    store = EdgeStore(fan_in=2)
    store.add(OrdinaryLink(predecessor_id="e1", successor_id="e4", score=0.60))
    store.add(OrdinaryLink(predecessor_id="e2", successor_id="e4", score=0.90))
    store.add(OrdinaryLink(predecessor_id="e3", successor_id="e4", score=0.75))

    incoming_ids = [link.predecessor_id for link in store.incoming("e4")]
    outgoing_e1 = [link.successor_id for link in store.outgoing("e1")]
    outgoing_e2 = [link.successor_id for link in store.outgoing("e2")]
    outgoing_e3 = [link.successor_id for link in store.outgoing("e3")]

    assert incoming_ids == ["e2", "e3"]
    assert outgoing_e1 == []
    assert outgoing_e2 == ["e4"]
    assert outgoing_e3 == ["e4"]


def test_chain_store_enforces_bound_per_tail_and_family() -> None:
    store = ChainStore(summaries_per_family=2)
    store.add(ChainSummary(chain_id="c1", family="payments", head_id="e1", tail_id="e9", dependency_confidence=0.2))
    store.add(ChainSummary(chain_id="c2", family="payments", head_id="e2", tail_id="e9", dependency_confidence=0.9))
    store.add(ChainSummary(chain_id="c3", family="payments", head_id="e3", tail_id="e9", dependency_confidence=0.5))
    store.add(ChainSummary(chain_id="c4", family="alerts", head_id="e4", tail_id="e9", dependency_confidence=0.7))

    summaries = store.get_for_tail("e9")
    payment_ids = [summary.chain_id for summary in summaries if summary.family == "payments"]
    alert_ids = [summary.chain_id for summary in summaries if summary.family == "alerts"]

    assert payment_ids == ["c2", "c3"]
    assert alert_ids == ["c4"]


def test_skip_link_store_enforces_incoming_fan_in() -> None:
    store = SkipLinkStore(fan_in=2)
    store.add(SkipLink(from_id="e1", to_id="e5", skip_value=0.1))
    store.add(SkipLink(from_id="e2", to_id="e5", skip_value=0.8))
    store.add(SkipLink(from_id="e3", to_id="e5", skip_value=0.5))

    incoming_ids = [link.from_id for link in store.incoming("e5")]
    assert incoming_ids == ["e2", "e3"]
    assert list(store.outgoing("e1")) == []


def test_skip_candidate_index_is_bounded_and_excludes_ordinary_predecessors() -> None:
    config = StoreConfig(
        anchor_candidates=3,
        correlation_candidates=3,
        rarity_candidates=3,
        intent_candidates=3,
        aspect_candidates=3,
    )
    index = SkipCandidateIndex(config)
    intent = DecisionIntent(name="fraud-review", aspects={"large_transfer", "beneficiary_novelty"})

    index.add_event_anchor(
        make_record("e1", aspects={"beneficiary_novelty"}, rarity=0.2, vector=[1.0, 0.0, 0.0]),
        intent,
    )
    index.add_event_anchor(
        make_record("e2", aspects={"large_transfer"}, rarity=0.8, vector=[0.9, 0.1, 0.0]),
        intent,
    )
    index.add_event_anchor(
        make_record("e3", aspects={"device_shift"}, rarity=0.9, vector=[0.2, 0.8, 0.0]),
        intent,
    )
    index.add_chain_anchor(
        ChainSummary(
            chain_id="c1",
            family="accumulation",
            head_id="e1",
            tail_id="e6",
            aspects={"large_transfer", "beneficiary_novelty"},
            dependency_confidence=0.7,
        ),
        intent,
    )

    query = make_record("e9", aspects={"large_transfer"}, rarity=0.1, vector=[1.0, 0.0, 0.0])
    candidates = index.get_skip_candidates(query, intent, ordinary_predecessors=["e2"])

    assert "e9" not in candidates
    assert "e2" not in candidates
    assert len(candidates) <= 3
    assert candidates[0] in {"e1", "c1"}


def test_skip_candidate_index_prefers_query_participant_overlap() -> None:
    config = StoreConfig(
        anchor_candidates=3,
        correlation_candidates=3,
        rarity_candidates=3,
        intent_candidates=3,
        aspect_candidates=3,
    )
    index = SkipCandidateIndex(config)
    intent = DecisionIntent(name="aml", aspects={"large_transfer"})

    index.add_event_anchor(
        make_record(
            "e1",
            aspects={"large_transfer"},
            rarity=0.3,
            attrs={"src_account": "X", "dst_account": "A"},
        ),
        intent,
    )
    index.add_event_anchor(
        make_record(
            "e2",
            aspects={"large_transfer"},
            rarity=0.9,
            attrs={"src_account": "M", "dst_account": "N"},
        ),
        intent,
    )

    query = make_record(
        "e9",
        aspects={"large_transfer"},
        attrs={"src_account": "A", "dst_account": "B"},
    )
    candidates = index.get_skip_candidates(query, intent, ordinary_predecessors=[])

    assert candidates
    assert candidates[0] == "e1"


def test_skip_candidate_index_prefers_richer_chain_bridge_over_newer_shallow_one() -> None:
    config = StoreConfig(
        anchor_candidates=4,
        correlation_candidates=0,
        rarity_candidates=0,
        intent_candidates=4,
        aspect_candidates=4,
    )
    index = SkipCandidateIndex(config)
    intent = DecisionIntent(name="aml", aspects={"source_accumulation", "large_transfer"})

    index.add_chain_anchor(
        ChainSummary(
            chain_id="c_rich",
            family="transaction_flow",
            head_id="e1",
            tail_id="e5",
            representative_event_ids=["e1", "e2", "e3", "e5"],
            source_entities={"seed"},
            destination_entities={"a"},
            aspects={"source_accumulation", "large_transfer"},
            dependency_confidence=0.7,
            hop_count=4,
            order_span=18,
            temporal_span_seconds=172800.0,
        ),
        intent,
    )
    index.add_chain_anchor(
        ChainSummary(
            chain_id="c_shallow",
            family="transaction_flow",
            head_id="e8",
            tail_id="e9",
            representative_event_ids=["e8", "e9"],
            source_entities={"seed"},
            destination_entities={"a"},
            aspects={"source_accumulation", "large_transfer"},
            dependency_confidence=0.9,
            hop_count=1,
            order_span=1,
            temporal_span_seconds=60.0,
        ),
        intent,
    )

    query = make_record(
        "e20",
        aspects={"large_transfer"},
        attrs={"src_account": "a", "dst_account": "sink"},
    )
    candidates = index.get_skip_candidates(query, intent, ordinary_predecessors=[])

    assert candidates
    assert candidates[0] == "c_rich"


def test_event_store_expire_purges_records_from_memory() -> None:
    store = EventStore()
    for event_id in ("e1", "e2", "e3"):
        store.insert(make_record(event_id))

    expired_ids = store.expire(max_size=1)

    assert expired_ids == ["e1", "e2"]
    assert store.get("e1") is None
    assert store.get("e2") is None
    assert [record.event.event_id for record in store.list()] == ["e3"]


def test_graph_and_chain_stores_drop_expired_references() -> None:
    edge_store = EdgeStore(fan_in=3)
    edge_store.add(OrdinaryLink(predecessor_id="e1", successor_id="e3", score=0.8))
    edge_store.add(OrdinaryLink(predecessor_id="e2", successor_id="e3", score=0.7))

    chain_store = ChainStore(summaries_per_family=3)
    chain_store.add(
        ChainSummary(
            chain_id="c1",
            family="transaction_flow",
            head_id="e1",
            tail_id="e3",
            representative_event_ids=["e1", "e2", "e3"],
            dependency_confidence=0.9,
        )
    )

    skip_store = SkipLinkStore(fan_in=3)
    skip_store.add(
        SkipLink(
            from_id="e1",
            to_id="e4",
            skip_value=0.8,
            representative_event_ids=["e1", "e3"],
        )
    )

    edge_store.expire(["e1"])
    chain_store.expire(["e1"])
    skip_store.expire(["e1"])

    assert [link.predecessor_id for link in edge_store.incoming("e3")] == ["e2"]
    assert chain_store.get_for_tail("e3") == []
    assert skip_store.incoming("e4") == []


def test_skip_candidate_index_expire_removes_event_and_chain_anchors() -> None:
    config = StoreConfig(
        anchor_candidates=5,
        correlation_candidates=5,
        rarity_candidates=5,
        intent_candidates=5,
        aspect_candidates=5,
    )
    index = SkipCandidateIndex(config)
    intent = DecisionIntent(name="aml", aspects={"large_transfer"})
    event_anchor = make_record("e1", aspects={"large_transfer"}, rarity=0.7)
    chain_anchor = ChainSummary(
        chain_id="c1",
        family="transaction_flow",
        head_id="e1",
        tail_id="e2",
        representative_event_ids=["e1", "e2"],
        aspects={"large_transfer"},
        dependency_confidence=0.8,
    )
    index.add_event_anchor(event_anchor, intent)
    index.add_chain_anchor(chain_anchor, intent)

    index.expire(["e1"])

    assert index.get_object("e1") is None
    assert index.get_object("c1") is None
    assert index.anchor_table.recent() == []


def test_skip_candidate_index_objects_stay_bounded_to_retained_entries() -> None:
    config = StoreConfig(
        anchor_candidates=2,
        correlation_candidates=2,
        rarity_candidates=2,
        intent_candidates=2,
        aspect_candidates=2,
    )
    index = SkipCandidateIndex(config)
    intent = DecisionIntent(name="aml", aspects={"large_transfer"})

    for i in range(12):
        index.add_event_anchor(
            make_record(f"e{i}", aspects={"large_transfer"}, rarity=float(i) / 20.0),
            intent,
        )

    assert len(index._objects) <= 10


def test_edge_and_skip_stores_can_disable_outgoing_links() -> None:
    edge_store = EdgeStore(fan_in=2, maintain_outgoing_links=False)
    edge_store.add(OrdinaryLink(predecessor_id="e1", successor_id="e3", score=0.8))
    edge_store.add(OrdinaryLink(predecessor_id="e2", successor_id="e3", score=0.7))
    assert [link.predecessor_id for link in edge_store.incoming("e3")] == ["e1", "e2"]
    assert list(edge_store.outgoing("e1")) == []

    skip_store = SkipLinkStore(fan_in=2, maintain_outgoing_links=False)
    skip_store.add(SkipLink(from_id="e1", to_id="e4", skip_value=0.8))
    skip_store.add(SkipLink(from_id="e2", to_id="e4", skip_value=0.7))
    assert [link.from_id for link in skip_store.incoming("e4")] == ["e1", "e2"]
    assert list(skip_store.outgoing("e1")) == []
