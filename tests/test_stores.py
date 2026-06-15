import numpy as np

from timeindex.candidate_index import SkipCandidateIndex
from timeindex.config import StoreConfig
from timeindex.event import ChainSummary, DecisionIntent, Event, EventMetadata, EventRecord, OrdinaryLink, SkipLink
from timeindex.stores import ChainStore, EdgeStore, EventStore, KeyDirectory, SkipLinkStore


def make_record(
    event_id: str,
    *,
    aspects: set[str] | None = None,
    rarity: float = 0.0,
    vector: list[float] | None = None,
) -> EventRecord:
    return EventRecord(
        event=Event(event_id=event_id, time=int(event_id.strip("e") or 0), event_type="transfer"),
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
