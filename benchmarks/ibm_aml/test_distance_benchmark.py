from __future__ import annotations

from datetime import datetime

from benchmarks.ibm_aml.run_distance_benchmark import (
    QueryProfile,
    _build_query_profiles,
    _compare_skip_effect,
    _gap_bucket,
    _size_bucket,
    _structure_bucket,
)
from benchmarks.ibm_aml.run_sqlite_realcases import QueryGold
from timeindex.event import Event, EventMetadata, EventRecord


def test_structure_bucket_prefers_flow_chain_when_accumulation_dominates() -> None:
    assert _structure_bucket(["e1"], ["e2", "e3"]) == "flow_chain_heavy"
    assert _structure_bucket(["e1", "e2"], ["e3"]) == "same_entity_heavy"
    assert _structure_bucket(["e1"], ["e2"]) == "mixed"


def test_gap_and_size_buckets_use_thresholds() -> None:
    assert _gap_bucket(10.0, 10.0) == "short_gap"
    assert _gap_bucket(11.0, 10.0) == "long_gap"
    assert _size_bucket(3, 3) == "small_gold"
    assert _size_bucket(4, 3) == "large_gold"


def test_compare_skip_effect_detects_new_gold_without_false_loss() -> None:
    comparison = _compare_skip_effect(
        {
            "query_event_id": "q1",
            "gold_union_ids": ["g1", "g2"],
            "retrieved_event_ids": ["g1", "g2"],
            "gap_bucket": "long_gap",
            "size_bucket": "large_gold",
            "structure_bucket": "flow_chain_heavy",
        },
        {
            "query_event_id": "q1",
            "gold_union_ids": ["g1", "g2"],
            "retrieved_event_ids": ["g1"],
            "gap_bucket": "long_gap",
            "size_bucket": "large_gold",
            "structure_bucket": "flow_chain_heavy",
        },
    )

    assert comparison["win"] is True
    assert comparison["loss"] is False
    assert comparison["new_gold_count"] == 1
    assert comparison["lost_gold_count"] == 0
    assert comparison["recall_delta"] > 0.0


def test_build_query_profiles_filters_by_cutoff_and_keeps_latest(monkeypatch) -> None:
    query_gold = [
        QueryGold(
            query_event_id="q1",
            query_time=datetime(2024, 1, 1, 0, 0),
            query_label="1",
            same_entity_ids=["s1"],
            weak_accumulation_ids=[],
        ),
        QueryGold(
            query_event_id="q2",
            query_time=datetime(2024, 1, 1, 0, 5),
            query_label="1",
            same_entity_ids=[],
            weak_accumulation_ids=["w2"],
        ),
        QueryGold(
            query_event_id="q3",
            query_time=datetime(2024, 1, 1, 0, 10),
            query_label="1",
            same_entity_ids=["s3"],
            weak_accumulation_ids=["w3", "w4"],
        ),
    ]

    monkeypatch.setattr(
        "benchmarks.ibm_aml.run_distance_benchmark._stream_realcase_gold",
        lambda *args, **kwargs: query_gold,
    )

    records = {
        "q1": EventRecord(
            event=Event("q1", 100.0, "wire", attrs={"src_account": "A", "dst_account": "B"}),
            metadata=EventMetadata(insertion_order=3_900_000),
        ),
        "q2": EventRecord(
            event=Event("q2", 200.0, "wire", attrs={"src_account": "A", "dst_account": "C"}),
            metadata=EventMetadata(insertion_order=4_100_000),
        ),
        "q3": EventRecord(
            event=Event("q3", 300.0, "wire", attrs={"src_account": "A", "dst_account": "D"}),
            metadata=EventMetadata(insertion_order=4_200_000),
        ),
        "s1": EventRecord(event=Event("s1", 90.0, "dep"), metadata=EventMetadata(insertion_order=1)),
        "w2": EventRecord(event=Event("w2", 180.0, "dep"), metadata=EventMetadata(insertion_order=2)),
        "s3": EventRecord(event=Event("s3", 250.0, "dep"), metadata=EventMetadata(insertion_order=3)),
        "w3": EventRecord(event=Event("w3", 260.0, "dep"), metadata=EventMetadata(insertion_order=4)),
        "w4": EventRecord(event=Event("w4", 270.0, "dep"), metadata=EventMetadata(insertion_order=5)),
    }

    class StubBackend:
        def get_event(self, event_id: str) -> EventRecord | None:
            return records.get(event_id)

    profiles = _build_query_profiles(
        csv_path=None,  # type: ignore[arg-type]
        backend=StubBackend(),  # type: ignore[arg-type]
        same_entity_window_hours=24,
        accumulation_window_hours=24,
        accumulation_threshold=0.8,
        late_insertion_order=4_000_000,
        query_limit=1,
    )

    assert len(profiles) == 1
    profile = profiles[0]
    assert isinstance(profile, QueryProfile)
    assert profile.query_event_id == "q3"
    assert profile.structure_bucket == "flow_chain_heavy"
    assert profile.gold_size == 3
