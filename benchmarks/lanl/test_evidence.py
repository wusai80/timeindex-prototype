from __future__ import annotations

from benchmarks.lanl.evidence import (
    LATERAL_CHAIN,
    NOVEL_ACCESS_BOOTSTRAP,
    SAME_COMPUTER_HISTORY,
    SAME_USER_HISTORY,
    UNION,
    build_gold_evidence,
)


def test_same_user_history_uses_only_previous_events() -> None:
    evidence = build_gold_evidence(_events(), policy=SAME_USER_HISTORY, window=10)
    assert evidence["q1"] == {"e1", "e2", "e3"}
    _assert_all_support_precedes_query(_events(), evidence)


def test_same_computer_history_uses_overlapping_hosts_only() -> None:
    evidence = build_gold_evidence(_events(), policy=SAME_COMPUTER_HISTORY, window=10)
    assert {"e3", "e4"} == evidence["q1"]
    _assert_all_support_precedes_query(_events(), evidence)


def test_lateral_chain_finds_connected_prior_path() -> None:
    evidence = build_gold_evidence(_events(), policy=LATERAL_CHAIN, window=10, max_hops=2)
    assert {"e1", "e2", "e3"}.issubset(evidence["q1"])
    _assert_all_support_precedes_query(_events(), evidence)


def test_novel_access_bootstrap_handles_first_seen_host() -> None:
    evidence = build_gold_evidence(_events(), policy=NOVEL_ACCESS_BOOTSTRAP, window=10)
    assert "e3" in evidence["q1"]
    _assert_all_support_precedes_query(_events(), evidence)


def test_union_produces_non_empty_evidence_for_positive_queries() -> None:
    evidence = build_gold_evidence(_events(), policy=UNION, window=10, max_hops=2)
    assert evidence["q1"]
    _assert_all_support_precedes_query(_events(), evidence)


def _assert_all_support_precedes_query(events: list[dict[str, object]], evidence: dict[str, set[str]]) -> None:
    time_by_id = {str(event["event_id"]): int(event["time"]) for event in events}
    for query_id, support_ids in evidence.items():
        for support_id in support_ids:
            assert time_by_id[support_id] < time_by_id[query_id]


def _events() -> list[dict[str, object]]:
    return [
        {"event_id": "e1", "time": 1, "label": "0", "src_user": "alice", "dst_user": "bob", "src_computer": "c1", "dst_computer": "c2"},
        {"event_id": "e2", "time": 2, "label": "0", "src_user": "alice", "dst_user": "bob", "src_computer": "c2", "dst_computer": "c3"},
        {"event_id": "e3", "time": 3, "label": "0", "src_user": "alice", "dst_user": "svc", "src_computer": "c3", "dst_computer": "c4"},
        {"event_id": "e4", "time": 4, "label": "0", "src_user": "charlie", "dst_user": "svc", "src_computer": "c4", "dst_computer": "c5"},
        {"event_id": "q1", "time": 5, "label": "1", "src_user": "alice", "dst_user": "admin", "src_computer": "c4", "dst_computer": "c5"},
        {"event_id": "future", "time": 6, "label": "0", "src_user": "alice", "dst_user": "admin", "src_computer": "c5", "dst_computer": "c6"},
    ]
