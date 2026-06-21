from __future__ import annotations

from datetime import timedelta

from benchmarks.ibm_aml.evidence import (
    FLOW_CHAIN,
    SAME_ENTITY_LAUNDERING_WINDOW,
    TYPOLOGY_GROUP_AWARE,
    WEAK_SOURCE_ACCUMULATION,
    build_gold_evidence,
)


def test_same_entity_policy_uses_only_previous_laundering_events() -> None:
    events = _events_with_pattern()

    evidence = build_gold_evidence(
        events,
        policy=SAME_ENTITY_LAUNDERING_WINDOW,
        window=4,
        max_hops=2,
        amount_threshold=0.8,
    )

    assert evidence["q1"] == {"e2", "e3"}
    _assert_all_support_precedes_query(events, evidence)


def test_flow_chain_policy_finds_prior_laundering_chain_without_future_events() -> None:
    events = _events_with_pattern()

    evidence = build_gold_evidence(
        events,
        policy=FLOW_CHAIN,
        window=5,
        max_hops=2,
        amount_threshold=0.8,
    )

    assert {"e1", "e2", "e3"}.issubset(evidence["q1"])
    assert "future_same_group" not in evidence["q1"]
    _assert_all_support_precedes_query(events, evidence)


def test_typology_group_policy_handles_missing_group_columns() -> None:
    events = _events_without_pattern()

    evidence = build_gold_evidence(
        events,
        policy=TYPOLOGY_GROUP_AWARE,
        window=timedelta(days=2),
        max_hops=2,
        amount_threshold=0.8,
    )

    assert evidence["q2"] == set()
    _assert_all_support_precedes_query(events, evidence)


def test_weak_source_accumulation_selects_incoming_support_until_threshold() -> None:
    events = _events_with_pattern()

    evidence = build_gold_evidence(
        events,
        policy=WEAK_SOURCE_ACCUMULATION,
        window=4,
        max_hops=2,
        amount_threshold=0.75,
    )

    assert evidence["q1"] == {"e4", "e5"}
    _assert_all_support_precedes_query(events, evidence)


def test_at_least_some_laundering_queries_have_non_empty_evidence() -> None:
    events = _events_with_pattern()

    policies = (
        SAME_ENTITY_LAUNDERING_WINDOW,
        FLOW_CHAIN,
        TYPOLOGY_GROUP_AWARE,
        WEAK_SOURCE_ACCUMULATION,
    )

    for policy in policies:
        evidence = build_gold_evidence(events, policy=policy, window=5, max_hops=2, amount_threshold=0.75)
        assert any(support for support in evidence.values())
        _assert_no_future_events(events, evidence)


def _assert_no_future_events(events: list[dict[str, object]], evidence: dict[str, set[str]]) -> None:
    _assert_all_support_precedes_query(events, evidence)


def _assert_all_support_precedes_query(events: list[dict[str, object]], evidence: dict[str, set[str]]) -> None:
    event_time = {str(event["event_id"]): event["timestamp"] for event in events}
    for query_id, support_ids in evidence.items():
        for support_id in support_ids:
            assert event_time[support_id] < event_time[query_id]


def _events_with_pattern() -> list[dict[str, object]]:
    return [
        {
            "event_id": "e1",
            "timestamp": 1,
            "src_account": "X",
            "dst_account": "A",
            "amount": 40.0,
            "is_laundering": True,
            "pattern_id": "p1",
        },
        {
            "event_id": "e2",
            "timestamp": 2,
            "src_account": "A",
            "dst_account": "B",
            "amount": 120.0,
            "is_laundering": True,
            "pattern_id": "p1",
        },
        {
            "event_id": "e3",
            "timestamp": 3,
            "src_account": "C",
            "dst_account": "A",
            "amount": 60.0,
            "is_laundering": True,
            "pattern_id": "p1",
        },
        {
            "event_id": "e4",
            "timestamp": 4,
            "src_account": "Y",
            "dst_account": "A",
            "amount": 30.0,
            "is_laundering": False,
            "pattern_id": "p2",
        },
        {
            "event_id": "e5",
            "timestamp": 5,
            "src_account": "Z",
            "dst_account": "A",
            "amount": 80.0,
            "is_laundering": False,
            "pattern_id": "p3",
        },
        {
            "event_id": "q1",
            "timestamp": 6,
            "src_account": "A",
            "dst_account": "D",
            "amount": 140.0,
            "is_laundering": True,
            "pattern_id": "p1",
        },
        {
            "event_id": "future_same_group",
            "timestamp": 7,
            "src_account": "D",
            "dst_account": "E",
            "amount": 20.0,
            "is_laundering": True,
            "pattern_id": "p1",
        },
    ]


def _events_without_pattern() -> list[dict[str, object]]:
    return [
        {
            "event_id": "e10",
            "timestamp": timedelta(days=1),
            "src_account": "M",
            "dst_account": "N",
            "amount": 20.0,
            "is_laundering": True,
        },
        {
            "event_id": "q2",
            "timestamp": timedelta(days=2),
            "src_account": "N",
            "dst_account": "O",
            "amount": 50.0,
            "is_laundering": True,
        },
    ]
