from __future__ import annotations

from benchmarks.ibm_aml.fake_agent import classify_query_from_retrieval, summarize_decisions
from timeindex.event import Event


def test_fake_agent_flags_query_with_positive_overlapping_support() -> None:
    query = Event(
        event_id="q1",
        time=10,
        event_type="wire",
        attrs={"src_account": "A", "dst_account": "B", "amount": 900.0},
        label="1",
    )
    support_a = Event(
        event_id="e1",
        time=7,
        event_type="deposit",
        attrs={"src_account": "X", "dst_account": "A", "amount": 500.0},
        label="1",
    )
    support_b = Event(
        event_id="e2",
        time=8,
        event_type="transfer",
        attrs={"src_account": "A", "dst_account": "Y", "amount": 600.0},
        label="1",
    )
    lookup = {event.event_id: event for event in (query, support_a, support_b)}

    decision = classify_query_from_retrieval(
        query,
        retrieved_event_ids=["e1", "e2"],
        retrieved_aspects={"large_transfer", "beneficiary_novelty"},
        event_lookup=lookup,
        gold_event_ids={"e1", "e2"},
    )

    assert decision.predicted_positive is True
    assert decision.laundering_support_count == 2
    assert decision.evidence_recall == 1.0
    assert decision.entity_overlap_count >= 1


def test_fake_agent_summary_handles_multiple_decisions() -> None:
    query = Event(event_id="q", time=1, event_type="wire", attrs={"src_account": "A", "amount": 50.0}, label="1")
    support = Event(event_id="e", time=0, event_type="deposit", attrs={"dst_account": "A", "amount": 25.0}, label="0")
    lookup = {event.event_id: event for event in (query, support)}

    decisions = [
        classify_query_from_retrieval(query, ["e"], set(), lookup, gold_event_ids=[]),
        classify_query_from_retrieval(query, [], set(), lookup, gold_event_ids=[]),
    ]
    summary = summarize_decisions(decisions)

    assert summary["queries"] == 2.0
    assert 0.0 <= summary["predicted_positive_rate"] <= 1.0
    assert 0.0 <= summary["mean_decision_score"] <= 1.0
