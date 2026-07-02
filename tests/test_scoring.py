from __future__ import annotations

import numpy as np

from timeindex.config import ScoringConfig
from timeindex.event import ChainSummary, DecisionIntent, Event, EventMetadata, EventQuery, EventRecord, EvidenceObject
from timeindex.scoring import (
    anchor_score,
    candidate_priority,
    cosine,
    coverage_score,
    dependency_score,
    impact_score,
    jaccard,
    rarity_score,
    retrieval_marginal_utility,
    skip_score,
    skip_score_breakdown,
)


def _record(
    event_id: str,
    time: float,
    *,
    keys: set[str],
    aspects: set[str],
    vector: list[float],
    attrs: dict[str, object] | None = None,
    rarity: float = 0.0,
    surprise: float = 0.0,
) -> EventRecord:
    return EventRecord(
        event=Event(event_id=event_id, time=time, event_type="transaction", attrs=attrs or {}),
        lookup_keys=keys,
        sketch=np.array(vector, dtype=float),
        aspects=aspects,
        metadata=EventMetadata(rarity=rarity, surprise=surprise),
    )


def test_jaccard_range_and_empty_behavior() -> None:
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard({"a", "b"}, {"b", "c"}) == 1.0 / 3.0


def test_cosine_range_and_zero_vector_behavior() -> None:
    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert cosine(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == 0.0
    assert cosine(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_dependency_score_supports_iso_datetime_strings() -> None:
    config = ScoringConfig(time_decay=3600.0)
    early = _record(
        "early",
        "2025-01-01T00:00:00",  # type: ignore[arg-type]
        keys={"entity:account:a", "type:transfer"},
        aspects={"large_transfer"},
        vector=[1.0, 0.0, 0.0],
        attrs={"src_account": "A", "dst_account": "B"},
    )
    late = _record(
        "late",
        "2025-01-01T00:10:00",  # type: ignore[arg-type]
        keys={"entity:account:a", "type:transfer"},
        aspects={"large_transfer"},
        vector=[1.0, 0.0, 0.0],
        attrs={"src_account": "A", "dst_account": "C"},
    )

    assert dependency_score(early, late, config) > 0.0


def test_rarity_score_prefers_rare_keys() -> None:
    rare = _record(
        "rare",
        1.0,
        keys={"entity:user:a", "attr:type:rare"},
        aspects={"beneficiary_novelty"},
        vector=[1.0, 0.0, 0.0],
    )
    common = _record(
        "common",
        1.0,
        keys={"entity:user:b", "attr:type:common"},
        aspects={"routine"},
        vector=[1.0, 0.0, 0.0],
    )

    frequencies = {
        "entity:user:a": 1,
        "attr:type:rare": 1,
        "entity:user:b": 9,
        "attr:type:common": 9,
    }

    assert 0.0 <= rarity_score(rare, frequencies, history_size=10) <= 1.0
    assert rarity_score(rare, frequencies, history_size=10) > rarity_score(common, frequencies, history_size=10)


def test_dependency_score_is_higher_for_more_similar_events() -> None:
    config = ScoringConfig(time_decay=10.0)
    target = _record(
        "target",
        10.0,
        keys={
            "entity:account:a",
            "attr:channel:wire",
            "ctx:region:us",
            "type:transfer",
        },
        aspects={"large_transfer"},
        vector=[1.0, 1.0, 0.0],
        rarity=0.4,
    )
    near = _record(
        "near",
        9.0,
        keys={
            "entity:account:a",
            "attr:channel:wire",
            "ctx:region:us",
            "type:transfer",
        },
        aspects={"large_transfer"},
        vector=[1.0, 1.0, 0.0],
        rarity=0.4,
    )
    far = _record(
        "far",
        100.0,
        keys={
            "entity:account:z",
            "attr:channel:cash",
            "ctx:region:eu",
            "type:deposit",
        },
        aspects={"routine"},
        vector=[0.0, 0.0, 1.0],
        rarity=0.0,
    )

    near_score = dependency_score(near, target, config)
    far_score = dependency_score(far, target, config)

    assert 0.0 <= near_score <= 1.0
    assert 0.0 <= far_score <= 1.0
    assert near_score > far_score


def test_dependency_score_counts_attr_bin_overlap() -> None:
    config = ScoringConfig(time_decay=10.0)
    target = _record(
        "target",
        10.0,
        keys={"entity:account:a", "attr_bin:amount=10^3", "type:transfer"},
        aspects={"large_transfer"},
        vector=[1.0, 0.0, 0.0],
    )
    matching = _record(
        "matching",
        9.0,
        keys={"entity:account:a", "attr_bin:amount=10^3", "type:transfer"},
        aspects={"large_transfer"},
        vector=[1.0, 0.0, 0.0],
    )
    different = _record(
        "different",
        9.0,
        keys={"entity:account:a", "attr_bin:amount=10^1", "type:transfer"},
        aspects={"large_transfer"},
        vector=[1.0, 0.0, 0.0],
    )

    assert dependency_score(matching, target, config) > dependency_score(different, target, config)


def test_dependency_score_rewards_cross_account_handoff_continuity() -> None:
    config = ScoringConfig(time_decay=10.0)
    predecessor = _record(
        "pred",
        9.0,
        keys={
            "entity:src_account=a",
            "entity:dst_account=b",
            "participant:a",
            "participant:b",
            "flow_src:a",
            "flow_dst:b",
            "flow_pair:a->b",
            "attr:payment_format=ach",
            "ctx:currency=usd",
        },
        attrs={"src_account": "A", "dst_account": "B", "payment_format": "ach", "currency": "USD"},
        aspects={"generic_evidence"},
        vector=[1.0, 0.0, 0.0],
    )
    target = _record(
        "target",
        10.0,
        keys={
            "entity:src_account=b",
            "entity:dst_account=c",
            "participant:b",
            "participant:c",
            "flow_src:b",
            "flow_dst:c",
            "flow_pair:b->c",
            "attr:payment_format=ach",
            "ctx:currency=usd",
        },
        attrs={"src_account": "B", "dst_account": "C", "payment_format": "ach", "currency": "USD"},
        aspects={"generic_evidence"},
        vector=[1.0, 0.0, 0.0],
    )
    unrelated = _record(
        "other",
        9.0,
        keys={
            "entity:src_account=x",
            "entity:dst_account=y",
            "participant:x",
            "participant:y",
            "flow_src:x",
            "flow_dst:y",
            "flow_pair:x->y",
            "attr:payment_format=ach",
            "ctx:currency=usd",
        },
        attrs={"src_account": "X", "dst_account": "Y", "payment_format": "ach", "currency": "USD"},
        aspects={"generic_evidence"},
        vector=[1.0, 0.0, 0.0],
    )

    assert dependency_score(predecessor, target, config) > dependency_score(unrelated, target, config)


def test_dependency_score_rejects_same_bank_without_account_continuity() -> None:
    config = ScoringConfig(time_decay=10.0)
    candidate = _record(
        "candidate",
        9.0,
        keys={
            "entity:transaction_id=candidate",
            "attr:src_bank=bank_1",
            "attr:dst_bank=bank_1",
            "participant:a",
            "participant:b",
            "flow_src:a",
            "flow_dst:b",
            "flow_pair:a->b",
        },
        attrs={"src_account": "A", "dst_account": "B", "src_bank": "BANK_1", "dst_bank": "BANK_1"},
        aspects={"generic_evidence"},
        vector=[1.0, 0.0, 0.0],
    )
    target = _record(
        "target",
        10.0,
        keys={
            "entity:transaction_id=target",
            "attr:src_bank=bank_1",
            "attr:dst_bank=bank_1",
            "participant:c",
            "participant:d",
            "flow_src:c",
            "flow_dst:d",
            "flow_pair:c->d",
        },
        attrs={"src_account": "C", "dst_account": "D", "src_bank": "BANK_1", "dst_bank": "BANK_1"},
        aspects={"generic_evidence"},
        vector=[1.0, 0.0, 0.0],
    )

    assert dependency_score(candidate, target, config) == 0.0


def test_dependency_score_uses_normalized_sketch_similarity_without_changing_range() -> None:
    config = ScoringConfig(time_decay=10.0)
    target = _record(
        "target",
        10.0,
        keys={"entity:account:a", "type:transfer"},
        aspects={"large_transfer"},
        vector=[1.0, 0.0, 0.0],
    )
    aligned = _record(
        "aligned",
        9.0,
        keys={"entity:account:a", "type:transfer"},
        aspects={"large_transfer"},
        vector=[1.0, 0.0, 0.0],
    )
    orthogonal = _record(
        "orthogonal",
        9.0,
        keys={"entity:account:a", "type:transfer"},
        aspects={"large_transfer"},
        vector=[0.0, 1.0, 0.0],
    )

    aligned_score = dependency_score(aligned, target, config)
    orthogonal_score = dependency_score(orthogonal, target, config)

    assert target.sketch_is_normalized is True
    assert aligned.sketch_is_normalized is True
    assert orthogonal.sketch_is_normalized is True
    assert 0.0 <= aligned_score <= 1.0
    assert 0.0 <= orthogonal_score <= 1.0
    assert aligned_score > orthogonal_score


def test_impact_and_coverage_reflect_intent_overlap() -> None:
    intent = DecisionIntent(
        aspects={"large_transfer", "beneficiary_novelty"},
        aspect_weights={"large_transfer": 2.0, "beneficiary_novelty": 1.0},
    )
    record = _record(
        "e1",
        1.0,
        keys={"entity:account:a"},
        aspects={"large_transfer"},
        vector=[1.0, 0.0, 0.0],
    )

    assert impact_score(record, intent) == 2.0 / 3.0
    assert coverage_score(record, intent) == 2.0 / 3.0


def test_anchor_and_skip_scores_are_normalized_and_reward_novelty() -> None:
    config = ScoringConfig(time_decay=10.0)
    intent = DecisionIntent(aspects={"beneficiary_novelty", "large_transfer"})
    target = _record(
        "target",
        8.0,
        keys={"entity:account:a", "attr:channel:wire", "ctx:region:us"},
        aspects={"large_transfer"},
        vector=[1.0, 0.0, 1.0],
    )
    anchor = _record(
        "anchor",
        1.0,
        keys={"entity:account:a", "attr:channel:wire", "ctx:region:us"},
        aspects={"beneficiary_novelty", "large_transfer"},
        vector=[1.0, 0.0, 1.0],
    )
    overlapping_existing = [
        ChainSummary(
            chain_id="c1",
            family="txn",
            head_id="anchor",
            tail_id="target",
            representative_event_ids=["anchor"],
            aspects={"beneficiary_novelty", "large_transfer"},
        )
    ]

    fresh_anchor_score = anchor_score(anchor, intent, existing_anchors=())
    repeated_anchor_score = anchor_score(anchor, intent, existing_anchors=overlapping_existing)
    skip_value = skip_score(anchor, target, intent, ordinary_predecessors=overlapping_existing, config=config)

    assert 0.0 <= fresh_anchor_score <= 1.0
    assert 0.0 <= repeated_anchor_score <= 1.0
    assert 0.0 <= skip_value <= 1.0
    assert fresh_anchor_score > repeated_anchor_score


def test_skip_score_rewards_anchor_that_bridges_into_query_source() -> None:
    config = ScoringConfig(time_decay=10.0)
    intent = DecisionIntent(aspects={"source_accumulation", "full_balance_transfer"})
    target = _record(
        "target",
        10.0,
        keys={
            "entity:src_account=a",
            "entity:dst_account=b",
            "participant:a",
            "participant:b",
            "flow_src:a",
            "flow_dst:b",
            "flow_pair:a->b",
        },
        aspects={"full_balance_transfer"},
        vector=[1.0, 0.0, 1.0],
        attrs={"src_account": "A", "dst_account": "B"},
    )
    bridged_anchor = _record(
        "anchor1",
        4.0,
        keys={
            "entity:src_account=x",
            "entity:dst_account=a",
            "participant:x",
            "participant:a",
            "flow_src:x",
            "flow_dst:a",
            "flow_pair:x->a",
        },
        aspects={"source_accumulation"},
        vector=[1.0, 0.0, 1.0],
        attrs={"src_account": "X", "dst_account": "A"},
    )
    weak_anchor = _record(
        "anchor2",
        4.0,
        keys={
            "entity:src_account=m",
            "entity:dst_account=n",
            "participant:m",
            "participant:n",
            "flow_src:m",
            "flow_dst:n",
            "flow_pair:m->n",
        },
        aspects={"generic_evidence"},
        vector=[1.0, 0.0, 1.0],
        attrs={"src_account": "M", "dst_account": "N"},
    )
    local_predecessor = _record(
        "pred",
        9.0,
        keys={
            "entity:src_account=a",
            "entity:dst_account=b",
            "participant:a",
            "participant:b",
            "flow_src:a",
            "flow_dst:b",
            "flow_pair:a->b",
        },
        aspects={"generic_evidence"},
        vector=[1.0, 0.0, 1.0],
        attrs={"src_account": "A", "dst_account": "B"},
    )

    bridged_score = skip_score(bridged_anchor, target, intent, ordinary_predecessors=[local_predecessor], config=config)
    weak_score = skip_score(weak_anchor, target, intent, ordinary_predecessors=[local_predecessor], config=config)

    assert bridged_score > weak_score


def test_skip_score_breakdown_matches_skip_score() -> None:
    config = ScoringConfig(time_decay=10.0)
    intent = DecisionIntent(aspects={"source_accumulation", "full_balance_transfer"})
    target = _record(
        "target",
        10.0,
        keys={"entity:src_account=a", "entity:dst_account=b", "flow_pair:a->b"},
        aspects={"full_balance_transfer"},
        vector=[1.0, 0.0, 1.0],
        attrs={"src_account": "A", "dst_account": "B"},
    )
    anchor = _record(
        "anchor",
        4.0,
        keys={"entity:src_account=x", "entity:dst_account=a", "flow_pair:x->a"},
        aspects={"source_accumulation"},
        vector=[1.0, 0.0, 1.0],
        attrs={"src_account": "X", "dst_account": "A"},
        rarity=0.5,
    )
    local_predecessor = _record(
        "pred",
        9.0,
        keys={"entity:src_account=a", "entity:dst_account=b", "flow_pair:a->b"},
        aspects={"generic_evidence"},
        vector=[1.0, 0.0, 1.0],
        attrs={"src_account": "A", "dst_account": "B"},
    )

    breakdown = skip_score_breakdown(anchor, target, intent, ordinary_predecessors=[local_predecessor], config=config)

    assert {
        "corr",
        "impact",
        "novelty",
        "anchor_value",
        "best_ordinary_value",
        "bridge",
        "participant_bridge",
        "cost",
        "generic_penalty",
        "score",
    } <= set(breakdown)
    assert breakdown["score"] == skip_score(anchor, target, intent, ordinary_predecessors=[local_predecessor], config=config)


def test_skip_score_penalizes_lanl_fanout_without_destination_bridge() -> None:
    config = ScoringConfig(time_decay=100.0, skip_lanl_temporal_gain_scale=400.0)
    intent = DecisionIntent(aspects={"lateral_movement", "credential_reuse"})
    target = _record(
        "target",
        1000.0,
        keys={"entity:user:u1", "type:authentication"},
        aspects={"lateral_movement"},
        vector=[1.0, 0.0, 0.0],
        attrs={
            "src_user": "u1",
            "src_computer": "c17693",
            "dst_computer": "c9000",
            "auth_type": "NTLM",
            "is_new_dst_for_user": True,
            "prior_user_event_count": 80,
            "prior_user_host_count": 16,
        },
    )
    fanout_anchor = _record(
        "fanout",
        997.0,
        keys={"entity:user:u1", "type:authentication"},
        aspects={"credential_reuse"},
        vector=[1.0, 0.0, 0.0],
        attrs={
            "src_user": "u1",
            "src_computer": "c17693",
            "dst_computer": "c500",
            "auth_type": "NTLM",
        },
    )
    destination_bridge_anchor = _record(
        "bridge",
        995.0,
        keys={"entity:user:u1", "type:authentication"},
        aspects={"lateral_movement"},
        vector=[1.0, 0.0, 0.0],
        attrs={
            "src_user": "u1",
            "src_computer": "c9000",
            "dst_computer": "c9000",
            "auth_type": "NTLM",
        },
    )
    ordinary_predecessors = [
        _record(
            "pred1",
            999.0,
            keys={"entity:user:u1", "type:authentication"},
            aspects={"credential_reuse"},
            vector=[1.0, 0.0, 0.0],
            attrs={"src_user": "u1", "src_computer": "c17693", "dst_computer": "c501", "auth_type": "NTLM"},
        ),
        _record(
            "pred2",
            998.0,
            keys={"entity:user:u1", "type:authentication"},
            aspects={"credential_reuse"},
            vector=[1.0, 0.0, 0.0],
            attrs={"src_user": "u1", "src_computer": "c17693", "dst_computer": "c502", "auth_type": "NTLM"},
        ),
    ]

    fanout_breakdown = skip_score_breakdown(fanout_anchor, target, intent, ordinary_predecessors, config)
    bridge_breakdown = skip_score_breakdown(destination_bridge_anchor, target, intent, ordinary_predecessors, config)

    assert fanout_breakdown["lanl_fanout_penalty"] > 0.0
    assert bridge_breakdown["lanl_bridge_bonus"] > 0.0
    assert skip_score(destination_bridge_anchor, target, intent, ordinary_predecessors, config) > skip_score(
        fanout_anchor,
        target,
        intent,
        ordinary_predecessors,
        config,
    )


def test_skip_score_rewards_lanl_temporal_bridge_gain_and_bootstrap_recovery() -> None:
    config = ScoringConfig(time_decay=100.0, skip_lanl_temporal_gain_scale=200.0)
    intent = DecisionIntent(aspects={"lateral_movement"})
    target = _record(
        "target",
        500.0,
        keys={"entity:user:u2", "type:authentication"},
        aspects={"lateral_movement"},
        vector=[1.0, 0.0, 0.0],
        attrs={
            "src_user": "u2",
            "src_computer": "c17693",
            "dst_computer": "c801",
            "auth_type": "NTLM",
            "is_new_dst_for_user": True,
            "prior_user_event_count": 0,
            "prior_user_host_count": 0,
        },
    )
    bootstrap_anchor = _record(
        "bootstrap",
        420.0,
        keys={"entity:user:u2", "type:authentication"},
        aspects={"lateral_movement"},
        vector=[1.0, 0.0, 0.0],
        attrs={
            "src_user": "u2",
            "src_computer": "c17693",
            "dst_computer": "c2731",
            "auth_type": "NTLM",
        },
    )
    recent_nonbridge = _record(
        "recent",
        495.0,
        keys={"entity:user:u3", "type:authentication"},
        aspects={"generic_evidence"},
        vector=[1.0, 0.0, 0.0],
        attrs={
            "src_user": "u3",
            "src_computer": "c400",
            "dst_computer": "c401",
            "auth_type": "NTLM",
        },
    )

    bootstrap_breakdown = skip_score_breakdown(bootstrap_anchor, target, intent, ordinary_predecessors=[], config=config)
    recent_breakdown = skip_score_breakdown(recent_nonbridge, target, intent, ordinary_predecessors=[], config=config)

    assert bootstrap_breakdown["lanl_temporal_gain"] > 0.0
    assert bootstrap_breakdown["lanl_bridge_bonus"] > 0.0
    assert skip_score(bootstrap_anchor, target, intent, ordinary_predecessors=[], config=config) > skip_score(
        recent_nonbridge,
        target,
        intent,
        ordinary_predecessors=[],
        config=config,
    )


def test_retrieval_marginal_utility_and_priority_are_bounded_and_cost_aware() -> None:
    config = ScoringConfig()
    intent = DecisionIntent(aspects={"source_accumulation", "full_balance_transfer"})
    query = EventQuery(
        event=Event(event_id="q", time=10.0, event_type="transaction"),
        intent=intent,
        budget=3,
    )
    candidate = EvidenceObject(
        object_id="skip-1",
        event_ids=["e1", "e2"],
        aspects={"source_accumulation", "full_balance_transfer"},
        summary="early accumulation",
        cost=1.0,
    )
    selected = [
        EvidenceObject(
            object_id="local-1",
            event_ids=["e2"],
            aspects={"source_accumulation"},
            summary="recent activity",
            cost=1.0,
        )
    ]

    marginal = retrieval_marginal_utility(candidate, selected, query, intent, config)
    cheaper = candidate_priority(marginal, cost=1.0, eta=1e-8)
    expensive = candidate_priority(marginal, cost=5.0, eta=1e-8)

    assert 0.0 <= marginal <= 1.0
    assert cheaper > expensive


def test_non_crashing_edge_cases_with_empty_inputs() -> None:
    config = ScoringConfig()
    intent = DecisionIntent()
    empty_record = EventRecord(event=Event(event_id="empty", time="unknown", event_type="generic"))
    chain = ChainSummary(chain_id="c", family="generic", head_id="h", tail_id="t")
    query = EventQuery(event=Event(event_id="q", time=0, event_type="generic"), intent=intent, budget=1)
    evidence = EvidenceObject(object_id="obj")

    assert 0.0 <= dependency_score(empty_record, empty_record, config) <= 1.0
    assert 0.0 <= anchor_score(chain, intent, existing_anchors=()) <= 1.0
    assert 0.0 <= skip_score(chain, empty_record, intent, ordinary_predecessors=(), config=config) <= 1.0
    assert 0.0 <= retrieval_marginal_utility(evidence, [], query, intent, config) <= 1.0
