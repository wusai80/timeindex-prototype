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
)


def _record(
    event_id: str,
    time: float,
    *,
    keys: set[str],
    aspects: set[str],
    vector: list[float],
    rarity: float = 0.0,
    surprise: float = 0.0,
) -> EventRecord:
    return EventRecord(
        event=Event(event_id=event_id, time=time, event_type="transaction"),
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
