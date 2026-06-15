"""Scoring functions for the TimeIndex prototype."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import exp
from typing import Any

import numpy as np

from .config import ScoringConfig
from .event import ChainSummary, DecisionIntent, EventQuery, EventRecord, EvidenceObject, SkipLink


def _clip01(value: float) -> float:
    """Clamp a score into the normalized [0, 1] range."""

    return max(0.0, min(1.0, float(value)))


def _as_set(values: Iterable[str] | None) -> set[str]:
    if values is None:
        return set()
    return set(values)


def _key_subset(record: EventRecord, prefixes: tuple[str, ...]) -> set[str]:
    return {key for key in record.lookup_keys if key.startswith(prefixes)}


def _event_time(record: EventRecord) -> float | None:
    raw_time = record.event.time
    if isinstance(raw_time, (int, float)):
        return float(raw_time)
    if isinstance(raw_time, str):
        try:
            return float(raw_time)
        except ValueError:
            return None
    return None


def _time_decay(candidate: EventRecord, target: EventRecord, config: ScoringConfig) -> float:
    candidate_time = _event_time(candidate)
    target_time = _event_time(target)
    if candidate_time is None or target_time is None:
        return 0.0

    scale = max(float(config.time_decay), 1e-8)
    gap = abs(target_time - candidate_time)
    return _clip01(exp(-gap / scale))


def _vector_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    return cosine(left, right)


def _event_ids(obj: Any) -> set[str]:
    if isinstance(obj, EvidenceObject):
        return set(obj.event_ids)
    if isinstance(obj, ChainSummary):
        return {obj.head_id, obj.tail_id, *obj.representative_event_ids}
    if isinstance(obj, SkipLink):
        return {obj.from_id, obj.to_id, *obj.representative_event_ids}
    if isinstance(obj, EventRecord):
        return {obj.event.event_id}
    return set()


def _object_aspects(obj: Any) -> set[str]:
    return set(getattr(obj, "aspects", set()))


def _object_cost(obj: Any) -> float:
    raw_cost = getattr(obj, "cost", 1.0)
    return max(float(raw_cost), 0.0)


def _weighted_overlap(values: set[str], intent: DecisionIntent) -> float:
    if not intent.aspects:
        return 1.0
    total_weight = sum(intent.aspect_weights.get(aspect, 1.0) for aspect in intent.aspects)
    if total_weight <= 0.0:
        return 0.0

    matched_weight = sum(intent.aspect_weights.get(aspect, 1.0) for aspect in values & intent.aspects)
    return _clip01(matched_weight / total_weight)


def _novelty_against_existing(obj: Any, existing_objects: Sequence[Any]) -> float:
    if not existing_objects:
        return 1.0

    object_ids = _event_ids(obj)
    object_aspects = _object_aspects(obj)
    overlaps: list[float] = []

    for existing in existing_objects:
        id_overlap = jaccard(object_ids, _event_ids(existing))
        aspect_overlap = jaccard(object_aspects, _object_aspects(existing))
        overlaps.append(max(id_overlap, aspect_overlap))

    if not overlaps:
        return 1.0
    return _clip01(1.0 - max(overlaps))


def jaccard(set_a: Iterable[str] | None, set_b: Iterable[str] | None) -> float:
    """Return the normalized Jaccard overlap between two sets."""

    left = _as_set(set_a)
    right = _as_set(set_b)
    union = left | right
    if not union:
        return 1.0
    return _clip01(len(left & right) / len(union))


def cosine(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float:
    """Return a normalized cosine similarity in [0, 1]."""

    if vec_a is None or vec_b is None:
        return 0.0

    left = np.asarray(vec_a, dtype=float).reshape(-1)
    right = np.asarray(vec_b, dtype=float).reshape(-1)
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return 0.0

    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    score = float(np.dot(left, right) / (left_norm * right_norm))
    return _clip01((score + 1.0) / 2.0)


def rarity_score(event: EventRecord, key_frequencies: dict[str, int], history_size: int) -> float:
    """Estimate rarity from inverse key frequency over the active history."""

    if history_size <= 0:
        return _clip01(max(event.metadata.rarity, event.metadata.surprise))

    if not event.lookup_keys:
        return _clip01(max(event.metadata.rarity, event.metadata.surprise))

    normalized_frequencies: list[float] = []
    denominator = max(float(history_size), 1.0)
    for key in event.lookup_keys:
        frequency = max(int(key_frequencies.get(key, 0)), 0)
        normalized_frequencies.append(1.0 - min(frequency / denominator, 1.0))

    rarity = sum(normalized_frequencies) / len(normalized_frequencies)
    metadata_signal = max(event.metadata.rarity, event.metadata.surprise)
    return _clip01(max(rarity, metadata_signal))


def dependency_score(candidate: EventRecord, target: EventRecord, config: ScoringConfig) -> float:
    """Score a local predecessor relationship using weighted normalized features."""

    entity_overlap = jaccard(
        _key_subset(candidate, ("entity:",)),
        _key_subset(target, ("entity:",)),
    )
    attribute_overlap = jaccard(
        _key_subset(candidate, ("attr:", "binned:")),
        _key_subset(target, ("attr:", "binned:")),
    )
    context_overlap = jaccard(
        _key_subset(candidate, ("ctx:", "type:", "time:")),
        _key_subset(target, ("ctx:", "type:", "time:")),
    )
    temporal_score = _time_decay(candidate, target, config)
    vector_score = _vector_similarity(candidate.sketch, target.sketch)
    deviation_score = _clip01(max(candidate.metadata.rarity, candidate.metadata.surprise))

    score = (
        0.30 * entity_overlap
        + 0.20 * attribute_overlap
        + 0.15 * context_overlap
        + 0.15 * temporal_score
        + 0.15 * vector_score
        + 0.05 * deviation_score
    )
    return _clip01(score)


def impact_score(event_or_chain: EventRecord | ChainSummary | EvidenceObject | SkipLink, intent: DecisionIntent) -> float:
    """Measure how much the object's aspects align with the intent."""

    return _weighted_overlap(_object_aspects(event_or_chain), intent)


def coverage_score(event_or_chain: EventRecord | ChainSummary | EvidenceObject | SkipLink, intent: DecisionIntent) -> float:
    """Measure the fraction of the intent covered by this object."""

    return _weighted_overlap(_object_aspects(event_or_chain), intent)


def anchor_score(
    object: EventRecord | ChainSummary | EvidenceObject | SkipLink,
    intent: DecisionIntent,
    existing_anchors: Sequence[EventRecord | ChainSummary | EvidenceObject | SkipLink],
) -> float:
    """Score an event or chain as a candidate skip anchor."""

    score = (
        0.45 * impact_score(object, intent)
        + 0.35 * coverage_score(object, intent)
        + 0.20 * _novelty_against_existing(object, existing_anchors)
    )
    return _clip01(score)


def skip_score(
    anchor: EventRecord | ChainSummary | EvidenceObject | SkipLink,
    target: EventRecord,
    intent: DecisionIntent,
    ordinary_predecessors: Sequence[EventRecord | ChainSummary | EvidenceObject | SkipLink],
    config: ScoringConfig,
) -> float:
    """Score a long-range skip link candidate against a target event."""

    if isinstance(anchor, EventRecord):
        relevance = dependency_score(anchor, target, config)
    else:
        relevance = _weighted_overlap(_object_aspects(anchor), intent)

    score = (
        0.40 * impact_score(anchor, intent)
        + 0.25 * coverage_score(anchor, intent)
        + 0.20 * relevance
        + 0.15 * _novelty_against_existing(anchor, ordinary_predecessors)
    )
    return _clip01(score)


def retrieval_marginal_utility(
    candidate: EvidenceObject | SkipLink | ChainSummary,
    selected: Sequence[EvidenceObject],
    target: EventQuery,
    intent: DecisionIntent,
    config: ScoringConfig,
) -> float:
    """Estimate the marginal utility of adding a candidate to the retrieval set."""

    del config  # Reserved for future formula tuning.

    selected_aspects = set().union(*(item.aspects for item in selected)) if selected else set()
    candidate_aspects = _object_aspects(candidate)

    if intent.aspects:
        total_weight = sum(intent.aspect_weights.get(aspect, 1.0) for aspect in intent.aspects)
        gained_weight = sum(
            intent.aspect_weights.get(aspect, 1.0)
            for aspect in (candidate_aspects & intent.aspects) - selected_aspects
        )
        incremental_coverage = 0.0 if total_weight <= 0.0 else gained_weight / total_weight
    else:
        incremental_coverage = 1.0 if candidate_aspects - selected_aspects else 0.0

    novelty = _novelty_against_existing(candidate, selected)
    target_relevance = _weighted_overlap(candidate_aspects, target.intent)

    score = 0.40 * target_relevance + 0.35 * incremental_coverage + 0.25 * novelty
    return _clip01(score)


def candidate_priority(marginal_utility: float, cost: float, eta: float) -> float:
    """Convert a marginal utility into a cost-aware priority."""

    denominator = max(float(cost) + float(eta), 1e-8)
    return max(float(marginal_utility), 0.0) / denominator


class PrototypeScorer:
    """Scorer for local links, anchors, skips, and retrieval utility."""

    def __init__(self, config: ScoringConfig) -> None:
        self.config = config

    def score_local_dependency(self, predecessor: EventRecord, current: EventRecord) -> float:
        return dependency_score(predecessor, current, self.config)

    def score_anchor(self, candidate: EventRecord | ChainSummary, intent: DecisionIntent) -> float:
        return anchor_score(candidate, intent, existing_anchors=())

    def score_skip(self, anchor: EventRecord | ChainSummary, query: EventRecord, intent: DecisionIntent) -> float:
        return skip_score(anchor, query, intent, ordinary_predecessors=(), config=self.config)

    def marginal_utility(
        self,
        candidate: EvidenceObject | SkipLink | ChainSummary,
        selected: Sequence[EvidenceObject],
        query: EventQuery,
    ) -> float:
        return retrieval_marginal_utility(candidate, selected, query, query.intent, self.config)
