"""Benchmark metrics for the IBM AML evaluation slice."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from statistics import mean, median, pstdev
from typing import Any


_LAUNDERING_KEYWORDS = (
    "launder",
    "layer",
    "structur",
    "smurf",
    "shell",
    "round-trip",
    "round trip",
    "funnel",
    "rapid movement",
)

_AMOUNT_KEYS = ("amount", "transaction_amount", "value", "usd_amount")
_ENTITY_KEYS = ("entity_id", "entity_ids", "account_id", "account_ids", "customer_id", "customer_ids")


def evidence_recall_at_budget(retrieved_ids: Sequence[Any], gold_ids: Iterable[Any]) -> float:
    """Compute recall over unique retrieved ids."""

    gold = _normalize_ids(gold_ids)
    if not gold:
        return 1.0
    retrieved = _normalize_ids(retrieved_ids)
    return len(gold & retrieved) / len(gold)


def evidence_precision_at_budget(retrieved_ids: Sequence[Any], gold_ids: Iterable[Any]) -> float:
    """Compute precision over unique retrieved ids."""

    retrieved = _normalize_ids(retrieved_ids)
    gold = _normalize_ids(gold_ids)
    if not retrieved:
        return 1.0 if not gold else 0.0
    return len(gold & retrieved) / len(retrieved)


def evidence_f1_at_budget(retrieved_ids: Sequence[Any], gold_ids: Iterable[Any]) -> float:
    """Compute the harmonic mean of recall and precision."""

    recall = evidence_recall_at_budget(retrieved_ids, gold_ids)
    precision = evidence_precision_at_budget(retrieved_ids, gold_ids)
    if recall == 0.0 and precision == 0.0:
        return 0.0
    return 2.0 * recall * precision / (recall + precision)


def context_efficiency(
    retrieved_ids: Sequence[Any],
    gold_ids: Iterable[Any],
    budget_used: int | float | None = None,
) -> float:
    """Measure recovered recall per unit of consumed budget."""

    effective_budget = float(len(_normalize_ids(retrieved_ids)) if budget_used is None else budget_used)
    if effective_budget <= 0.0:
        return 0.0
    return evidence_recall_at_budget(retrieved_ids, gold_ids) / effective_budget


def mean_latency_ms(latencies_ms: Iterable[int | float]) -> float:
    """Return the arithmetic mean latency in milliseconds."""

    values = [float(value) for value in latencies_ms if isfinite(float(value))]
    if not values:
        return 0.0
    return mean(values)


def update_throughput_events_per_sec(
    num_events: int | None = None,
    elapsed_seconds: int | float | None = None,
    *,
    event_timestamps_s: Sequence[int | float] | None = None,
) -> float:
    """Compute event-update throughput."""

    if event_timestamps_s is not None:
        timestamps = [float(value) for value in event_timestamps_s]
        if not timestamps:
            return 0.0
        if len(timestamps) == 1:
            return 1.0
        span = max(timestamps) - min(timestamps)
        if span <= 0.0:
            return float(len(timestamps))
        return len(timestamps) / span

    if num_events is None or elapsed_seconds is None:
        raise ValueError("num_events and elapsed_seconds are required when event_timestamps_s is not provided")

    if elapsed_seconds <= 0:
        return 0.0 if num_events <= 0 else float(num_events)
    return float(num_events) / float(elapsed_seconds)


def index_size_stats(index_sizes: Mapping[str, int | float] | Sequence[int | float]) -> dict[str, float]:
    """Summarize index size measurements."""

    if isinstance(index_sizes, Mapping):
        values = [float(value) for value in index_sizes.values()]
        stats = _numeric_stats(values)
        stats["components"] = float(len(index_sizes))
        return stats

    values = [float(value) for value in index_sizes]
    return _numeric_stats(values)


def decision_proxy_score(
    retrieved_evidence: Sequence[Any],
    *,
    positive_threshold: float = 0.55,
) -> dict[str, float | bool]:
    """Heuristic classifier over retrieved evidence features."""

    laundering_like_count = 0
    total_amount = 0.0
    entity_counter: Counter[str] = Counter()
    unique_aspects: set[str] = set()

    for evidence in retrieved_evidence:
        text_blob = _evidence_text(evidence).lower()
        aspects = _evidence_aspects(evidence)
        amount = _evidence_amount(evidence)
        entities = _evidence_entities(evidence)

        if any(keyword in text_blob for keyword in _LAUNDERING_KEYWORDS) or any(
            any(keyword in aspect.lower() for keyword in _LAUNDERING_KEYWORDS)
            for aspect in aspects
        ):
            laundering_like_count += 1

        total_amount += amount
        entity_counter.update(entities)
        unique_aspects.update(aspects)

    repeated_entities = sum(1 for count in entity_counter.values() if count > 1)
    shared_entity_instances = sum(count - 1 for count in entity_counter.values() if count > 1)
    entity_overlap = float(repeated_entities + shared_entity_instances)
    aspect_coverage = float(len(unique_aspects))

    laundering_signal = min(laundering_like_count / 3.0, 1.0)
    amount_signal = min(total_amount / 100_000.0, 1.0)
    entity_signal = min(entity_overlap / 3.0, 1.0)
    coverage_signal = min(aspect_coverage / 4.0, 1.0)

    raw_score = (
        0.35 * laundering_signal
        + 0.25 * amount_signal
        + 0.20 * entity_signal
        + 0.20 * coverage_signal
    )

    return {
        "laundering_like_count": float(laundering_like_count),
        "amount_sum": total_amount,
        "entity_overlap": entity_overlap,
        "aspect_coverage": aspect_coverage,
        "raw_score": raw_score,
        "predicted_positive": raw_score >= positive_threshold,
    }


def _normalize_ids(values: Iterable[Any]) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        if value is None:
            continue
        normalized.add(str(value))
    return normalized


def _numeric_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "total": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
        }

    return {
        "count": float(len(values)),
        "total": float(sum(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "std": float(pstdev(values)),
    }


def _evidence_mapping(evidence: Any) -> Mapping[str, Any]:
    if isinstance(evidence, Mapping):
        return evidence
    attrs = getattr(evidence, "attrs", None)
    if isinstance(attrs, Mapping):
        return attrs
    return {}


def _evidence_text(evidence: Any) -> str:
    mapping = _evidence_mapping(evidence)
    text_parts: list[str] = []
    for key in ("text", "summary", "label", "description"):
        value = mapping.get(key) if key in mapping else getattr(evidence, key, None)
        if value:
            text_parts.append(str(value))
    return " ".join(text_parts)


def _evidence_aspects(evidence: Any) -> set[str]:
    mapping = _evidence_mapping(evidence)
    raw = mapping.get("aspects", getattr(evidence, "aspects", set()))
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, Iterable):
        return {str(value) for value in raw}
    return set()


def _evidence_amount(evidence: Any) -> float:
    mapping = _evidence_mapping(evidence)
    for key in _AMOUNT_KEYS:
        if key in mapping:
            return _coerce_float(mapping[key])
        value = getattr(evidence, key, None)
        if value is not None:
            return _coerce_float(value)
    return 0.0


def _evidence_entities(evidence: Any) -> set[str]:
    mapping = _evidence_mapping(evidence)
    entities: set[str] = set()
    for key in _ENTITY_KEYS:
        raw = mapping.get(key) if key in mapping else getattr(evidence, key, None)
        if raw is None:
            continue
        if isinstance(raw, str):
            entities.add(raw)
            continue
        if isinstance(raw, Iterable):
            entities.update(str(value) for value in raw if value is not None)
    return entities


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
