"""Deterministic stand-in agent for IBM AML retrieval experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from benchmarks.metrics import evidence_precision_at_budget, evidence_recall_at_budget
from timeindex.event import Event


@dataclass(slots=True)
class FakeAgentDecision:
    """One deterministic agent judgment for a query event."""

    query_event_id: str
    query_label: str | None
    predicted_positive: bool
    decision_score: float
    laundering_support_count: int
    retrieved_count: int
    total_retrieved_amount: float
    entity_overlap_count: int
    aspect_coverage: int
    evidence_recall: float
    evidence_precision: float
    rationale: str
    retrieved_event_ids: list[str]
    retrieved_aspects: list[str]
    gold_event_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""

        return asdict(self)


def classify_query_from_retrieval(
    query_event: Event,
    retrieved_event_ids: Sequence[str],
    retrieved_aspects: Iterable[str],
    event_lookup: Mapping[str, Event],
    gold_event_ids: Iterable[str] = (),
    positive_threshold: float = 0.40,
) -> FakeAgentDecision:
    """Produce a deterministic suspicious/not-suspicious judgment."""

    normalized_ids = []
    for event_id in retrieved_event_ids:
        event_id_text = str(event_id)
        if event_id_text == query_event.event_id or event_id_text not in event_lookup:
            continue
        if event_id_text not in normalized_ids:
            normalized_ids.append(event_id_text)

    retrieved_events = [event_lookup[event_id] for event_id in normalized_ids]
    query_entities = _event_entities(query_event)

    laundering_support_count = 0
    total_retrieved_amount = 0.0
    overlap_entities: set[str] = set()
    for event in retrieved_events:
        if _is_positive_label(event.label):
            laundering_support_count += 1
        total_retrieved_amount += _event_amount(event)
        overlap_entities.update(query_entities & _event_entities(event))

    aspect_set = {str(aspect) for aspect in retrieved_aspects if str(aspect)}
    coverage = len(aspect_set)
    gold_ids = [str(event_id) for event_id in gold_event_ids]
    recall = evidence_recall_at_budget(normalized_ids, gold_ids)
    precision = evidence_precision_at_budget(normalized_ids, gold_ids)

    laundering_signal = min(laundering_support_count / 2.0, 1.0)
    amount_signal = min(total_retrieved_amount / max(1.0, _event_amount(query_event) * 2.0), 1.0)
    overlap_signal = min(len(overlap_entities) / max(1, len(query_entities) or 1), 1.0)
    aspect_signal = min(coverage / 3.0, 1.0)
    score = (
        0.45 * laundering_signal
        + 0.20 * amount_signal
        + 0.20 * overlap_signal
        + 0.15 * aspect_signal
    )
    predicted_positive = score >= positive_threshold

    rationale_bits = [
        f"{laundering_support_count} laundering-labeled supports",
        f"{len(overlap_entities)} shared entities",
        f"{coverage} retrieved aspects",
        f"{total_retrieved_amount:.2f} total evidence amount",
    ]

    return FakeAgentDecision(
        query_event_id=query_event.event_id,
        query_label=query_event.label,
        predicted_positive=predicted_positive,
        decision_score=score,
        laundering_support_count=laundering_support_count,
        retrieved_count=len(normalized_ids),
        total_retrieved_amount=total_retrieved_amount,
        entity_overlap_count=len(overlap_entities),
        aspect_coverage=coverage,
        evidence_recall=recall,
        evidence_precision=precision,
        rationale=", ".join(rationale_bits),
        retrieved_event_ids=normalized_ids,
        retrieved_aspects=sorted(aspect_set),
        gold_event_ids=gold_ids,
    )


def summarize_decisions(decisions: Sequence[FakeAgentDecision]) -> dict[str, float]:
    """Aggregate a deterministic agent run."""

    if not decisions:
        return {
            "queries": 0.0,
            "predicted_positive_rate": 0.0,
            "mean_decision_score": 0.0,
            "mean_evidence_recall": 0.0,
            "mean_evidence_precision": 0.0,
        }

    predicted_positive_rate = sum(1 for decision in decisions if decision.predicted_positive) / len(decisions)
    return {
        "queries": float(len(decisions)),
        "predicted_positive_rate": predicted_positive_rate,
        "mean_decision_score": sum(decision.decision_score for decision in decisions) / len(decisions),
        "mean_evidence_recall": sum(decision.evidence_recall for decision in decisions) / len(decisions),
        "mean_evidence_precision": sum(decision.evidence_precision for decision in decisions) / len(decisions),
    }


def _is_positive_label(label: Any) -> bool:
    if isinstance(label, bool):
        return label
    if isinstance(label, (int, float)):
        return float(label) > 0.0
    return str(label or "").strip().lower() in {"1", "true", "yes", "y", "laundering", "suspicious", "fraud"}


def _event_entities(event: Event) -> set[str]:
    values: set[str] = set()
    for key in ("src_account", "dst_account", "account_id", "beneficiary_id", "beneficiary_account"):
        value = event.attrs.get(key)
        if value not in (None, ""):
            values.add(str(value))
    return values


def _event_amount(event: Event) -> float:
    for key in ("amount", "transaction_amount", "amount_paid", "value"):
        value = event.attrs.get(key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0
