"""Synthetic evaluation helpers for the TimeIndex prototype."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .event import EvidenceObject, Event


def evidence_recall(retrieved: Sequence[Any], gold_event_ids: Iterable[str]) -> float:
    """Compute event-level recall against a gold set."""

    gold = set(gold_event_ids)
    if not gold:
        return 1.0

    retrieved_ids: set[str] = set()
    for item in retrieved:
        retrieved_ids.update(_extract_event_ids(item))

    return len(gold & retrieved_ids) / len(gold)


def _extract_event_ids(item: Any) -> set[str]:
    if isinstance(item, Event):
        return {item.event_id}

    if isinstance(item, EvidenceObject):
        return set(item.event_ids)

    if hasattr(item, "event_ids"):
        event_ids = getattr(item, "event_ids")
        if isinstance(event_ids, Sequence) and not isinstance(event_ids, (str, bytes)):
            return {str(event_id) for event_id in event_ids}

    if hasattr(item, "representative_event_ids"):
        event_ids = getattr(item, "representative_event_ids")
        if isinstance(event_ids, Sequence) and not isinstance(event_ids, (str, bytes)):
            return {str(event_id) for event_id in event_ids}

    if hasattr(item, "event_id"):
        return {str(getattr(item, "event_id"))}

    if isinstance(item, str):
        return {item}

    return set()
