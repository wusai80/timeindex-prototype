"""Synthetic data generation helpers for the TimeIndex prototype."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from .config import SyntheticConfig
from .event import DecisionIntent, Event


def synthetic_transaction_stream() -> list[Event]:
    """Return a deterministic seven-event transaction scenario."""

    return [
        Event(
            event_id="e1",
            time=1,
            event_type="deposit",
            attrs={
                "account_id": "A",
                "src_account": "employer_payroll",
                "dst_account": "A",
                "amount": 2500.0,
                "currency": "USD",
                "channel": "ach",
            },
            ctx={"customer_segment": "retail", "geo": "US"},
            text="Payroll deposit into account A",
            label="background_income",
        ),
        Event(
            event_id="e2",
            time=2,
            event_type="payment",
            attrs={
                "account_id": "A",
                "src_account": "A",
                "dst_account": "landlord_01",
                "amount": 1200.0,
                "currency": "USD",
                "merchant": "rent",
                "channel": "billpay",
            },
            ctx={"customer_segment": "retail", "geo": "US"},
            text="Routine rent payment from account A",
            label="routine_payment",
        ),
        Event(
            event_id="e3",
            time=3,
            event_type="deposit",
            attrs={
                "account_id": "A",
                "src_account": "marketplace_payout",
                "dst_account": "A",
                "amount": 1900.0,
                "currency": "USD",
                "channel": "ach",
            },
            ctx={"customer_segment": "retail", "geo": "US"},
            text="Marketplace payout deposit into account A",
            label="background_income",
        ),
        Event(
            event_id="e4",
            time=4,
            event_type="payment",
            attrs={
                "account_id": "A",
                "src_account": "A",
                "dst_account": "grocery_mart",
                "amount": 85.0,
                "currency": "USD",
                "merchant": "groceries",
                "channel": "card",
            },
            ctx={"customer_segment": "retail", "geo": "US"},
            text="Routine grocery card payment from account A",
            label="routine_payment",
        ),
        Event(
            event_id="e5",
            time=5,
            event_type="beneficiary_add",
            attrs={
                "account_id": "A",
                "beneficiary_account": "B",
                "beneficiary_status": "new",
                "amount": 25.0,
                "currency": "USD",
                "channel": "online",
            },
            ctx={"customer_segment": "retail", "geo": "US"},
            text="New beneficiary B added and verified with a small transfer",
            label="new_beneficiary",
        ),
        Event(
            event_id="e6",
            time=6,
            event_type="balance_snapshot",
            attrs={
                "account_id": "A",
                "balance": 3090.0,
                "available_balance": 3090.0,
                "currency": "USD",
                "status": "accumulated",
            },
            ctx={"customer_segment": "retail", "geo": "US"},
            text="Account A accumulated a high balance before transfer",
            label="balance_accumulation",
        ),
        Event(
            event_id="e7",
            time=7,
            event_type="transfer",
            attrs={
                "account_id": "A",
                "src_account": "A",
                "dst_account": "B",
                "beneficiary_account": "B",
                "amount": 3090.0,
                "balance_before": 3090.0,
                "balance_after": 0.0,
                "currency": "USD",
                "channel": "online",
            },
            ctx={"customer_segment": "retail", "geo": "US"},
            text="Full balance transfer from account A to new beneficiary B",
            label="full_balance_transfer",
        ),
    ]


def synthetic_log_stream() -> list[Event]:
    """Return a deterministic five-event infrastructure incident scenario."""

    return [
        Event(
            event_id="l1",
            time=1,
            event_type="deployment",
            attrs={"service": "checkout", "version": "v2.3.0", "host": "node-a"},
            ctx={"env": "prod", "cluster": "blue"},
            text="Checkout service deployed to production",
            label="deployment",
        ),
        Event(
            event_id="l2",
            time=2,
            event_type="config_change",
            attrs={"service": "checkout", "config_key": "timeout_ms", "value": "1500"},
            ctx={"env": "prod", "cluster": "blue"},
            text="Timeout configuration changed for checkout service",
            label="config_change",
        ),
        Event(
            event_id="l3",
            time=3,
            event_type="error",
            attrs={"service": "checkout", "upstream_service": "payments", "status": "502"},
            ctx={"env": "prod", "cluster": "blue"},
            text="Checkout receives upstream errors from payments service",
            label="upstream_error",
        ),
        Event(
            event_id="l4",
            time=4,
            event_type="metric",
            attrs={"service": "checkout", "metric": "cpu", "value": 0.96, "host": "node-a"},
            ctx={"env": "prod", "cluster": "blue"},
            text="Checkout CPU saturation observed on node-a",
            label="resource_saturation",
        ),
        Event(
            event_id="l5",
            time=5,
            event_type="timeout",
            attrs={"service": "checkout", "count": 42, "window": "5m"},
            ctx={"env": "prod", "cluster": "blue"},
            text="Repeated timeouts affect checkout requests",
            label="timeout",
        ),
    ]


def baseline_recent_window(events: Sequence[Event], query: Event | str, budget: int) -> list[Event]:
    """Return the most recent predecessors before the query event."""

    query_id = _query_id(query)
    selected: list[Event] = []
    for event in reversed(_events_before_query(events, query_id)):
        if len(selected) >= max(budget, 0):
            break
        selected.append(event)
    return selected


def baseline_nearest_neighbor(events: Sequence[Event], query: Event | str, budget: int) -> list[Event]:
    """Return the most similar prior events using lightweight token overlap."""

    query_event = _resolve_query_event(events, query)
    scored: list[tuple[float, float, str, Event]] = []

    for event in _events_before_query(events, query_event.event_id):
        score = _similarity_score(query_event, event)
        scored.append((score, -_time_value(event.time), event.event_id, event))

    scored.sort(reverse=True)
    return [event for score, _, _, event in scored[: max(budget, 0)] if score > 0.0]


def baseline_chain_only(index: Any, query: Event | str, budget: int) -> list[Event]:
    """Walk incoming ordinary links backward without using skip evidence."""

    query_id = _query_id(query)
    if budget <= 0:
        return []

    selected: list[Event] = []
    visited: set[str] = {query_id}
    frontier: list[str] = [query_id]

    while frontier and len(selected) < budget:
        current_id = frontier.pop(0)
        incoming = list(index.ordinary_links(current_id))
        incoming.sort(key=lambda link: (-link.score, _event_sort_key(link.predecessor_id)))
        for link in incoming:
            predecessor_id = getattr(link, "predecessor_id", None)
            if predecessor_id is None or predecessor_id in visited:
                continue
            visited.add(predecessor_id)
            predecessor = index.get_event(predecessor_id)
            if predecessor is None:
                continue
            selected.append(predecessor)
            frontier.append(predecessor_id)
            if len(selected) >= budget:
                break

    return selected


class SyntheticStreamGenerator:
    """Compatibility scaffold retained until the generator class is assigned."""

    def __init__(self, config: SyntheticConfig) -> None:
        self.config = config

    def generate_events(self) -> Sequence[Event]:
        raise NotImplementedError("Synthetic event generation is not implemented yet.")

    def generate_intents(self) -> Sequence[DecisionIntent]:
        raise NotImplementedError("Synthetic intent generation is not implemented yet.")


def _query_id(query: Event | str) -> str:
    return query.event_id if isinstance(query, Event) else query


def _resolve_query_event(events: Sequence[Event], query: Event | str) -> Event:
    if isinstance(query, Event):
        return query

    for event in events:
        if event.event_id == query:
            return event
    raise KeyError(f"query event not found: {query}")


def _events_before_query(events: Sequence[Event], query_id: str) -> list[Event]:
    ordered = list(events)
    for index, event in enumerate(ordered):
        if event.event_id == query_id:
            return ordered[:index]
    raise KeyError(f"query event not found: {query_id}")


def _event_tokens(event: Event) -> Counter[str]:
    tokens: Counter[str] = Counter()
    tokens.update(_flatten_mapping("attr", event.attrs))
    tokens.update(_flatten_mapping("ctx", event.ctx))
    tokens.update([f"type:{event.event_type}", f"label:{event.label or ''}"])
    if event.text:
        tokens.update(f"text:{token}" for token in event.text.lower().split())
    return tokens


def _flatten_mapping(prefix: str, mapping: dict[str, Any]) -> Iterable[str]:
    for key in sorted(mapping):
        value = mapping[key]
        yield f"{prefix}:{key}"
        yield f"{prefix}:{key}={value}"
        yield f"value:{value}"


def _overlap_score(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = sum(min(left[token], right[token]) for token in left.keys() & right.keys())
    left_total = sum(left.values())
    right_total = sum(right.values())
    denominator = left_total + right_total - intersection
    if denominator <= 0:
        return 0.0
    return intersection / denominator


def _similarity_score(query_event: Event, candidate: Event) -> float:
    query_tokens = _event_tokens(query_event)
    candidate_tokens = _event_tokens(candidate)
    score = _overlap_score(query_tokens, candidate_tokens)

    query_entities = _entity_values(query_event)
    candidate_entities = _entity_values(candidate)
    shared_entities = len(query_entities & candidate_entities)
    if shared_entities:
        score += 0.20 * shared_entities

    query_numbers = _numeric_values(query_event)
    candidate_numbers = _numeric_values(candidate)
    shared_numbers = len(query_numbers & candidate_numbers)
    if shared_numbers:
        score += 0.25 * shared_numbers

    return score


def _time_value(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except ValueError:
        return float(abs(hash(value)) % 1_000_000)


def _event_sort_key(event_id: str) -> tuple[int, str]:
    suffix = event_id[1:] if len(event_id) > 1 else event_id
    if suffix.isdigit():
        return (int(suffix), event_id)
    return (10**9, event_id)


def _entity_values(event: Event) -> set[str]:
    entity_values: set[str] = set()
    for key, value in event.attrs.items():
        lowered = key.lower()
        if any(marker in lowered for marker in ("account", "beneficiary", "service", "host", "device")):
            entity_values.add(str(value))
    return entity_values


def _numeric_values(event: Event) -> set[float]:
    values: set[float] = set()
    for value in event.attrs.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.add(float(value))
    return values
