"""Deterministic baselines for the IBM AML benchmark."""

from __future__ import annotations

from collections.abc import Sequence
import random
from typing import Any

import numpy as np

from timeindex.config import ExtractorConfig
from timeindex.event import Event, EventRecord, OrdinaryLink
from timeindex.extractors import compute_vector, extract_keys


def recent_window_retrieval(
    events: Sequence[Event],
    query_event: Event | str,
    budget: int,
    window: int,
) -> list[dict[str, Any]]:
    """Return the most recent prior events within a bounded lookback window."""

    candidates = _recent_predecessors(events, query_event, window)
    return [_event_result(event, score=float(window - offset)) for offset, event in enumerate(candidates[: max(budget, 0)])]


def same_entity_window_retrieval(
    events: Sequence[Event],
    query_event: Event | str,
    budget: int,
    window: int,
) -> list[dict[str, Any]]:
    """Return recent prior events sharing source or destination entities."""

    query = _resolve_query_event(events, query_event)
    if query is None:
        return []

    query_entities = _entity_values(query)
    if not query_entities:
        return []

    matches: list[tuple[int, Event]] = []
    for offset, event in enumerate(_recent_predecessors(events, query, window)):
        shared = query_entities & _entity_values(event)
        if not shared:
            continue
        matches.append((offset, event))
        if len(matches) >= max(budget, 0):
            break

    return [
        _event_result(
            event,
            score=float(window - offset),
            summary=f"Recent shared-entity event via {', '.join(sorted(query_entities & _entity_values(event)))}",
            result_type="same_entity_window",
        )
        for offset, event in matches
    ]


def flow_chain_retrieval(
    events: Sequence[Event],
    query_event: Event | str,
    budget: int,
    max_hops: int = 3,
) -> list[dict[str, Any]]:
    """Walk backward through interacting transaction endpoints, not just exact same-entity repeats."""

    query = _resolve_query_event(events, query_event)
    if query is None or budget <= 0:
        return []

    prior_events = _previous_events(events, query)
    frontier_accounts = _source_entities(query) | _destination_entities(query)
    visited_accounts = set(frontier_accounts)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for hop in range(max(max_hops, 0)):
        matches: list[tuple[float, float, str, Event, set[str]]] = []
        for event in reversed(prior_events):
            if event.event_id in selected_ids:
                continue
            source_accounts = _source_entities(event)
            destination_accounts = _destination_entities(event)
            handoff_accounts = destination_accounts & frontier_accounts
            source_reuse_accounts = source_accounts & frontier_accounts
            any_overlap_accounts = (source_accounts | destination_accounts) & frontier_accounts
            if not any_overlap_accounts:
                continue

            if handoff_accounts:
                score = 1.0
                matched_accounts = handoff_accounts
            elif source_reuse_accounts:
                score = 0.8
                matched_accounts = source_reuse_accounts
            else:
                score = 0.6
                matched_accounts = any_overlap_accounts

            matches.append((score, _time_value(event.time), event.event_id, event, matched_accounts))

        if not matches:
            break

        matches.sort(key=lambda item: (-item[0], -item[1], item[2]))
        next_frontier: set[str] = set()
        for score, _time, _event_id, event, matched_accounts in matches:
            if len(selected) >= budget:
                break
            selected_ids.add(event.event_id)
            next_frontier.update(_source_entities(event) | _destination_entities(event))
            selected.append(
                _event_result(
                    event,
                    score=score,
                    summary=f"Flow-chain interaction via {', '.join(sorted(matched_accounts))}",
                    result_type="flow_chain",
                )
            )
        if len(selected) >= budget or not next_frontier:
            break
        frontier_accounts = next_frontier - visited_accounts
        if not frontier_accounts:
            break
        visited_accounts.update(frontier_accounts)

    return selected[:budget]


def nearest_neighbor_retrieval(
    events: Sequence[Event],
    query_event: Event | str,
    budget: int,
) -> list[dict[str, Any]]:
    """Return prior events ranked by cosine similarity of TimeIndex sketches."""

    query = _resolve_query_event(events, query_event)
    if query is None or budget <= 0:
        return []

    config = ExtractorConfig()
    query_vector = _event_sketch(query, config)
    scored: list[tuple[float, float, str, Event]] = []

    for event in _previous_events(events, query):
        score = float(np.dot(query_vector, _event_sketch(event, config)))
        scored.append((score, _time_value(event.time), event.event_id, event))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [
        _event_result(
            event,
            score=score,
            summary=f"Nearest neighbor by sketch cosine similarity ({score:.3f})",
            result_type="nearest_neighbor",
        )
        for score, _, _, event in scored[:budget]
    ]


def chain_only_retrieval(
    index: Any,
    query_event_id: str,
    budget: int,
) -> list[dict[str, Any]]:
    """Walk incoming ordinary links only, never consulting skip links."""

    if budget <= 0:
        return []

    selected: list[dict[str, Any]] = []
    visited: set[str] = {query_event_id}
    frontier: list[str] = [query_event_id]

    while frontier and len(selected) < budget:
        current_id = frontier.pop(0)
        incoming = list(_ordinary_links(index, current_id))
        incoming.sort(key=lambda link: (-float(getattr(link, "score", 0.0)), getattr(link, "predecessor_id", "")))
        for link in incoming:
            predecessor_id = getattr(link, "predecessor_id", None)
            if predecessor_id is None or predecessor_id in visited:
                continue
            predecessor = _get_event(index, predecessor_id)
            if predecessor is None:
                continue
            visited.add(predecessor_id)
            frontier.append(predecessor_id)
            selected.append(
                _event_result(
                    predecessor,
                    score=float(getattr(link, "score", 0.0)),
                    summary=f"Ordinary-link predecessor of {current_id}",
                    result_type="chain_only",
                )
            )
            if len(selected) >= budget:
                break

    return selected


def random_retrieval(
    events: Sequence[Event],
    query_event: Event | str,
    budget: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return a deterministic random sample of prior events."""

    candidates = list(_previous_events(events, query_event))
    if budget <= 0 or not candidates:
        return []

    generator = random.Random(seed)
    sample_size = min(len(candidates), budget)
    selected = generator.sample(candidates, sample_size)
    selected.sort(key=lambda event: (-_time_value(event.time), event.event_id))
    return [
        _event_result(
            event,
            score=1.0,
            summary=f"Random prior event sampled with seed={seed}",
            result_type="random",
        )
        for event in selected
    ]


def _previous_events(events: Sequence[Event], query_event: Event | str) -> list[Event]:
    query = _resolve_query_event(events, query_event)
    if query is None:
        return []

    query_index = _query_index(events, query)
    if query_index is not None:
        return [event for event in events[:query_index] if _time_value(event.time) <= _time_value(query.time)]

    return [event for event in events if event.event_id != query.event_id and _time_value(event.time) < _time_value(query.time)]


def _recent_predecessors(events: Sequence[Event], query_event: Event | str, window: int) -> list[Event]:
    if window <= 0:
        return []
    previous = _previous_events(events, query_event)
    return list(reversed(previous[-window:]))


def _resolve_query_event(events: Sequence[Event], query_event: Event | str) -> Event | None:
    if isinstance(query_event, Event):
        return query_event
    for event in events:
        if event.event_id == query_event:
            return event
    return None


def _query_index(events: Sequence[Event], query: Event) -> int | None:
    for index, event in enumerate(events):
        if event.event_id == query.event_id:
            return index
    return None


def _entity_values(event: Event) -> set[str]:
    return _source_entities(event) | _destination_entities(event)


def _source_entities(event: Event) -> set[str]:
    values: set[str] = set()
    for field in ("account_id", "src_account", "source_account", "from_account", "origin_account", "sender_account"):
        value = event.attrs.get(field)
        if value not in (None, ""):
            values.add(str(value))
    return values


def _destination_entities(event: Event) -> set[str]:
    values: set[str] = set()
    for field in (
        "dst_account",
        "destination_account",
        "to_account",
        "beneficiary_account",
        "beneficiary_id",
        "counterparty_account",
        "recipient_account",
        "receiver_account",
        "target_account",
    ):
        value = event.attrs.get(field)
        if value not in (None, ""):
            values.add(str(value))
    return values


def _event_sketch(event: Event, config: ExtractorConfig) -> np.ndarray:
    keys = extract_keys(event, config)
    return compute_vector(event, keys, dim=config.sketch_dim)


def _ordinary_links(index: Any, event_id: str) -> Sequence[OrdinaryLink]:
    for owner in (getattr(index, "edge_store", None), index):
        if owner is None:
            continue
        for name in ("ordinary_links", "incoming", "In"):
            method = getattr(owner, name, None)
            if callable(method):
                links = method(event_id)
                if links is not None:
                    return list(links)
    return []


def _get_event(index: Any, event_id: str) -> Event | None:
    for owner in (getattr(index, "event_store", None), index):
        if owner is None:
            continue
        for name in ("get_event", "get"):
            method = getattr(owner, name, None)
            if not callable(method):
                continue
            value = method(event_id)
            if isinstance(value, EventRecord):
                return value.event
            if isinstance(value, Event):
                return value
    return None


def _event_result(
    event: Event,
    score: float,
    summary: str | None = None,
    result_type: str = "event",
) -> dict[str, Any]:
    return {
        "event_ids": [event.event_id],
        "score": float(score),
        "type": result_type,
        "summary": summary or f"Retrieved prior event {event.event_id}",
    }


def _time_value(value: str | int | float) -> float:
    if isinstance(value, str):
        digits = "".join(character for character in value if character.isdigit())
        return float(digits or 0.0)
    return float(value)
