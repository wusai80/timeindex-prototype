"""Gold evidence policies for the LANL auth benchmark slice."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

from timeindex.event import Event, EventRecord


SAME_USER_HISTORY = "same_user_history"
SAME_COMPUTER_HISTORY = "same_computer_history"
LATERAL_CHAIN = "lateral_chain"
NOVEL_ACCESS_BOOTSTRAP = "novel_access_bootstrap"
UNION = "union"


@dataclass(frozen=True, slots=True)
class NormalizedEvent:
    """Compact event view used by LANL gold policies."""

    event_id: str
    time: int
    label: bool
    src_user: str
    dst_user: str
    src_computer: str
    dst_computer: str


def build_gold_evidence(
    events: Iterable[Event | EventRecord | dict[str, Any]],
    *,
    policy: str = UNION,
    window: int = 86_400,
    max_hops: int = 2,
) -> dict[str, set[str]]:
    """Build gold evidence for positive LANL auth queries."""

    normalized = sorted((_coerce_event(item) for item in events), key=lambda event: (event.time, event.event_id))
    if policy == SAME_USER_HISTORY:
        return _same_user_history(normalized, window=window)
    if policy == SAME_COMPUTER_HISTORY:
        return _same_computer_history(normalized, window=window)
    if policy == LATERAL_CHAIN:
        return _lateral_chain(normalized, window=window, max_hops=max_hops)
    if policy == NOVEL_ACCESS_BOOTSTRAP:
        return _novel_access_bootstrap(normalized, window=window)
    if policy == UNION:
        merged: dict[str, set[str]] = {}
        for part in (
            _same_user_history(normalized, window=window),
            _same_computer_history(normalized, window=window),
            _lateral_chain(normalized, window=window, max_hops=max_hops),
            _novel_access_bootstrap(normalized, window=window),
        ):
            for query_id, support in part.items():
                merged.setdefault(query_id, set()).update(support)
        return merged
    raise ValueError(f"Unsupported LANL evidence policy: {policy}")


def _same_user_history(events: list[NormalizedEvent], *, window: int) -> dict[str, set[str]]:
    evidence: dict[str, set[str]] = {}
    for index, event in enumerate(events):
        if not event.label:
            continue
        support = {
            prior.event_id
            for prior in events[:index]
            if prior.src_user == event.src_user and _within_window(prior.time, event.time, window)
        }
        evidence[event.event_id] = support
    return evidence


def _same_computer_history(events: list[NormalizedEvent], *, window: int) -> dict[str, set[str]]:
    evidence: dict[str, set[str]] = {}
    for index, event in enumerate(events):
        if not event.label:
            continue
        query_computers = {event.src_computer, event.dst_computer}
        support = {
            prior.event_id
            for prior in events[:index]
            if _within_window(prior.time, event.time, window)
            and query_computers.intersection({prior.src_computer, prior.dst_computer})
        }
        evidence[event.event_id] = support
    return evidence


def _lateral_chain(events: list[NormalizedEvent], *, window: int, max_hops: int) -> dict[str, set[str]]:
    evidence: dict[str, set[str]] = {}
    for index, event in enumerate(events):
        if not event.label:
            continue
        prior_events = [prior for prior in events[:index] if _within_window(prior.time, event.time, window)]
        seeds = {
            f"user:{event.src_user}",
            f"computer:{event.src_computer}",
            f"computer:{event.dst_computer}",
        }
        support: set[str] = set()
        frontier: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds if seed.split(":", 1)[1])
        visited_nodes = set(seeds)
        while frontier:
            node, depth = frontier.popleft()
            if depth >= max(1, max_hops):
                continue
            for prior in prior_events:
                prior_nodes = {
                    f"user:{prior.src_user}",
                    f"user:{prior.dst_user}",
                    f"computer:{prior.src_computer}",
                    f"computer:{prior.dst_computer}",
                }
                if node not in prior_nodes:
                    continue
                support.add(prior.event_id)
                for next_node in prior_nodes:
                    if next_node in visited_nodes or not next_node.split(":", 1)[1]:
                        continue
                    visited_nodes.add(next_node)
                    frontier.append((next_node, depth + 1))
        evidence[event.event_id] = support
    return evidence


def _novel_access_bootstrap(events: list[NormalizedEvent], *, window: int) -> dict[str, set[str]]:
    seen_pairs: set[tuple[str, str]] = set()
    history_by_user: dict[str, list[NormalizedEvent]] = {}
    history_by_dst_computer: dict[str, list[NormalizedEvent]] = {}
    evidence: dict[str, set[str]] = {}
    for event in events:
        support: set[str] = set()
        pair = (event.src_user, event.dst_computer)
        if event.label and pair not in seen_pairs:
            for prior in reversed(history_by_user.get(event.src_user, [])):
                if _within_window(prior.time, event.time, window):
                    support.add(prior.event_id)
                    break
            for prior in reversed(history_by_dst_computer.get(event.dst_computer, [])):
                if _within_window(prior.time, event.time, window):
                    support.add(prior.event_id)
                    break
        if event.label:
            evidence[event.event_id] = support
        seen_pairs.add(pair)
        history_by_user.setdefault(event.src_user, []).append(event)
        history_by_dst_computer.setdefault(event.dst_computer, []).append(event)
    return evidence


def _coerce_event(item: Event | EventRecord | dict[str, Any]) -> NormalizedEvent:
    if isinstance(item, EventRecord):
        return _coerce_event(item.event)
    if isinstance(item, Event):
        return NormalizedEvent(
            event_id=item.event_id,
            time=int(item.time),
            label=_is_positive_label(item.label),
            src_user=_text_attr(item.attrs, "src_user"),
            dst_user=_text_attr(item.attrs, "dst_user"),
            src_computer=_text_attr(item.attrs, "src_computer"),
            dst_computer=_text_attr(item.attrs, "dst_computer"),
        )
    return NormalizedEvent(
        event_id=str(item["event_id"]),
        time=int(item["time"]),
        label=_is_positive_label(item.get("label")),
        src_user=str(item.get("src_user", "")),
        dst_user=str(item.get("dst_user", "")),
        src_computer=str(item.get("src_computer", "")),
        dst_computer=str(item.get("dst_computer", "")),
    )


def _is_positive_label(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "red"}


def _text_attr(attrs: dict[str, Any], key: str) -> str:
    value = attrs.get(key)
    return "" if value in (None, "") else str(value)


def _within_window(prior_time: int, query_time: int, window: int) -> bool:
    return prior_time < query_time and (query_time - prior_time) <= max(0, window)

