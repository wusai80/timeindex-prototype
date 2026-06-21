"""Gold evidence labeling utilities for the IBM AML benchmark."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any


SAME_ENTITY_LAUNDERING_WINDOW = "same-entity-laundering-window"
FLOW_CHAIN = "flow-chain"
TYPOLOGY_GROUP_AWARE = "typology/group-aware"
WEAK_SOURCE_ACCUMULATION = "weak-source-accumulation"

_POLICIES = {
    SAME_ENTITY_LAUNDERING_WINDOW,
    FLOW_CHAIN,
    TYPOLOGY_GROUP_AWARE,
    WEAK_SOURCE_ACCUMULATION,
}

_EVENT_ID_KEYS = ("event_id", "tx_id", "transaction_id", "id")
_TIMESTAMP_KEYS = ("timestamp", "ts", "time", "date", "step")
_SRC_KEYS = ("src_account", "source_account", "account_from", "from_account", "orig_acct")
_DST_KEYS = ("dst_account", "target_account", "account_to", "to_account", "bene_acct")
_AMOUNT_KEYS = ("amount", "transaction_amount", "amt")
_LAUNDERING_KEYS = (
    "is_laundering",
    "laundering",
    "label",
    "suspicious",
    "is_suspicious",
)
_GROUP_KEYS = (
    "pattern_id",
    "alert_id",
    "group_id",
    "typology_id",
    "laundering_pattern_id",
)


def build_gold_evidence(
    events: Sequence[Mapping[str, Any]],
    policy: str,
    window: Any,
    max_hops: int = 2,
    amount_threshold: float = 0.8,
) -> dict[str, set[str]]:
    """Return gold support event ids for each laundering query transaction."""

    if policy not in _POLICIES:
        raise ValueError(f"Unsupported evidence policy: {policy}")

    normalized = [_normalize_event(event, index) for index, event in enumerate(events)]
    normalized.sort(key=lambda event: (event["sort_time"], event["index"]))

    evidence: dict[str, set[str]] = {}
    history: list[dict[str, Any]] = []

    for event in normalized:
        if not event["is_laundering"]:
            history.append(event)
            continue

        if policy == SAME_ENTITY_LAUNDERING_WINDOW:
            support = _same_entity_laundering_window(event, history, window)
        elif policy == FLOW_CHAIN:
            support = _flow_chain(event, history, window, max_hops)
        elif policy == TYPOLOGY_GROUP_AWARE:
            support = _typology_group_aware(event, history, window)
        else:
            support = _weak_source_accumulation(event, history, window, amount_threshold)

        evidence[event["event_id"]] = support
        history.append(event)

    return evidence


def _same_entity_laundering_window(
    query: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    window: Any,
) -> set[str]:
    src = query["src_account"]
    dst = query["dst_account"]
    return {
        event["event_id"]
        for event in _prior_window(history, query, window)
        if event["is_laundering"] and (event["src_account"] in {src, dst} or event["dst_account"] in {src, dst})
    }


def _flow_chain(
    query: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    window: Any,
    max_hops: int,
) -> set[str]:
    prior = [event for event in _prior_window(history, query, window) if event["is_laundering"]]
    if max_hops < 1 or not prior:
        return set()

    directed_reverse: dict[str, set[str]] = defaultdict(set)
    undirected: dict[str, set[str]] = defaultdict(set)
    for event in prior:
        src = event["src_account"]
        dst = event["dst_account"]
        directed_reverse[dst].add(src)
        undirected[src].add(dst)
        undirected[dst].add(src)

    reachable_to_src = _bfs_accounts({query["src_account"]}, directed_reverse, max_hops)
    connected_to_src = _bfs_accounts({query["src_account"]}, undirected, max_hops)

    support: set[str] = set()
    for event in prior:
        if event["dst_account"] in reachable_to_src:
            support.add(event["event_id"])
            continue
        if event["src_account"] in connected_to_src or event["dst_account"] in connected_to_src:
            support.add(event["event_id"])
    return support


def _typology_group_aware(
    query: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    window: Any,
) -> set[str]:
    group_value = query.get("group_value")
    if group_value is None:
        return set()

    return {
        event["event_id"]
        for event in _prior_window(history, query, window)
        if event.get("group_value") == group_value
    }


def _weak_source_accumulation(
    query: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    window: Any,
    amount_threshold: float,
) -> set[str]:
    incoming = [
        event
        for event in _prior_window(history, query, window)
        if event["dst_account"] == query["src_account"] and event["amount"] > 0.0
    ]
    incoming.sort(key=lambda event: (event["sort_time"], event["index"]), reverse=True)

    target_amount = max(float(amount_threshold), 0.0) * query["amount"]
    running_amount = 0.0
    support: set[str] = set()
    for event in incoming:
        support.add(event["event_id"])
        running_amount += event["amount"]
        if running_amount >= target_amount:
            break

    if running_amount < target_amount:
        return set()
    return support


def _prior_window(
    history: Sequence[Mapping[str, Any]],
    query: Mapping[str, Any],
    window: Any,
) -> Iterable[Mapping[str, Any]]:
    for event in history:
        if event["sort_time"] >= query["sort_time"]:
            continue
        if _within_window(event["time_value"], query["time_value"], window):
            yield event


def _bfs_accounts(
    start_accounts: set[str],
    adjacency: Mapping[str, set[str]],
    max_hops: int,
) -> set[str]:
    visited = set(start_accounts)
    queue: deque[tuple[str, int]] = deque((account, 0) for account in start_accounts)

    while queue:
        account, hops = queue.popleft()
        if hops >= max_hops:
            continue
        for neighbor in adjacency.get(account, set()):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, hops + 1))

    return visited


def _normalize_event(event: Mapping[str, Any], index: int) -> dict[str, Any]:
    event_id = str(_first_present(event, _EVENT_ID_KEYS))
    time_value = _first_present(event, _TIMESTAMP_KEYS)
    src_account = str(_first_present(event, _SRC_KEYS))
    dst_account = str(_first_present(event, _DST_KEYS))
    amount = float(_first_present(event, _AMOUNT_KEYS, default=0.0))
    group_value = _first_present(event, _GROUP_KEYS, default=None)

    return {
        "index": index,
        "event_id": event_id,
        "time_value": time_value,
        "sort_time": _sort_key(time_value),
        "src_account": src_account,
        "dst_account": dst_account,
        "amount": amount,
        "is_laundering": _coerce_laundering_flag(_first_present(event, _LAUNDERING_KEYS, default=False)),
        "group_value": None if group_value is None else str(group_value),
    }


def _first_present(event: Mapping[str, Any], keys: Sequence[str], default: Any = ... ) -> Any:
    for key in keys:
        if key in event:
            return event[key]
    if default is ...:
        raise KeyError(f"Expected one of {keys}, found keys {sorted(event)}")
    return default


def _coerce_laundering_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "y", "laundering", "suspicious"}
    return False


def _within_window(previous: Any, current: Any, window: Any) -> bool:
    if window is None:
        return True

    try:
        return (current - previous) <= window
    except TypeError:
        previous_dt = _to_datetime(previous)
        current_dt = _to_datetime(current)
        return (current_dt - previous_dt) <= window


def _sort_key(value: Any) -> Any:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    return value


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise TypeError(f"Unsupported timestamp type for window comparison: {type(value)!r}")
