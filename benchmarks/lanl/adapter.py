"""Adapter utilities for the LANL authentication benchmark slice."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from benchmarks.lanl.schema import LanlAuthSchema, default_auth_schema, detect_auth_schema
from timeindex.event import Event, EventRecord


@dataclass(frozen=True, slots=True)
class RedTeamActivity:
    """Normalized red-team activity row."""

    time: int
    src_user: str
    src_computer: str
    dst_computer: str


def load_lanl_redteam(path: str | Path) -> set[RedTeamActivity]:
    """Load normalized red-team activities from a LANL redteam file."""

    activities: set[RedTeamActivity] = set()
    with _open_text(path) as handle:
        for line in handle:
            row = line.strip()
            if not row:
                continue
            parts = [piece.strip() for piece in row.split(",")]
            if len(parts) < 4 or parts[0].lower() == "time":
                continue
            time_value = _coerce_int(parts[0], default=0)
            src_user, _src_domain = _split_principal(parts[1])
            activities.add(
                RedTeamActivity(
                    time=time_value,
                    src_user=src_user,
                    src_computer=_normalize_token(parts[2]),
                    dst_computer=_normalize_token(parts[3]),
                )
            )
    return activities


def load_lanl_auth(
    path: str | Path,
    *,
    schema: LanlAuthSchema | None = None,
    max_rows: int | None = None,
    row_index_offset: int = 0,
) -> list[dict[str, Any]]:
    """Load LANL authentication rows into dictionaries with stable row indices."""

    path_text = str(path)
    schema = schema or default_auth_schema(source_file=path_text)
    return list(_iter_auth_rows(path, schema=schema, max_rows=max_rows, row_index_offset=row_index_offset))


def convert_auth_row_to_event(
    row: dict[str, Any],
    redteam_index: set[RedTeamActivity],
    schema: LanlAuthSchema | None = None,
) -> EventRecord:
    """Convert one normalized auth row into an EventRecord."""

    schema = schema or default_auth_schema()
    event_id = str(row.get("__event_id__", f"auth-{row.get('__row_index__', 0)}"))
    src_user_raw = _clean_text(row.get(schema.src_user))
    dst_user_raw = _clean_text(row.get(schema.dst_user))
    src_user, src_domain = _split_principal(src_user_raw)
    dst_user, dst_domain = _split_principal(dst_user_raw)
    src_computer = _normalize_token(row.get(schema.src_computer))
    dst_computer = _normalize_token(row.get(schema.dst_computer))
    auth_type = _clean_text(row.get(schema.auth_type))
    logon_type = _clean_text(row.get(schema.logon_type))
    orientation = _clean_text(row.get(schema.auth_orientation))
    success = _coerce_success(row.get(schema.success))
    event_time = _coerce_int(row.get(schema.time), default=int(row.get("__row_index__", 0)))

    label = "1" if RedTeamActivity(event_time, src_user, src_computer, dst_computer) in redteam_index else "0"
    attrs = {
        "src_user": src_user,
        "dst_user": dst_user,
        "src_computer": src_computer,
        "dst_computer": dst_computer,
        "src_domain": src_domain,
        "dst_domain": dst_domain,
        "auth_type": auth_type,
        "logon_type": logon_type,
        "auth_orientation": orientation,
        "success": success,
        "is_cross_host_auth": bool(row.get("__is_cross_host_auth__", False)),
        "is_new_dst_for_user": bool(row.get("__is_new_dst_for_user__", False)),
        "prior_user_event_count": int(row.get("__prior_user_event_count__", 0)),
        "prior_user_host_count": int(row.get("__prior_user_host_count__", 0)),
        "prior_pair_seen": bool(row.get("__prior_pair_seen__", False)),
        "is_machine_account": bool(row.get("__is_machine_account__", False)),
        "is_anonymous_logon": bool(row.get("__is_anonymous_logon__", False)),
    }
    event = Event(
        event_id=event_id,
        time=event_time,
        event_type="authentication",
        attrs=attrs,
        ctx={
            "dataset": schema.dataset_name,
            "source_file": schema.source_file,
            "redteam_file": schema.redteam_file,
        },
        text=_build_text(src_user, dst_user, src_computer, dst_computer, auth_type, success),
        label=label,
    )
    return EventRecord(event=event)


def stream_events(
    auth_path: str | Path,
    redteam_path: str | Path,
    schema: LanlAuthSchema | None = None,
    *,
    sort_by_time: bool = True,
    max_rows: int | None = None,
    row_index_offset: int = 0,
) -> Iterable[EventRecord]:
    """Yield LANL authentication events as EventRecord objects."""

    schema = schema or default_auth_schema(source_file=str(auth_path), redteam_file=str(redteam_path))
    redteam_index = load_lanl_redteam(redteam_path)
    if sort_by_time:
        rows = load_lanl_auth(auth_path, schema=schema, max_rows=max_rows, row_index_offset=row_index_offset)
        rows.sort(key=lambda row: (_coerce_int(row.get(schema.time), default=0), int(row.get("__row_index__", 0))))
        _enrich_history_features(rows, schema)
        for row in rows:
            yield convert_auth_row_to_event(row, redteam_index, schema)
        return

    history_state = _new_history_state()
    for row in _iter_auth_rows(auth_path, schema=schema, max_rows=max_rows, row_index_offset=row_index_offset):
        _annotate_history_features(row, schema, history_state)
        yield convert_auth_row_to_event(row, redteam_index, schema)


def _enrich_history_features(rows: list[dict[str, Any]], schema: LanlAuthSchema) -> None:
    history_state = _new_history_state()
    for row in rows:
        _annotate_history_features(row, schema, history_state)


def _iter_auth_rows(
    path: str | Path,
    *,
    schema: LanlAuthSchema,
    max_rows: int | None = None,
    row_index_offset: int = 0,
) -> Iterator[dict[str, Any]]:
    path_text = str(path)
    schema = schema or default_auth_schema(source_file=path_text)
    yielded = 0
    with _open_text(path) as handle:
        first_line = handle.readline()
        if not first_line:
            return
        first_parts = [piece.strip() for piece in first_line.strip().split(",")]
        has_header = _looks_like_header(first_parts)
        if has_header:
            schema = detect_auth_schema(
                first_parts,
                source_file=path_text,
                redteam_file=schema.redteam_file,
                dataset_name=schema.dataset_name,
            )
            iterator: Iterator[str] = iter(handle)
        else:
            iterator = chain((first_line,), handle)
        for row_index, raw_line in enumerate(iterator):
            line = raw_line.strip()
            if not line:
                continue
            global_row_index = int(row_index_offset) + row_index
            yield _parse_auth_line(
                line,
                row_index=global_row_index,
                schema=schema,
                has_header=has_header,
            )
            yielded += 1
            if max_rows is not None and yielded >= max_rows:
                break


def _new_history_state() -> dict[str, Any]:
    return {
        "user_seen_destinations": {},
        "user_event_counts": {},
        "user_pair_history": set(),
    }


def _annotate_history_features(
    row: dict[str, Any],
    schema: LanlAuthSchema,
    history_state: dict[str, Any],
) -> None:
    user_seen_destinations: dict[str, set[str]] = history_state["user_seen_destinations"]
    user_event_counts: dict[str, int] = history_state["user_event_counts"]
    user_pair_history: set[tuple[str, str]] = history_state["user_pair_history"]
    src_user, _src_domain = _split_principal(row.get(schema.src_user))
    _dst_user, _dst_domain = _split_principal(row.get(schema.dst_user))
    src_computer = _normalize_token(row.get(schema.src_computer))
    dst_computer = _normalize_token(row.get(schema.dst_computer))
    seen_destinations = user_seen_destinations.setdefault(src_user, set())
    pair = (src_user, dst_computer)
    src_user_raw = _normalize_token(row.get(schema.src_user))
    dst_user_raw = _normalize_token(row.get(schema.dst_user))

    row["__is_cross_host_auth__"] = bool(src_computer and dst_computer and src_computer != dst_computer)
    row["__is_new_dst_for_user__"] = bool(dst_computer and dst_computer not in seen_destinations)
    row["__prior_user_event_count__"] = int(user_event_counts.get(src_user, 0))
    row["__prior_user_host_count__"] = int(len(seen_destinations))
    row["__prior_pair_seen__"] = pair in user_pair_history
    row["__is_machine_account__"] = src_user_raw.endswith("$") or dst_user_raw.endswith("$")
    row["__is_anonymous_logon__"] = "anonymous logon" in src_user_raw or "anonymous logon" in dst_user_raw

    user_event_counts[src_user] = user_event_counts.get(src_user, 0) + 1
    if dst_computer:
        seen_destinations.add(dst_computer)
        user_pair_history.add(pair)


def _parse_auth_line(
    line: str,
    *,
    row_index: int,
    schema: LanlAuthSchema,
    has_header: bool,
) -> dict[str, Any]:
    parts = [piece.strip() for piece in line.split(",")]
    if has_header:
        values = dict(zip(schema.columns.values(), parts, strict=False))
    else:
        padded = parts + [""] * max(0, 9 - len(parts))
        values = {
            schema.time: padded[0],
            schema.src_user: padded[1],
            schema.dst_user: padded[2],
            schema.src_computer: padded[3],
            schema.dst_computer: padded[4],
            schema.auth_type: padded[5],
            schema.logon_type: padded[6],
            schema.auth_orientation: padded[7],
            schema.success: padded[8],
        }
    values["__row_index__"] = row_index
    values["__event_id__"] = f"auth-{row_index:08d}"
    return values


def _build_text(
    src_user: str,
    dst_user: str,
    src_computer: str,
    dst_computer: str,
    auth_type: str | None,
    success: bool,
) -> str:
    auth_label = auth_type or "auth"
    return (
        f"user {src_user or 'unknown'} authenticated as {dst_user or 'unknown'} "
        f"from {src_computer or 'unknown'} to {dst_computer or 'unknown'} "
        f"via {auth_label} success={str(success).lower()}"
    )


def _open_text(path: str | Path) -> TextIO:
    path_obj = Path(path)
    if path_obj.suffix == ".gz":
        return gzip.open(path_obj, "rt", encoding="utf-8")
    return path_obj.open("r", encoding="utf-8")


def _looks_like_header(parts: list[str]) -> bool:
    if not parts:
        return False
    return parts[0].strip().lower() in {"time", "timestamp"} and any(not _is_int_like(part) for part in parts[1:])


def _is_int_like(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _split_principal(value: Any) -> tuple[str, str | None]:
    text = _clean_text(value)
    if not text:
        return "", None
    if "@" not in text:
        return _normalize_token(text), None
    user, domain = text.split("@", 1)
    return _normalize_token(user), _normalize_token(domain)


def _normalize_token(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    return text.lower()


def _clean_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip()


def _coerce_success(value: Any) -> bool:
    text = _normalize_token(value)
    return text in {"success", "s", "true", "1", "yes"}
