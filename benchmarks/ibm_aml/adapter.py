"""IBM AML benchmark adapter utilities."""

from __future__ import annotations

from csv import DictReader
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from timeindex.event import Event, EventRecord


_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "transaction_id": ("transaction_id", "tx_id", "transactionid", "id"),
    "timestamp": ("timestamp", "time", "datetime", "date", "step"),
    "payment_format": ("payment_format", "payment system", "payment_system", "format"),
    "src_account": ("src_account", "source_account", "account", "nameorig", "account_id"),
    "dst_account": ("dst_account", "target_account", "beneficiary_account", "nameDest", "namedest"),
    "amount": ("amount", "amount_paid", "payment_amount", "amountusd", "amount_received"),
    "currency": ("currency", "ccy", "cur"),
    "src_bank": ("src_bank", "source_bank", "bankorig", "bank"),
    "dst_bank": ("dst_bank", "target_bank", "bankdest"),
    "label": ("label", "is_laundering", "laundering", "is_sar", "sar"),
    "type": ("type", "transaction_type", "payment_type"),
}


def load_ibm_aml_csv(path: str | Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    """Load IBM AML CSV rows as dictionaries with a stable row index."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        for row_index, row in enumerate(reader):
            enriched = dict(row)
            enriched["__row_index__"] = row_index
            rows.append(enriched)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def convert_row_to_event(row: dict[str, Any], schema: Any) -> EventRecord:
    """Convert one IBM AML CSV row into an EventRecord."""

    row_id = _first_value(row, schema, "transaction_id")
    event_id = str(row_id) if row_id not in (None, "") else f"row-{row.get('__row_index__', 0)}"

    raw_time = _first_value(row, schema, "timestamp")
    event_time = _coerce_time(raw_time, row.get("__row_index__"))

    payment_format = _clean_text(_first_value(row, schema, "payment_format"))
    event_type = _clean_text(_first_value(row, schema, "type")) or payment_format or "transaction"

    amount = _coerce_number(_first_value(row, schema, "amount"))
    attrs = {
        "src_account": _clean_text(_first_value(row, schema, "src_account")),
        "dst_account": _clean_text(_first_value(row, schema, "dst_account")),
        "amount": amount,
        "currency": _clean_text(_first_value(row, schema, "currency")),
        "src_bank": _clean_text(_first_value(row, schema, "src_bank")),
        "dst_bank": _clean_text(_first_value(row, schema, "dst_bank")),
        "payment_format": payment_format,
    }
    label = _normalize_label(_first_value(row, schema, "label"))
    source_path = _extract_source_file(schema)
    dataset_name = _extract_dataset_name(schema, source_path)

    event = Event(
        event_id=event_id,
        time=event_time,
        event_type=event_type,
        attrs=attrs,
        ctx={"dataset": dataset_name, "source_file": source_path},
        label=label,
        text=_build_text(event_id, event_type, attrs, label),
    )
    return EventRecord(event=event)


def stream_events(
    path: str | Path,
    schema: Any,
    sort_by_time: bool = True,
    max_rows: int | None = None,
) -> Iterable[EventRecord]:
    """Yield IBM AML rows as EventRecord objects."""

    rows = load_ibm_aml_csv(path, max_rows=max_rows)
    if not sort_by_time:
        for row in rows:
            yield convert_row_to_event(row, schema)
        return

    records = [convert_row_to_event(row, schema) for row in rows]
    records.sort(key=lambda record: _sort_key(record.event.time, record.event.event_id))
    for record in records:
        yield record


def _sort_key(value: Any, event_id: str) -> tuple[int, Any, str]:
    if isinstance(value, (int, float)):
        return (0, float(value), event_id)
    return (1, str(value), event_id)


def _extract_source_file(schema: Any) -> str:
    source = _schema_value(schema, "source_file") or _schema_value(schema, "path")
    return str(source) if source else ""


def _extract_dataset_name(schema: Any, source_file: str) -> str:
    dataset = _schema_value(schema, "dataset") or _schema_value(schema, "dataset_name") or "ibm_aml"
    if dataset == "ibm_aml" and source_file:
        dataset = Path(source_file).stem or dataset
    return str(dataset)


def _schema_value(schema: Any, key: str) -> Any:
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema.get(key)
    return getattr(schema, key, None)


def _first_value(row: dict[str, Any], schema: Any, logical_name: str) -> Any:
    for key in _schema_candidates(schema, logical_name):
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _schema_candidates(schema: Any, logical_name: str) -> list[str]:
    candidates = list(_COLUMN_ALIASES.get(logical_name, (logical_name,)))
    mapped = _schema_value(schema, logical_name)
    if isinstance(mapped, str) and mapped:
        candidates.insert(0, mapped)
    columns = _schema_value(schema, "columns")
    if isinstance(columns, dict):
        mapped = columns.get(logical_name)
        if isinstance(mapped, str) and mapped:
            candidates.insert(0, mapped)
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def _coerce_time(value: Any, fallback_index: Any) -> float | int:
    parsed = _parse_datetime(value)
    if parsed is not None:
        return parsed.timestamp()
    number = _coerce_number(value)
    if number is not None:
        return number
    if isinstance(fallback_index, int):
        return fallback_index
    return 0


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    for candidate in (text, text.replace("Z", "+00:00"), text.replace("/", "-")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _coerce_number(value: Any) -> int | float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return number


def _normalize_label(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _clean_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip()


def _build_text(event_id: str, event_type: str, attrs: dict[str, Any], label: str | None) -> str:
    amount = attrs.get("amount")
    currency = attrs.get("currency") or "units"
    src = attrs.get("src_account") or "unknown source"
    dst = attrs.get("dst_account") or "unknown destination"
    bank_bits = [item for item in (attrs.get("src_bank"), attrs.get("dst_bank")) if item]
    format_bit = attrs.get("payment_format")

    parts = [f"{event_type} {event_id}", f"from {src} to {dst}"]
    if amount is not None:
        parts.append(f"for {amount} {currency}")
    if format_bit:
        parts.append(f"via {format_bit}")
    if bank_bits:
        parts.append(f"banks {' -> '.join(bank_bits)}")
    if label:
        parts.append(f"label {label}")
    return ", ".join(parts)
