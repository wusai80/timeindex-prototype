"""Profile IBM AML transaction CSV files for the TimeIndex benchmark."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

try:
    from .schema import (
        CANONICAL_FIELDS,
        DESTINATION_ACCOUNT_FIELD,
        LABEL_FIELD,
        PATTERN_FIELD,
        SOURCE_ACCOUNT_FIELD,
        TIMESTAMP_FIELD,
        normalize_column_name,
    )
except ImportError:
    from schema import (  # type: ignore
        CANONICAL_FIELDS,
        DESTINATION_ACCOUNT_FIELD,
        LABEL_FIELD,
        PATTERN_FIELD,
        SOURCE_ACCOUNT_FIELD,
        TIMESTAMP_FIELD,
        normalize_column_name,
    )


@dataclass(slots=True)
class DatasetTable:
    """Lightweight in-memory CSV table."""

    columns: list[str]
    rows: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        nargs="+",
        required=True,
        help="One or more CSV files or directories containing CSV files.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=1000,
        help="Rows to inspect when inferring datetime-like columns.",
    )
    return parser.parse_args()


def collect_csv_paths(raw_paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(candidate for candidate in path.rglob("*.csv") if candidate.is_file()))
        elif path.is_file() and path.suffix.lower() == ".csv":
            files.append(path)
        else:
            raise FileNotFoundError(f"CSV path not found: {raw_path}")
    if not files:
        raise FileNotFoundError("No CSV files found under the provided path(s).")
    return sorted(set(files))


def load_dataset(paths: list[Path]) -> DatasetTable:
    all_columns: list[str] = []
    rows: list[dict[str, str]] = []
    seen_columns: set[str] = set()

    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            for column in fieldnames:
                if column not in seen_columns:
                    seen_columns.add(column)
                    all_columns.append(column)
            for raw_row in reader:
                row = {column: (raw_row.get(column) or "") for column in fieldnames}
                row["__source_file__"] = str(path)
                rows.append(row)

    for row in rows:
        for column in all_columns:
            row.setdefault(column, "")

    return DatasetTable(columns=all_columns, rows=rows)


def build_column_lookup(columns: Iterable[str]) -> dict[str, str]:
    return {normalize_column_name(column): column for column in columns}


def resolve_alias(field_aliases: Iterable[str], lookup: dict[str, str]) -> str | None:
    for alias in field_aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def detect_time_column(table: DatasetTable, sample_rows: int) -> str | None:
    lookup = build_column_lookup(table.columns)
    direct_match = resolve_alias((normalize_column_name(alias) for alias in TIMESTAMP_FIELD.aliases), lookup)
    if direct_match:
        return direct_match

    best_column: str | None = None
    best_score = -1.0
    for column in table.columns:
        values = [row[column] for row in table.rows[:sample_rows] if row.get(column, "").strip()]
        if not values:
            continue

        normalized = normalize_column_name(column)
        lexical_bonus = 0.25 if any(token in normalized for token in ("time", "date", "step")) else 0.0
        datetime_score = _parse_ratio(values, _parse_datetime)
        numeric_score = _parse_ratio(values, _parse_float)

        if datetime_score > 0.8 and datetime_score + lexical_bonus > best_score:
            best_column = column
            best_score = datetime_score + lexical_bonus
            continue
        if numeric_score > 0.95 and lexical_bonus > 0.0 and numeric_score + lexical_bonus > best_score:
            best_column = column
            best_score = numeric_score + lexical_bonus

    return best_column


def detect_account_columns(table: DatasetTable) -> tuple[str | None, str | None]:
    lookup = build_column_lookup(table.columns)
    source = resolve_alias((normalize_column_name(alias) for alias in SOURCE_ACCOUNT_FIELD.aliases), lookup)
    destination = resolve_alias((normalize_column_name(alias) for alias in DESTINATION_ACCOUNT_FIELD.aliases), lookup)

    if source and destination:
        return source, destination

    normalized_pairs = [(normalize_column_name(column), column) for column in table.columns]
    if source is None:
        for normalized, original in normalized_pairs:
            if any(token in normalized for token in ("from", "source", "sender", "origin")) and "account" in normalized:
                source = original
                break
    if destination is None:
        for normalized, original in normalized_pairs:
            if any(token in normalized for token in ("to", "dest", "receiver", "beneficiary", "recipient")) and "account" in normalized:
                destination = original
                break

    account_like_columns = [
        original
        for normalized, original in normalized_pairs
        if "account" in normalized and original not in {source, destination}
    ]
    if source is None and account_like_columns:
        source = account_like_columns[0]
    if destination is None and len(account_like_columns) > 1:
        destination = account_like_columns[1]

    return source, destination


def detect_label_column(table: DatasetTable) -> str | None:
    lookup = build_column_lookup(table.columns)
    label = resolve_alias((normalize_column_name(alias) for alias in LABEL_FIELD.aliases), lookup)
    if label:
        return label

    for column in table.columns:
        normalized = normalize_column_name(column)
        if any(token in normalized for token in ("launder", "label", "target", "suspicious", "fraud")):
            return column
    return None


def detect_pattern_column(table: DatasetTable) -> str | None:
    lookup = build_column_lookup(table.columns)
    return resolve_alias((normalize_column_name(alias) for alias in PATTERN_FIELD.aliases), lookup)


def format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.4%}"


def summarize_missing(table: DatasetTable) -> list[tuple[str, int, float]]:
    total_rows = len(table.rows)
    summaries: list[tuple[str, int, float]] = []
    for column in table.columns:
        missing = sum(1 for row in table.rows if not row.get(column, "").strip())
        ratio = (missing / total_rows) if total_rows else math.nan
        summaries.append((column, missing, ratio))
    return sorted(summaries, key=lambda item: (-item[1], item[0]))


def summarize_label_distribution(table: DatasetTable, label_column: str) -> list[str]:
    values = [row[label_column].strip() for row in table.rows if row.get(label_column, "").strip()]
    if not values:
        return [f"Label column `{label_column}` has no non-null values."]

    counts = Counter(values)
    lines = [f"Class imbalance (`{label_column}`):"]
    for value, count in counts.most_common():
        lines.append(f"  {value}: {count} ({format_ratio(count, len(values))})")
    if len(counts) == 2:
        minority = min(counts.values())
        majority = max(counts.values())
        ratio = (majority / minority) if minority else math.inf
        lines.append(f"  imbalance ratio (majority/minority): {ratio:.2f}")
    return lines


def summarize_accounts(table: DatasetTable, source_column: str | None, destination_column: str | None) -> list[str]:
    if source_column is None and destination_column is None:
        return ["Accounts: unable to detect source/destination account columns."]

    counts: Counter[str] = Counter()
    if source_column is not None:
        counts.update(row[source_column].strip() for row in table.rows if row.get(source_column, "").strip())
    if destination_column is not None:
        counts.update(row[destination_column].strip() for row in table.rows if row.get(destination_column, "").strip())

    if not counts:
        return ["Accounts: no account values available after dropping nulls."]

    values = sorted(counts.values())
    lines = [f"Unique accounts/entities: {len(counts)}"]
    lines.append(
        "Transactions per account:"
        f" min={values[0]}, median={median(values):.1f}, mean={mean(values):.2f}, max={values[-1]}"
    )
    top_accounts = ", ".join(f"{account}={count}" for account, count in counts.most_common(5))
    lines.append(f"Top account activity: {top_accounts}")
    return lines


def summarize_time_range(table: DatasetTable, time_column: str | None) -> list[str]:
    if time_column is None:
        return ["Time range: unable to detect a time column."]

    values = [row[time_column].strip() for row in table.rows if row.get(time_column, "").strip()]
    if not values:
        return [f"Time range: `{time_column}` is entirely null."]

    parsed_datetimes = [item for item in (_parse_datetime(value) for value in values) if item is not None]
    if parsed_datetimes:
        return [
            f"Time column: {time_column}",
            f"Time range: {min(parsed_datetimes).isoformat()} -> {max(parsed_datetimes).isoformat()}",
        ]

    parsed_numbers = [item for item in (_parse_float(value) for value in values) if item is not None]
    if parsed_numbers:
        return [
            f"Time column: {time_column}",
            f"Time range: {min(parsed_numbers)} -> {max(parsed_numbers)}",
        ]

    return [f"Time range: unable to parse values from `{time_column}`."]


def print_schema_hints() -> None:
    print("Canonical schema hints:")
    for field in CANONICAL_FIELDS:
        aliases = ", ".join(field.aliases[:5])
        suffix = " ..." if len(field.aliases) > 5 else ""
        required = "required" if field.required else "optional"
        print(f"  {field.name} ({required}): {aliases}{suffix}")


def main() -> None:
    args = parse_args()
    csv_paths = collect_csv_paths(args.path)
    table = load_dataset(csv_paths)

    time_column = detect_time_column(table, sample_rows=args.sample_rows)
    source_column, destination_column = detect_account_columns(table)
    label_column = detect_label_column(table)
    pattern_column = detect_pattern_column(table)

    print("IBM AML Dataset Profile")
    print("=======================")
    print(f"Files loaded: {len(csv_paths)}")
    for path in csv_paths:
        print(f"  - {path}")
    print(f"Rows: {len(table.rows)}")
    print(f"Columns ({len(table.columns)}): {', '.join(table.columns)}")
    print()

    print("Detected columns:")
    print(f"  time: {time_column or 'not found'}")
    print(f"  source account: {source_column or 'not found'}")
    print(f"  destination account: {destination_column or 'not found'}")
    print(f"  laundering label: {label_column or 'not found'}")
    print(f"  laundering pattern: {pattern_column or 'not found'}")
    print()

    print("Missing values:")
    for column, missing, ratio in summarize_missing(table)[:20]:
        print(f"  {column}: {missing} missing ({ratio:.2%})")
    print()

    if label_column:
        for line in summarize_label_distribution(table, label_column):
            print(line)
        print()

    for line in summarize_accounts(table, source_column, destination_column):
        print(line)
    print()

    for line in summarize_time_range(table, time_column):
        print(line)
    print()

    print_schema_hints()


def _parse_ratio(values: list[str], parser: Any) -> float:
    if not values:
        return 0.0
    successes = sum(1 for value in values if parser(value) is not None)
    return successes / len(values)


def _parse_datetime(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    candidates = (text, text.replace("Z", "+00:00"), text.replace("/", "-"))
    for candidate in candidates:
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


def _parse_float(value: str) -> float | None:
    text = value.strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
