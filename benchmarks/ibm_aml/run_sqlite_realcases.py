"""Evaluate an existing SQLite TimeIndex on real IBM AML laundering cases."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmarks.metrics import (
    context_efficiency,
    evidence_f1_at_budget,
    evidence_precision_at_budget,
    evidence_recall_at_budget,
    mean_latency_ms,
)
from timeindex import DecisionIntent
from timeindex.retrieval import retrieve
from timeindex.sqlite_backend import SqliteTimeIndexBackend


DEFAULT_BUDGETS: tuple[int, ...] = (3, 5, 10, 20)


@dataclass(slots=True)
class QueryGold:
    """Gold evidence for one laundering query under multiple policies."""

    query_event_id: str
    query_time: datetime
    query_label: str
    same_entity_ids: list[str]
    weak_accumulation_ids: list[str]

    @property
    def union_ids(self) -> list[str]:
        merged: list[str] = []
        for event_id in self.same_entity_ids + self.weak_accumulation_ids:
            if event_id not in merged:
                merged.append(event_id)
        return merged


def run_sqlite_realcase_evaluation(
    csv_path: str | Path,
    index_path: str | Path,
    *,
    output_dir: str | Path = "outputs/ibm_aml/realcase_eval",
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    same_entity_window_hours: int = 24,
    accumulation_window_hours: int = 24,
    accumulation_threshold: float = 0.8,
    limit_queries: int | None = None,
) -> dict[str, Any]:
    """Run a retrieval-only evaluation on laundering cases using an existing SQLite index."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    query_gold = _stream_realcase_gold(
        Path(csv_path),
        same_entity_window=timedelta(hours=max(1, same_entity_window_hours)),
        accumulation_window=timedelta(hours=max(1, accumulation_window_hours)),
        accumulation_threshold=float(accumulation_threshold),
        limit_queries=limit_queries,
    )

    index = SqliteTimeIndexBackend.open(index_path)
    results: list[dict[str, Any]] = []

    try:
        for query in query_gold:
            query_record = index.get_event(query.query_event_id)
            if query_record is None:
                continue
            intent = _build_query_intent(query_record.event)
            for budget in budgets:
                started = perf_counter()
                evidence_objects = retrieve(index, query.query_event_id, intent, budget)
                latency_ms = (perf_counter() - started) * 1000.0
                retrieved_ids, object_types, aspects = _normalize_retrieval(evidence_objects, query.query_event_id)
                results.append(
                    {
                        "query_event_id": query.query_event_id,
                        "query_label": query.query_label,
                        "query_time": query.query_time.isoformat(),
                        "budget": int(budget),
                        "retrieved_event_ids": retrieved_ids[:budget],
                        "retrieved_object_types": object_types,
                        "retrieved_aspects": aspects,
                        "gold_same_entity_ids": list(query.same_entity_ids),
                        "gold_weak_accumulation_ids": list(query.weak_accumulation_ids),
                        "gold_union_ids": query.union_ids,
                        "latency_ms": latency_ms,
                    }
                )
    finally:
        index.close()

    results_path = output_path / "retrieval_results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    aggregate_rows = _aggregate_results(results, budgets)
    aggregate_path = output_path / "aggregate.csv"
    _write_csv(aggregate_path, aggregate_rows)

    summary = {
        "csv_path": str(csv_path),
        "index_path": str(index_path),
        "output_dir": str(output_path),
        "query_count": len(query_gold),
        "budget_count": len(tuple(budgets)),
        "same_entity_window_hours": same_entity_window_hours,
        "accumulation_window_hours": accumulation_window_hours,
        "accumulation_threshold": float(accumulation_threshold),
        "positive_queries_with_same_entity_gold": sum(1 for item in query_gold if item.same_entity_ids),
        "positive_queries_with_weak_accumulation_gold": sum(1 for item in query_gold if item.weak_accumulation_ids),
        "positive_queries_with_union_gold": sum(1 for item in query_gold if item.union_ids),
        "results_path": str(results_path),
        "aggregate_path": str(aggregate_path),
    }
    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("index_path")
    parser.add_argument("--output-dir", default="outputs/ibm_aml/realcase_eval")
    parser.add_argument("--same-entity-window-hours", type=int, default=24)
    parser.add_argument("--accumulation-window-hours", type=int, default=24)
    parser.add_argument("--accumulation-threshold", type=float, default=0.8)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--budgets", nargs="*", type=int, default=list(DEFAULT_BUDGETS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_sqlite_realcase_evaluation(
        args.csv_path,
        args.index_path,
        output_dir=args.output_dir,
        budgets=tuple(args.budgets),
        same_entity_window_hours=args.same_entity_window_hours,
        accumulation_window_hours=args.accumulation_window_hours,
        accumulation_threshold=args.accumulation_threshold,
        limit_queries=args.limit_queries,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _stream_realcase_gold(
    csv_path: Path,
    *,
    same_entity_window: timedelta,
    accumulation_window: timedelta,
    accumulation_threshold: float,
    limit_queries: int | None,
) -> list[QueryGold]:
    laundering_history_by_account: dict[str, list[tuple[datetime, str]]] = {}
    incoming_history_by_account: dict[str, list[tuple[datetime, str, float]]] = {}
    queries: list[QueryGold] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            event_id = str(row["transaction_id"])
            event_time = _parse_timestamp(row["timestamp"])
            src_account = str(row["src_account"])
            dst_account = str(row["dst_account"])
            amount = _parse_float(row.get("amount"))
            is_laundering = _is_positive_label(row.get("is_laundering"))

            _prune_history(laundering_history_by_account, src_account, event_time, same_entity_window)
            _prune_history(laundering_history_by_account, dst_account, event_time, same_entity_window)
            _prune_incoming(incoming_history_by_account, src_account, event_time, accumulation_window)
            _prune_incoming(incoming_history_by_account, dst_account, event_time, accumulation_window)

            if is_laundering:
                same_entity_ids = _same_entity_gold(
                    laundering_history_by_account,
                    src_account,
                    dst_account,
                )
                weak_accumulation_ids = _weak_accumulation_gold(
                    incoming_history_by_account.get(src_account, []),
                    threshold_amount=max(0.0, float(accumulation_threshold)) * amount,
                )
                queries.append(
                    QueryGold(
                        query_event_id=event_id,
                        query_time=event_time,
                        query_label="1",
                        same_entity_ids=same_entity_ids,
                        weak_accumulation_ids=weak_accumulation_ids,
                    )
                )
                if limit_queries is not None and len(queries) >= max(0, limit_queries):
                    break

            if is_laundering:
                laundering_history_by_account.setdefault(src_account, []).append((event_time, event_id))
                if dst_account != src_account:
                    laundering_history_by_account.setdefault(dst_account, []).append((event_time, event_id))
            incoming_history_by_account.setdefault(dst_account, []).append((event_time, event_id, amount))

    return queries


def _same_entity_gold(
    laundering_history_by_account: dict[str, list[tuple[datetime, str]]],
    src_account: str,
    dst_account: str,
) -> list[str]:
    merged: list[str] = []
    for account in (src_account, dst_account):
        for _time, event_id in laundering_history_by_account.get(account, []):
            if event_id not in merged:
                merged.append(event_id)
    return merged


def _weak_accumulation_gold(
    incoming_events: list[tuple[datetime, str, float]],
    *,
    threshold_amount: float,
) -> list[str]:
    if threshold_amount <= 0.0:
        return []
    running = 0.0
    selected: list[str] = []
    for _time, event_id, amount in reversed(incoming_events):
        selected.append(event_id)
        running += max(0.0, amount)
        if running >= threshold_amount:
            return list(reversed(selected))
    return []


def _prune_history(
    history_by_account: dict[str, list[tuple[datetime, str]]],
    account: str,
    now: datetime,
    window: timedelta,
) -> None:
    history = history_by_account.get(account)
    if not history:
        return
    cutoff = now - window
    index = 0
    while index < len(history) and history[index][0] < cutoff:
        index += 1
    if index:
        history_by_account[account] = history[index:]


def _prune_incoming(
    incoming_by_account: dict[str, list[tuple[datetime, str, float]]],
    account: str,
    now: datetime,
    window: timedelta,
) -> None:
    incoming = incoming_by_account.get(account)
    if not incoming:
        return
    cutoff = now - window
    index = 0
    while index < len(incoming) and incoming[index][0] < cutoff:
        index += 1
    if index:
        incoming_by_account[account] = incoming[index:]


def _build_query_intent(event: Any) -> DecisionIntent:
    aspects: set[str] = set()
    amount = _parse_float(event.attrs.get("amount"))
    if amount >= 1_000.0:
        aspects.add("large_transfer")
    if event.attrs.get("dst_account") or event.attrs.get("beneficiary_account"):
        aspects.add("beneficiary_novelty")
    if not aspects:
        aspects.add("generic_evidence")
    return DecisionIntent(aspects=aspects, name="ibm_aml_realcase")


def _normalize_retrieval(evidence_objects: list[Any], query_event_id: str) -> tuple[list[str], list[str], list[str]]:
    event_ids: list[str] = []
    object_types: list[str] = []
    aspects: set[str] = set()
    for evidence in evidence_objects:
        object_id = str(getattr(evidence, "object_id", "evidence"))
        object_types.append(_object_type(object_id))
        aspects.update(str(aspect) for aspect in getattr(evidence, "aspects", ()))
        for event_id in getattr(evidence, "event_ids", ()):
            event_id_text = str(event_id)
            if event_id_text != query_event_id and event_id_text not in event_ids:
                event_ids.append(event_id_text)
    return event_ids, object_types, sorted(aspects)


def _aggregate_results(results: list[dict[str, Any]], budgets: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gold_policies = (
        ("same_entity", "gold_same_entity_ids"),
        ("weak_accumulation", "gold_weak_accumulation_ids"),
        ("union", "gold_union_ids"),
    )
    for policy_name, gold_key in gold_policies:
        for budget in budgets:
            bucket = [row for row in results if int(row["budget"]) == int(budget)]
            active_bucket = [row for row in bucket if row[gold_key]]
            rows.append(
                {
                    "gold_policy": policy_name,
                    "budget": int(budget),
                    "query_count": len(bucket),
                    "active_query_count": len(active_bucket),
                    "recall": _mean(
                        [evidence_recall_at_budget(row["retrieved_event_ids"], row[gold_key]) for row in active_bucket]
                    ),
                    "precision": _mean(
                        [evidence_precision_at_budget(row["retrieved_event_ids"], row[gold_key]) for row in active_bucket]
                    ),
                    "f1": _mean(
                        [evidence_f1_at_budget(row["retrieved_event_ids"], row[gold_key]) for row in active_bucket]
                    ),
                    "context_efficiency": _mean(
                        [
                            context_efficiency(
                                row["retrieved_event_ids"],
                                row[gold_key],
                                budget_used=min(int(budget), len(row["retrieved_event_ids"])) or int(budget),
                            )
                            for row in active_bucket
                        ]
                    ),
                    "latency_ms": mean_latency_ms([float(row["latency_ms"]) for row in bucket]),
                    "mean_gold_size": _mean([len(row[gold_key]) for row in active_bucket]),
                    "mean_retrieved_size": _mean([len(row["retrieved_event_ids"]) for row in bucket]),
                }
            )
    return rows


def _object_type(object_id: str) -> str:
    if object_id.startswith("skip:"):
        return "skip"
    if object_id.startswith("ordinary:"):
        return "ordinary"
    return "chain"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%Y/%m/%d %H:%M")


def _parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_positive_label(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "laundering", "suspicious", "fraud"}


def _mean(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values)) / float(len(values))


if __name__ == "__main__":
    raise SystemExit(main())
