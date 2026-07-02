"""Compare retrieval-side TimeIndex ablations on an existing SQLite index."""

from __future__ import annotations

import argparse
import csv
from datetime import timedelta
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.baselines import chain_only_retrieval
from benchmarks.ibm_aml.run_sqlite_realcases import (
    DEFAULT_BUDGETS,
    QueryGold,
    _build_query_intent,
    _mean,
    _normalize_retrieval,
    _stream_realcase_gold,
)
from benchmarks.metrics import (
    context_efficiency,
    evidence_f1_at_budget,
    evidence_precision_at_budget,
    evidence_recall_at_budget,
    mean_latency_ms,
)
from timeindex.retrieval import retrieve
from timeindex.sqlite_backend import SqliteTimeIndexBackend


class _EmptySkipLinkStore:
    """Skip-link store view used for retrieval-time no-skip ablations."""

    def incoming(self, _event_id: str) -> list[Any]:
        return []


class _NoSkipIndexView:
    """Proxy over a SQLite backend that suppresses skip links during retrieval."""

    def __init__(self, backend: SqliteTimeIndexBackend) -> None:
        self._backend = backend
        self.config = backend.config
        self.event_store = backend.event_store
        self.edge_store = backend.edge_store
        self.chain_store = backend.chain_store
        self.skip_link_store = _EmptySkipLinkStore()

    def get_event(self, event_id: str) -> Any:
        return self._backend.get_event(event_id)


def run_sqlite_ablation_compare(
    csv_path: str | Path,
    index_path: str | Path,
    *,
    output_dir: str | Path = "outputs/ibm_aml/sqlite_ablation_compare",
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    same_entity_window_hours: int = 24,
    accumulation_window_hours: int = 24,
    accumulation_threshold: float = 0.8,
    limit_queries: int | None = None,
) -> dict[str, Any]:
    """Run retrieval-side ablation comparisons on an existing SQLite index."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    query_gold = _stream_realcase_gold(
        Path(csv_path),
        same_entity_window=timedelta(hours=max(1, same_entity_window_hours)),
        accumulation_window=timedelta(hours=max(1, accumulation_window_hours)),
        accumulation_threshold=float(accumulation_threshold),
        limit_queries=limit_queries,
    )

    backend = SqliteTimeIndexBackend.open(index_path)
    no_skip_backend = _NoSkipIndexView(backend)
    results: list[dict[str, Any]] = []

    try:
        for query in query_gold:
            query_record = backend.get_event(query.query_event_id)
            if query_record is None:
                continue
            intent = _build_query_intent(query_record.event)
            for budget in budgets:
                for variant_name, variant_mode in (
                    ("chain_only", "chain_only"),
                    ("timeindex_no_skip", "no_skip"),
                    ("timeindex_full", "full"),
                ):
                    started = perf_counter()
                    evidence_objects = _run_variant(
                        variant_mode,
                        backend,
                        no_skip_backend,
                        query.query_event_id,
                        intent,
                        int(budget),
                    )
                    latency_ms = (perf_counter() - started) * 1000.0
                    retrieved_ids, object_types, aspects = _normalize_variant_retrieval(
                        evidence_objects,
                        query.query_event_id,
                    )
                    results.append(
                        {
                            "variant": variant_name,
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
                            "budget_used": min(int(budget), len(retrieved_ids)),
                        }
                    )
    finally:
        backend.close()

    results_path = output_path / "retrieval_results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    aggregate_rows = _aggregate_variant_results(results, budgets)
    aggregate_path = output_path / "aggregate.csv"
    _write_csv(aggregate_path, aggregate_rows)

    summary = {
        "csv_path": str(csv_path),
        "index_path": str(index_path),
        "output_dir": str(output_path),
        "query_count": len(query_gold),
        "budget_count": len(tuple(budgets)),
        "variants": ["chain_only", "timeindex_no_skip", "timeindex_full"],
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
    _write_markdown_report(output_path / "report.md", summary, aggregate_rows)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("index_path")
    parser.add_argument("--output-dir", default="outputs/ibm_aml/sqlite_ablation_compare")
    parser.add_argument("--same-entity-window-hours", type=int, default=24)
    parser.add_argument("--accumulation-window-hours", type=int, default=24)
    parser.add_argument("--accumulation-threshold", type=float, default=0.8)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--budgets", nargs="*", type=int, default=list(DEFAULT_BUDGETS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_sqlite_ablation_compare(
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


def _run_variant(
    variant_mode: str,
    backend: SqliteTimeIndexBackend,
    no_skip_backend: _NoSkipIndexView,
    query_event_id: str,
    intent: Any,
    budget: int,
) -> list[Any]:
    if variant_mode == "chain_only":
        return chain_only_retrieval(backend, query_event_id, budget)
    if variant_mode == "no_skip":
        return retrieve(no_skip_backend, query_event_id, intent, budget)
    return retrieve(backend, query_event_id, intent, budget)


def _normalize_variant_retrieval(evidence_objects: list[Any], query_event_id: str) -> tuple[list[str], list[str], list[str]]:
    if evidence_objects and isinstance(evidence_objects[0], dict):
        event_ids: list[str] = []
        object_types: list[str] = []
        aspects: set[str] = set()
        for item in evidence_objects:
            object_types.append(str(item.get("type", "event")))
            aspects.update(str(aspect) for aspect in item.get("aspects", ()))
            for event_id in item.get("event_ids", ()):
                event_id_text = str(event_id)
                if event_id_text != query_event_id and event_id_text not in event_ids:
                    event_ids.append(event_id_text)
        return event_ids, object_types, sorted(aspects)
    return _normalize_retrieval(evidence_objects, query_event_id)


def _aggregate_variant_results(results: list[dict[str, Any]], budgets: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gold_policies = (
        ("same_entity", "gold_same_entity_ids"),
        ("weak_accumulation", "gold_weak_accumulation_ids"),
        ("union", "gold_union_ids"),
    )
    variants = ("chain_only", "timeindex_no_skip", "timeindex_full")
    for variant in variants:
        variant_rows = [row for row in results if row["variant"] == variant]
        for policy_name, gold_key in gold_policies:
            for budget in budgets:
                bucket = [row for row in variant_rows if int(row["budget"]) == int(budget)]
                active_bucket = [row for row in bucket if row[gold_key]]
                rows.append(
                    {
                        "variant": variant,
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
                                    budget_used=row["budget_used"] or int(budget),
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# SQLite Retrieval Ablation Compare")
    lines.append("")
    lines.append(f"- Dataset slice: `{summary['csv_path']}`")
    lines.append(f"- Index: `{summary['index_path']}`")
    lines.append(f"- Laundering queries evaluated: `{summary['query_count']}`")
    lines.append(f"- Variants: `{', '.join(summary['variants'])}`")
    lines.append("")
    for policy in ("same_entity", "weak_accumulation", "union"):
        lines.append(f"## {policy}")
        lines.append("")
        lines.append("| variant | budget | recall | precision | f1 | context_efficiency | latency_ms |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        subset = [row for row in rows if row["gold_policy"] == policy]
        subset.sort(key=lambda row: (int(row["budget"]), str(row["variant"])))
        for row in subset:
            lines.append(
                f"| {row['variant']} | {row['budget']} | {float(row['recall']):.3f} | {float(row['precision']):.3f} | {float(row['f1']):.3f} | {float(row['context_efficiency']):.3f} | {float(row['latency_ms']):.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
