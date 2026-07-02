"""Compare retrieval-side TimeIndex ablations on a LANL SQLite index."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.baselines import chain_only_retrieval
from benchmarks.lanl.adapter import stream_events
from benchmarks.lanl.evidence import UNION, build_gold_evidence
from benchmarks.lanl.schema import default_auth_schema
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


DEFAULT_BUDGETS: tuple[int, ...] = (4, 8, 12, 20)


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
    auth_path: str | Path,
    redteam_path: str | Path,
    index_path: str | Path,
    *,
    output_dir: str | Path = "outputs/lanl/sqlite_ablation_compare",
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    evidence_window: int = 86_400,
    max_hops: int = 2,
    positive_query_limit: int | None = None,
    negative_query_sample_size: int = 0,
) -> dict[str, Any]:
    """Run retrieval-side ablation comparisons on an existing LANL SQLite index."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    events = [
        record.event
        for record in stream_events(
            auth_path,
            redteam_path,
            default_auth_schema(source_file=str(auth_path), redteam_file=str(redteam_path)),
            sort_by_time=True,
        )
    ]
    gold_evidence = build_gold_evidence(events, policy=UNION, window=evidence_window, max_hops=max_hops)
    event_time_by_id = {event.event_id: int(event.time) for event in events}
    query_ids = _select_query_ids(
        events,
        positive_query_limit=positive_query_limit,
        negative_query_sample_size=negative_query_sample_size,
    )

    backend = SqliteTimeIndexBackend.open(index_path)
    no_skip_backend = _NoSkipIndexView(backend)
    results: list[dict[str, Any]] = []

    try:
        for query_id in query_ids:
            query_record = backend.get_event(query_id)
            if query_record is None:
                continue
            intent = _build_query_intent(query_record.event)
            gold_ids = sorted(gold_evidence.get(query_id, set()))
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
                        query_id,
                        intent,
                        int(budget),
                    )
                    latency_ms = (perf_counter() - started) * 1000.0
                    retrieved_ids, object_types, aspects = _normalize_variant_retrieval(evidence_objects, query_id)
                    results.append(
                        {
                            "variant": variant_name,
                            "query_event_id": query_id,
                            "query_label": query_record.event.label,
                            "query_time": int(query_record.event.time),
                            "budget": int(budget),
                            "retrieved_event_ids": [
                                event_id
                                for event_id in retrieved_ids[:budget]
                                if event_time_by_id.get(event_id, int(query_record.event.time)) < int(query_record.event.time)
                            ],
                            "retrieved_object_types": object_types,
                            "retrieved_aspects": aspects,
                            "gold_union_ids": gold_ids,
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
        "auth_path": str(auth_path),
        "redteam_path": str(redteam_path),
        "index_path": str(index_path),
        "output_dir": str(output_path),
        "query_count": len(query_ids),
        "budget_count": len(tuple(budgets)),
        "variants": ["chain_only", "timeindex_no_skip", "timeindex_full"],
        "positive_query_count": sum(1 for event in events if event.event_id in query_ids and str(event.label) == "1"),
        "negative_query_count": sum(1 for event in events if event.event_id in query_ids and str(event.label) != "1"),
        "results_path": str(results_path),
        "aggregate_path": str(aggregate_path),
    }
    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown_report(output_path / "report.md", summary, aggregate_rows)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("auth_path")
    parser.add_argument("redteam_path")
    parser.add_argument("index_path")
    parser.add_argument("--output-dir", default="outputs/lanl/sqlite_ablation_compare")
    parser.add_argument("--evidence-window", type=int, default=86_400)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--positive-query-limit", type=int, default=None)
    parser.add_argument("--negative-query-sample-size", type=int, default=0)
    parser.add_argument("--budgets", nargs="*", type=int, default=list(DEFAULT_BUDGETS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_sqlite_ablation_compare(
        args.auth_path,
        args.redteam_path,
        args.index_path,
        output_dir=args.output_dir,
        budgets=tuple(args.budgets),
        evidence_window=args.evidence_window,
        max_hops=args.max_hops,
        positive_query_limit=args.positive_query_limit,
        negative_query_sample_size=args.negative_query_sample_size,
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


def _build_query_intent(event: Any) -> DecisionIntent:
    aspects = {
        "credential_reuse",
        "lateral_movement",
        "new_host_access",
        "rare_auth_path",
        "privilege_spread",
        "generic_evidence",
    }
    if str(event.attrs.get("is_anonymous_logon", "")).lower() == "true":
        aspects.discard("credential_reuse")
    return DecisionIntent(aspects=aspects, name=f"lanl:{event.event_id}")


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


def _aggregate_variant_results(results: list[dict[str, Any]], budgets: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variants = ("chain_only", "timeindex_no_skip", "timeindex_full")
    for variant in variants:
        variant_rows = [row for row in results if row["variant"] == variant]
        for budget in budgets:
            bucket = [row for row in variant_rows if int(row["budget"]) == int(budget)]
            active_bucket = [row for row in bucket if row["gold_union_ids"]]
            rows.append(
                {
                    "variant": variant,
                    "budget": int(budget),
                    "query_count": len(bucket),
                    "active_query_count": len(active_bucket),
                    "recall": _mean([evidence_recall_at_budget(row["retrieved_event_ids"], row["gold_union_ids"]) for row in active_bucket]),
                    "precision": _mean([evidence_precision_at_budget(row["retrieved_event_ids"], row["gold_union_ids"]) for row in active_bucket]),
                    "f1": _mean([evidence_f1_at_budget(row["retrieved_event_ids"], row["gold_union_ids"]) for row in active_bucket]),
                    "context_efficiency": _mean(
                        [
                            context_efficiency(
                                row["retrieved_event_ids"],
                                row["gold_union_ids"],
                                budget_used=row["budget_used"] or int(budget),
                            )
                            for row in active_bucket
                        ]
                    ),
                    "latency_ms": mean_latency_ms([float(row["latency_ms"]) for row in bucket]),
                    "mean_gold_size": _mean([len(row["gold_union_ids"]) for row in active_bucket]),
                    "mean_retrieved_size": _mean([len(row["retrieved_event_ids"]) for row in bucket]),
                    "mean_skip_objects": _mean(
                        [sum(1 for item in row["retrieved_object_types"] if item == "skip") for row in bucket]
                    ),
                }
            )
    return rows


def _select_query_ids(
    events: list[Any],
    *,
    positive_query_limit: int | None,
    negative_query_sample_size: int,
) -> list[str]:
    positives = [event.event_id for event in events if str(event.label) == "1"]
    negatives = [event.event_id for event in events if str(event.label) != "1"][: max(0, negative_query_sample_size)]
    if positive_query_limit is not None:
        positives = positives[: max(0, positive_query_limit)]
    return negatives + positives


def _object_type(object_id: str) -> str:
    if object_id.startswith("skip:"):
        return "skip"
    if object_id.startswith("ordinary:"):
        return "ordinary"
    return "chain"


def _mean(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values)) / float(len(values))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# LANL SQLite Retrieval Ablation Compare")
    lines.append("")
    lines.append(f"- Auth slice: `{summary['auth_path']}`")
    lines.append(f"- Red-team slice: `{summary['redteam_path']}`")
    lines.append(f"- Index: `{summary['index_path']}`")
    lines.append(f"- Query count: `{summary['query_count']}`")
    lines.append("")
    lines.append("| variant | budget | recall | precision | f1 | context_efficiency | latency_ms | mean_skip_objects |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    subset = sorted(rows, key=lambda row: (int(row["budget"]), str(row["variant"])))
    for row in subset:
        lines.append(
            f"| {row['variant']} | {row['budget']} | {float(row['recall']):.3f} | {float(row['precision']):.3f} | {float(row['f1']):.3f} | {float(row['context_efficiency']):.3f} | {float(row['latency_ms']):.3f} | {float(row['mean_skip_objects']):.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
