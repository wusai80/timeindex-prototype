"""Distance-aware retrieval benchmark for the IBM AML TimeIndex evaluation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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
from benchmarks.ibm_aml.run_sqlite_ablation_compare import _NoSkipIndexView
from benchmarks.ibm_aml.run_sqlite_realcases import (
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


DEFAULT_BUDGETS: tuple[int, ...] = (4, 8, 12, 20)
DEFAULT_LATE_INSERTION_ORDER = 4_000_000


@dataclass(slots=True)
class QueryProfile:
    """Benchmark-ready query description with difficulty annotations."""

    query_event_id: str
    query_label: str
    query_time: float
    insertion_order: int
    same_entity_ids: list[str]
    weak_accumulation_ids: list[str]
    union_ids: list[str]
    gold_size: int
    same_entity_count: int
    weak_accumulation_count: int
    oldest_gold_gap_seconds: float
    nearest_gold_gap_seconds: float
    structure_bucket: str


def run_distance_benchmark(
    csv_path: str | Path,
    index_path: str | Path,
    *,
    output_dir: str | Path = "outputs/ibm_aml/distance_benchmark",
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    same_entity_window_hours: int = 24,
    accumulation_window_hours: int = 24,
    accumulation_threshold: float = 0.8,
    late_insertion_order: int = DEFAULT_LATE_INSERTION_ORDER,
    query_limit: int = 100,
) -> dict[str, Any]:
    """Evaluate retrieval variants on late-timeline laundering queries."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    backend = SqliteTimeIndexBackend.open(index_path)
    no_skip_backend = _NoSkipIndexView(backend)
    try:
        profiles = _build_query_profiles(
            Path(csv_path),
            backend,
            same_entity_window_hours=same_entity_window_hours,
            accumulation_window_hours=accumulation_window_hours,
            accumulation_threshold=accumulation_threshold,
            late_insertion_order=late_insertion_order,
            query_limit=query_limit,
        )

        gap_threshold = _median_int([profile.oldest_gold_gap_seconds for profile in profiles])
        size_threshold = _median_int([profile.gold_size for profile in profiles])

        results: list[dict[str, Any]] = []
        for profile in profiles:
            query_record = backend.get_event(profile.query_event_id)
            if query_record is None:
                continue
            intent = _build_query_intent(query_record.event)
            gap_bucket = _gap_bucket(profile.oldest_gold_gap_seconds, gap_threshold)
            size_bucket = _size_bucket(profile.gold_size, size_threshold)
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
                        profile.query_event_id,
                        intent,
                        int(budget),
                    )
                    latency_ms = (perf_counter() - started) * 1000.0
                    retrieved_ids, object_types, aspects = _normalize_variant_retrieval(
                        evidence_objects,
                        profile.query_event_id,
                    )
                    budget_used = min(int(budget), len(retrieved_ids)) or int(budget)
                    results.append(
                        {
                            "variant": variant_name,
                            "query_event_id": profile.query_event_id,
                            "query_label": profile.query_label,
                            "query_time": profile.query_time,
                            "insertion_order": profile.insertion_order,
                            "budget": int(budget),
                            "retrieved_event_ids": retrieved_ids[:budget],
                            "retrieved_object_types": object_types,
                            "retrieved_aspects": aspects,
                            "gold_same_entity_ids": list(profile.same_entity_ids),
                            "gold_weak_accumulation_ids": list(profile.weak_accumulation_ids),
                            "gold_union_ids": list(profile.union_ids),
                            "gold_size": profile.gold_size,
                            "same_entity_count": profile.same_entity_count,
                            "weak_accumulation_count": profile.weak_accumulation_count,
                            "oldest_gold_gap_seconds": profile.oldest_gold_gap_seconds,
                            "nearest_gold_gap_seconds": profile.nearest_gold_gap_seconds,
                            "gap_bucket": gap_bucket,
                            "size_bucket": size_bucket,
                            "structure_bucket": profile.structure_bucket,
                            "latency_ms": latency_ms,
                            "budget_used": budget_used,
                        }
                    )
    finally:
        backend.close()

    results_path = output_path / "retrieval_results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    overall_rows = _aggregate_overall(results, budgets)
    strata_rows = _aggregate_strata(results, budgets)
    skip_rows = _aggregate_skip_deltas(results, budgets)
    _write_csv(output_path / "overall.csv", overall_rows)
    _write_csv(output_path / "strata.csv", strata_rows)
    _write_csv(output_path / "skip_delta.csv", skip_rows)

    summary = {
        "csv_path": str(csv_path),
        "index_path": str(index_path),
        "output_dir": str(output_path),
        "query_count": len(profiles),
        "budgets": [int(budget) for budget in budgets],
        "late_insertion_order": int(late_insertion_order),
        "query_limit": int(query_limit),
        "gap_threshold_seconds": float(gap_threshold),
        "size_threshold": int(size_threshold),
        "results_path": str(results_path),
        "overall_path": str(output_path / "overall.csv"),
        "strata_path": str(output_path / "strata.csv"),
        "skip_delta_path": str(output_path / "skip_delta.csv"),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(output_path / "report.md", summary, overall_rows, strata_rows, skip_rows)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("index_path")
    parser.add_argument("--output-dir", default="outputs/ibm_aml/distance_benchmark")
    parser.add_argument("--same-entity-window-hours", type=int, default=24)
    parser.add_argument("--accumulation-window-hours", type=int, default=24)
    parser.add_argument("--accumulation-threshold", type=float, default=0.8)
    parser.add_argument("--late-insertion-order", type=int, default=DEFAULT_LATE_INSERTION_ORDER)
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--budgets", nargs="*", type=int, default=list(DEFAULT_BUDGETS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_distance_benchmark(
        args.csv_path,
        args.index_path,
        output_dir=args.output_dir,
        budgets=tuple(args.budgets),
        same_entity_window_hours=args.same_entity_window_hours,
        accumulation_window_hours=args.accumulation_window_hours,
        accumulation_threshold=args.accumulation_threshold,
        late_insertion_order=args.late_insertion_order,
        query_limit=args.query_limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _build_query_profiles(
    csv_path: Path,
    backend: SqliteTimeIndexBackend,
    *,
    same_entity_window_hours: int,
    accumulation_window_hours: int,
    accumulation_threshold: float,
    late_insertion_order: int,
    query_limit: int,
) -> list[QueryProfile]:
    query_gold = _stream_realcase_gold(
        csv_path,
        same_entity_window=timedelta(hours=max(1, same_entity_window_hours)),
        accumulation_window=timedelta(hours=max(1, accumulation_window_hours)),
        accumulation_threshold=float(accumulation_threshold),
        limit_queries=None,
    )
    candidates: list[QueryProfile] = []
    for query in query_gold:
        profile = _profile_query(query, backend)
        if profile is None:
            continue
        if profile.insertion_order < int(late_insertion_order):
            continue
        candidates.append(profile)

    candidates.sort(key=lambda item: (item.insertion_order, item.query_event_id))
    if query_limit > 0:
        candidates = candidates[-int(query_limit):]
    return candidates


def _profile_query(query: QueryGold, backend: SqliteTimeIndexBackend) -> QueryProfile | None:
    query_record = backend.get_event(query.query_event_id)
    if query_record is None or query_record.metadata.insertion_order is None:
        return None

    query_time = _time_value(query_record.event.time)
    valid_same: list[str] = []
    valid_weak: list[str] = []
    gold_times: dict[str, float] = {}

    for event_id in query.same_entity_ids:
        gold_time = _gold_event_time(backend, event_id, query_time)
        if gold_time is None or event_id in valid_same:
            continue
        valid_same.append(event_id)
        gold_times[event_id] = gold_time

    for event_id in query.weak_accumulation_ids:
        gold_time = _gold_event_time(backend, event_id, query_time)
        if gold_time is None or event_id in valid_weak:
            continue
        valid_weak.append(event_id)
        gold_times[event_id] = gold_time

    union_ids: list[str] = []
    for event_id in valid_same + valid_weak:
        if event_id not in union_ids:
            union_ids.append(event_id)

    if not union_ids:
        return None

    gaps = [max(0.0, query_time - gold_times[event_id]) for event_id in union_ids if event_id in gold_times]
    if not gaps:
        return None

    return QueryProfile(
        query_event_id=query.query_event_id,
        query_label=query.query_label,
        query_time=query_time,
        insertion_order=int(query_record.metadata.insertion_order),
        same_entity_ids=valid_same,
        weak_accumulation_ids=valid_weak,
        union_ids=union_ids,
        gold_size=len(union_ids),
        same_entity_count=len(valid_same),
        weak_accumulation_count=len(valid_weak),
        oldest_gold_gap_seconds=max(gaps),
        nearest_gold_gap_seconds=min(gaps),
        structure_bucket=_structure_bucket(valid_same, valid_weak),
    )


def _gold_event_time(backend: SqliteTimeIndexBackend, event_id: str, query_time: float) -> float | None:
    record = backend.get_event(event_id)
    if record is None:
        return None
    event_time = _time_value(record.event.time)
    if event_time >= query_time:
        return None
    return event_time


def _structure_bucket(same_entity_ids: list[str], weak_accumulation_ids: list[str]) -> str:
    same_count = len(same_entity_ids)
    weak_count = len(weak_accumulation_ids)
    if same_count and weak_count:
        if weak_count > same_count:
            return "flow_chain_heavy"
        if same_count > weak_count:
            return "same_entity_heavy"
        return "mixed"
    if weak_count:
        return "flow_chain_heavy"
    if same_count:
        return "same_entity_heavy"
    return "unlabeled"


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


def _normalize_variant_retrieval(
    evidence_objects: list[Any],
    query_event_id: str,
) -> tuple[list[str], list[str], list[str]]:
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


def _aggregate_overall(results: list[dict[str, Any]], budgets: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in ("chain_only", "timeindex_no_skip", "timeindex_full"):
        variant_rows = [row for row in results if row["variant"] == variant]
        for budget in budgets:
            bucket = [row for row in variant_rows if int(row["budget"]) == int(budget)]
            rows.append(
                _metric_row(bucket, variant=variant, budget=int(budget), dimension="overall", bucket_name="all")
            )
    return rows


def _aggregate_strata(results: list[dict[str, Any]], budgets: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in ("gap_bucket", "size_bucket", "structure_bucket"):
        values = sorted({str(row[dimension]) for row in results})
        for variant in ("chain_only", "timeindex_no_skip", "timeindex_full"):
            variant_rows = [row for row in results if row["variant"] == variant]
            for budget in budgets:
                budget_rows = [row for row in variant_rows if int(row["budget"]) == int(budget)]
                for value in values:
                    bucket = [row for row in budget_rows if str(row[dimension]) == value]
                    rows.append(
                        _metric_row(
                            bucket,
                            variant=variant,
                            budget=int(budget),
                            dimension=dimension,
                            bucket_name=value,
                        )
                    )
    return rows


def _aggregate_skip_deltas(results: list[dict[str, Any]], budgets: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        full_rows = {
            row["query_event_id"]: row
            for row in results
            if row["variant"] == "timeindex_full" and int(row["budget"]) == int(budget)
        }
        no_skip_rows = {
            row["query_event_id"]: row
            for row in results
            if row["variant"] == "timeindex_no_skip" and int(row["budget"]) == int(budget)
        }
        common_query_ids = sorted(set(full_rows) & set(no_skip_rows))
        comparisons = [_compare_skip_effect(full_rows[event_id], no_skip_rows[event_id]) for event_id in common_query_ids]
        rows.append(_skip_delta_row(comparisons, budget=int(budget), dimension="overall", bucket="all"))
        for dimension in ("gap_bucket", "size_bucket", "structure_bucket"):
            values = sorted({str(full_rows[event_id][dimension]) for event_id in common_query_ids})
            for value in values:
                subset = [
                    comparison
                    for comparison in comparisons
                    if str(comparison[dimension]) == value
                ]
                rows.append(_skip_delta_row(subset, budget=int(budget), dimension=dimension, bucket=value))
    return rows


def _metric_row(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    budget: int,
    dimension: str,
    bucket_name: str,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "budget": int(budget),
        "dimension": dimension,
        "bucket": bucket_name,
        "query_count": len(rows),
        "recall": _mean([evidence_recall_at_budget(row["retrieved_event_ids"], row["gold_union_ids"]) for row in rows]),
        "precision": _mean(
            [evidence_precision_at_budget(row["retrieved_event_ids"], row["gold_union_ids"]) for row in rows]
        ),
        "f1": _mean([evidence_f1_at_budget(row["retrieved_event_ids"], row["gold_union_ids"]) for row in rows]),
        "context_efficiency": _mean(
            [
                context_efficiency(
                    row["retrieved_event_ids"],
                    row["gold_union_ids"],
                    budget_used=row["budget_used"],
                )
                for row in rows
            ]
        ),
        "latency_ms": mean_latency_ms([float(row["latency_ms"]) for row in rows]),
        "mean_gold_size": _mean([int(row["gold_size"]) for row in rows]),
        "mean_retrieved_size": _mean([len(row["retrieved_event_ids"]) for row in rows]),
        "mean_oldest_gap_seconds": _mean([float(row["oldest_gold_gap_seconds"]) for row in rows]),
    }


def _compare_skip_effect(full_row: dict[str, Any], no_skip_row: dict[str, Any]) -> dict[str, Any]:
    gold = set(str(event_id) for event_id in full_row["gold_union_ids"])
    full_ids = set(str(event_id) for event_id in full_row["retrieved_event_ids"])
    no_skip_ids = set(str(event_id) for event_id in no_skip_row["retrieved_event_ids"])
    full_gold = gold & full_ids
    no_skip_gold = gold & no_skip_ids
    new_gold = sorted(full_gold - no_skip_gold)
    lost_gold = sorted(no_skip_gold - full_gold)
    return {
        "query_event_id": full_row["query_event_id"],
        "gap_bucket": full_row["gap_bucket"],
        "size_bucket": full_row["size_bucket"],
        "structure_bucket": full_row["structure_bucket"],
        "win": bool(new_gold),
        "loss": bool(lost_gold),
        "identical": full_ids == no_skip_ids,
        "new_gold_count": len(new_gold),
        "lost_gold_count": len(lost_gold),
        "recall_delta": evidence_recall_at_budget(list(full_ids), gold) - evidence_recall_at_budget(list(no_skip_ids), gold),
        "precision_delta": evidence_precision_at_budget(list(full_ids), gold)
        - evidence_precision_at_budget(list(no_skip_ids), gold),
    }


def _skip_delta_row(
    comparisons: list[dict[str, Any]],
    *,
    budget: int,
    dimension: str,
    bucket: str,
) -> dict[str, Any]:
    count = len(comparisons)
    return {
        "budget": int(budget),
        "dimension": dimension,
        "bucket": bucket,
        "query_count": count,
        "skip_win_rate": _rate(sum(1 for row in comparisons if row["win"]), count),
        "skip_loss_rate": _rate(sum(1 for row in comparisons if row["loss"]), count),
        "identical_event_set_rate": _rate(sum(1 for row in comparisons if row["identical"]), count),
        "mean_new_gold_count": _mean([row["new_gold_count"] for row in comparisons]),
        "mean_lost_gold_count": _mean([row["lost_gold_count"] for row in comparisons]),
        "mean_recall_delta": _mean([row["recall_delta"] for row in comparisons]),
        "mean_precision_delta": _mean([row["precision_delta"] for row in comparisons]),
    }


def _gap_bucket(oldest_gap_seconds: float, threshold_seconds: float) -> str:
    if oldest_gap_seconds <= threshold_seconds:
        return "short_gap"
    return "long_gap"


def _size_bucket(gold_size: int, threshold: int) -> str:
    if gold_size <= threshold:
        return "small_gold"
    return "large_gold"


def _time_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        if not text:
            return 0.0
        from datetime import datetime

        return datetime.strptime(text, "%Y/%m/%d %H:%M").timestamp()


def _median_int(values: list[int | float]) -> int:
    if not values:
        return 0
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return int(ordered[middle])
    return int((ordered[middle - 1] + ordered[middle]) / 2.0)


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    summary: dict[str, Any],
    overall_rows: list[dict[str, Any]],
    strata_rows: list[dict[str, Any]],
    skip_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Distance-Aware AML Retrieval Benchmark")
    lines.append("")
    lines.append(f"- Dataset slice: `{summary['csv_path']}`")
    lines.append(f"- Index: `{summary['index_path']}`")
    lines.append(f"- Late positive queries: `{summary['query_count']}`")
    lines.append(f"- Budgets: `{summary['budgets']}`")
    lines.append(f"- Gap threshold (seconds): `{summary['gap_threshold_seconds']}`")
    lines.append(f"- Gold-size threshold: `{summary['size_threshold']}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| variant | budget | recall | precision | f1 | context_efficiency | latency_ms |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(overall_rows, key=lambda item: (int(item["budget"]), str(item["variant"]))):
        lines.append(
            f"| {row['variant']} | {row['budget']} | {float(row['recall']):.3f} | {float(row['precision']):.3f} | {float(row['f1']):.3f} | {float(row['context_efficiency']):.3f} | {float(row['latency_ms']):.3f} |"
        )
    lines.append("")
    lines.append("## Skip Delta")
    lines.append("")
    lines.append("| budget | bucket | skip_win_rate | skip_loss_rate | identical_event_set_rate | mean_recall_delta |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
    for row in sorted(
        [item for item in skip_rows if item["dimension"] == "overall"],
        key=lambda item: int(item["budget"]),
    ):
        lines.append(
            f"| {row['budget']} | {row['bucket']} | {float(row['skip_win_rate']):.3f} | {float(row['skip_loss_rate']):.3f} | {float(row['identical_event_set_rate']):.3f} | {float(row['mean_recall_delta']):.3f} |"
        )
    lines.append("")
    for dimension in ("gap_bucket", "size_bucket", "structure_bucket"):
        lines.append(f"## {dimension}")
        lines.append("")
        lines.append("| variant | budget | bucket | recall | precision | f1 | query_count |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
        subset = [row for row in strata_rows if row["dimension"] == dimension]
        subset.sort(key=lambda item: (int(item["budget"]), str(item["bucket"]), str(item["variant"])))
        for row in subset:
            lines.append(
                f"| {row['variant']} | {row['budget']} | {row['bucket']} | {float(row['recall']):.3f} | {float(row['precision']):.3f} | {float(row['f1']):.3f} | {row['query_count']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
