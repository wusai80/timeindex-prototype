"""Run a deterministic DeepSeek evaluation on sampled IBM AML cases."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

from benchmarks.ibm_aml.deepseek_agent import classify_query_with_deepseek
from timeindex import DecisionIntent
from timeindex.retrieval import retrieve
from timeindex.sqlite_backend import SqliteTimeIndexBackend


DEFAULT_OUTPUT_DIR = Path("outputs/ibm_aml/deepseek_sample")
DEFAULT_POSITIVE_MIN_INSERTION_ORDER = 4_000_000


def run_sqlite_deepseek_sample(
    index_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    positives: int = 100,
    negatives: int = 100,
    budget: int = 8,
    adaptive_budget: int | None = None,
    adaptive_min_events: int = 4,
    adaptive_min_aspects: int = 2,
    min_insertion_order: int | None = None,
    positive_min_insertion_order: int | None = None,
    negative_min_insertion_order: int | None = None,
    seed: int = 0,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Run DeepSeek on a deterministic sample of positive and negative cases."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    progress_path = output_path / "progress.json"
    results_path = output_path / "results.json"

    sampled = _sample_query_ids(
        index_path,
        positives=positives,
        negatives=negatives,
        seed=seed,
        min_insertion_order=min_insertion_order,
        positive_min_insertion_order=positive_min_insertion_order,
        negative_min_insertion_order=negative_min_insertion_order,
    )
    query_ids = sampled["negative_ids"] + sampled["positive_ids"]

    index = SqliteTimeIndexBackend.open(index_path)
    intent = DecisionIntent(
        aspects={"large_transfer", "beneficiary_novelty", "generic_evidence"},
        name="deepseek_sqlite_sample",
    )

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    retrieval_latencies: list[float] = []
    llm_latencies: list[float] = []
    effective_budgets: list[float] = []

    def write_progress(stage: str) -> None:
        payload = {
            "stage": stage,
            "index_path": str(index_path),
            "budget": int(budget),
            "adaptive_budget": None if adaptive_budget is None else int(adaptive_budget),
            "adaptive_min_events": int(adaptive_min_events),
            "adaptive_min_aspects": int(adaptive_min_aspects),
            "min_insertion_order": None if min_insertion_order is None else int(min_insertion_order),
            "positive_min_insertion_order": None if positive_min_insertion_order is None else int(positive_min_insertion_order),
            "negative_min_insertion_order": None if negative_min_insertion_order is None else int(negative_min_insertion_order),
            "seed": int(seed),
            "positive_target": int(positives),
            "negative_target": int(negatives),
            "processed_queries": len(results),
            "total_queries": len(query_ids),
            "errors": errors,
            "positive_summary": _label_summary(results, positive=True),
            "negative_summary": _label_summary(results, positive=False),
            "mean_retrieval_latency_ms": _mean(retrieval_latencies),
            "mean_llm_latency_ms": _mean(llm_latencies),
            "mean_effective_budget": _mean(effective_budgets),
            "max_llm_latency_ms": max(llm_latencies) if llm_latencies else 0.0,
            "cache_stats": index.cache_stats(),
        }
        progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    try:
        write_progress("starting")
        for position, query_id in enumerate(query_ids, start=1):
            query_record = index.get_event(query_id)
            if query_record is None:
                errors.append({"query_event_id": query_id, "error": "missing query record"})
                write_progress("running")
                continue

            retrieval_started = time.perf_counter()
            initial_retrieval = _collect_retrieval(index, query_id, budget)
            effective_budget = int(budget)
            if _should_expand_budget(
                initial_retrieval["event_ids"],
                initial_retrieval["aspects"],
                adaptive_budget=adaptive_budget,
                base_budget=budget,
                adaptive_min_events=adaptive_min_events,
                adaptive_min_aspects=adaptive_min_aspects,
            ):
                expanded_retrieval = _collect_retrieval(index, query_id, int(adaptive_budget or budget))
                if len(expanded_retrieval["event_ids"]) >= len(initial_retrieval["event_ids"]):
                    initial_retrieval = expanded_retrieval
                    effective_budget = int(adaptive_budget or budget)
            retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000.0
            retrieval_latencies.append(retrieval_latency_ms)
            effective_budgets.append(float(effective_budget))

            retrieved_ids = list(initial_retrieval["event_ids"])
            retrieved_aspects = set(initial_retrieval["aspects"])
            retrieved_events = list(initial_retrieval["events"])

            decision = None
            last_error = None
            llm_latency_ms = 0.0
            for attempt in range(max(1, int(max_retries))):
                try:
                    llm_started = time.perf_counter()
                    decision = classify_query_with_deepseek(
                        query_record.event,
                        retrieved_events,
                        sorted(retrieved_aspects),
                    )
                    llm_latency_ms = (time.perf_counter() - llm_started) * 1000.0
                    llm_latencies.append(llm_latency_ms)
                    break
                except Exception as exc:  # pragma: no cover - network variability
                    last_error = str(exc)
                    time.sleep(1.5 * (attempt + 1))

            if decision is None:
                errors.append({"query_event_id": query_id, "error": last_error or "unknown llm error"})
                write_progress("running")
                continue

            row = decision.to_dict()
            row.update(
                {
                    "retrieval_budget": int(budget),
                    "effective_retrieval_budget": int(effective_budget),
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "llm_latency_ms": llm_latency_ms,
                    "query_time": query_record.event.time,
                    "query_attrs": dict(query_record.event.attrs),
                }
            )
            results.append(row)
            print(
                f"[{position}/{len(query_ids)}] {query_id} label={query_record.event.label} "
                f"predicted={decision.predicted_positive} conf={decision.confidence:.2f} "
                f"retrieved={len(decision.retrieved_event_ids)} support={len(decision.supporting_event_ids)}",
                flush=True,
            )
            write_progress("running")

        summary = {
            "index_path": str(index_path),
            "budget": int(budget),
            "adaptive_budget": None if adaptive_budget is None else int(adaptive_budget),
            "adaptive_min_events": int(adaptive_min_events),
            "adaptive_min_aspects": int(adaptive_min_aspects),
            "min_insertion_order": None if min_insertion_order is None else int(min_insertion_order),
            "positive_min_insertion_order": None if positive_min_insertion_order is None else int(positive_min_insertion_order),
            "negative_min_insertion_order": None if negative_min_insertion_order is None else int(negative_min_insertion_order),
            "seed": int(seed),
            "positive_target": int(positives),
            "negative_target": int(negatives),
            "positive_summary": _label_summary(results, positive=True),
            "negative_summary": _label_summary(results, positive=False),
            "mean_retrieval_latency_ms": _mean(retrieval_latencies),
            "mean_llm_latency_ms": _mean(llm_latencies),
            "mean_effective_budget": _mean(effective_budgets),
            "max_llm_latency_ms": max(llm_latencies) if llm_latencies else 0.0,
            "errors": errors,
            "cache_stats": index.cache_stats(),
        }
        results_path.write_text(
            json.dumps({"summary": summary, "results": results}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_progress("complete")
        return summary
    finally:
        index.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--positives", type=int, default=100)
    parser.add_argument("--negatives", type=int, default=100)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--adaptive-budget", type=int, default=None)
    parser.add_argument("--adaptive-min-events", type=int, default=4)
    parser.add_argument("--adaptive-min-aspects", type=int, default=2)
    parser.add_argument("--min-insertion-order", type=int, default=None)
    parser.add_argument("--positive-min-insertion-order", type=int, default=None)
    parser.add_argument("--negative-min-insertion-order", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_sqlite_deepseek_sample(
        args.index_path,
        output_dir=args.output_dir,
        positives=args.positives,
        negatives=args.negatives,
        budget=args.budget,
        adaptive_budget=args.adaptive_budget,
        adaptive_min_events=args.adaptive_min_events,
        adaptive_min_aspects=args.adaptive_min_aspects,
        min_insertion_order=args.min_insertion_order,
        positive_min_insertion_order=args.positive_min_insertion_order,
        negative_min_insertion_order=args.negative_min_insertion_order,
        seed=args.seed,
        max_retries=args.max_retries,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _collect_retrieval(
    index: SqliteTimeIndexBackend,
    query_id: str,
    budget: int,
) -> dict[str, Any]:
    intent = DecisionIntent(
        aspects={"large_transfer", "beneficiary_novelty", "generic_evidence"},
        name="deepseek_sqlite_sample",
    )
    evidence_objects = retrieve(index, query_id, intent, budget)
    retrieved_ids: list[str] = []
    retrieved_aspects: set[str] = set()
    retrieved_events = []
    for evidence in evidence_objects:
        retrieved_aspects.update(str(aspect) for aspect in getattr(evidence, "aspects", ()))
        for event_id in getattr(evidence, "event_ids", ()):
            event_id_text = str(event_id)
            if event_id_text == query_id or event_id_text in retrieved_ids:
                continue
            retrieved_ids.append(event_id_text)
    for event_id in retrieved_ids[:budget]:
        record = index.get_event(event_id)
        if record is not None:
            retrieved_events.append(record.event)
    return {
        "event_ids": retrieved_ids[:budget],
        "aspects": sorted(retrieved_aspects),
        "events": retrieved_events,
    }


def _should_expand_budget(
    retrieved_event_ids: list[str],
    retrieved_aspects: list[str] | set[str],
    *,
    adaptive_budget: int | None,
    base_budget: int,
    adaptive_min_events: int,
    adaptive_min_aspects: int,
) -> bool:
    if adaptive_budget is None or int(adaptive_budget) <= int(base_budget):
        return False
    if len(retrieved_event_ids) < max(0, int(adaptive_min_events)):
        return True
    return len(set(str(aspect) for aspect in retrieved_aspects if str(aspect))) < max(0, int(adaptive_min_aspects))


def _sample_query_ids(
    index_path: str | Path,
    *,
    positives: int,
    negatives: int,
    seed: int,
    min_insertion_order: int | None = None,
    positive_min_insertion_order: int | None = None,
    negative_min_insertion_order: int | None = None,
) -> dict[str, list[str]]:
    rng = random.Random(int(seed))
    positive_reservoir: list[str] = []
    negative_reservoir: list[str] = []
    positive_seen = 0
    negative_seen = 0

    positive_cutoff, negative_cutoff = _resolve_cutoffs(
        min_insertion_order=min_insertion_order,
        positive_min_insertion_order=positive_min_insertion_order,
        negative_min_insertion_order=negative_min_insertion_order,
    )

    connection = sqlite3.connect(str(index_path))
    try:
        cursor = connection.execute(
            """
            SELECT event_id, label_value, insertion_order
            FROM events
            WHERE expired = 0
            ORDER BY insertion_order ASC, event_id ASC
            """
        )
        for event_id, label_value, insertion_order in cursor:
            label_is_positive = _is_positive_label(label_value)
            if label_is_positive:
                if positive_cutoff is not None and int(insertion_order) < positive_cutoff:
                    continue
                positive_seen += 1
                _reservoir_add(positive_reservoir, str(event_id), positive_seen, max(0, int(positives)), rng)
            else:
                if negative_cutoff is not None and int(insertion_order) < negative_cutoff:
                    continue
                negative_seen += 1
                _reservoir_add(negative_reservoir, str(event_id), negative_seen, max(0, int(negatives)), rng)
    finally:
        connection.close()

    positive_reservoir.sort()
    negative_reservoir.sort()
    return {
        "positive_ids": positive_reservoir,
        "negative_ids": negative_reservoir,
    }


def _resolve_cutoffs(
    *,
    min_insertion_order: int | None,
    positive_min_insertion_order: int | None,
    negative_min_insertion_order: int | None,
) -> tuple[int | None, int | None]:
    shared = None if min_insertion_order is None else int(min_insertion_order)
    if positive_min_insertion_order is None:
        positive = DEFAULT_POSITIVE_MIN_INSERTION_ORDER if shared is None else shared
    else:
        positive = int(positive_min_insertion_order)
    negative = shared if negative_min_insertion_order is None else int(negative_min_insertion_order)
    return positive, negative


def _reservoir_add(
    reservoir: list[str],
    item: str,
    seen: int,
    size: int,
    rng: random.Random,
) -> None:
    if size <= 0:
        return
    if len(reservoir) < size:
        reservoir.append(item)
        return
    replacement_index = rng.randrange(seen)
    if replacement_index < size:
        reservoir[replacement_index] = item


def _is_positive_label(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "laundering", "suspicious", "fraud"}


def _label_summary(results: list[dict[str, Any]], *, positive: bool) -> dict[str, float]:
    matching = [
        item for item in results
        if _is_positive_label(item.get("query_label")) == positive
    ]
    if not matching:
        return {
            "count": 0.0,
            "predicted_positive_rate": 0.0,
            "mean_confidence": 0.0,
            "mean_retrieved_events": 0.0,
            "mean_supporting_events": 0.0,
        }
    return {
        "count": float(len(matching)),
        "predicted_positive_rate": sum(1 for item in matching if item.get("predicted_positive")) / len(matching),
        "mean_confidence": _mean(float(item.get("confidence", 0.0)) for item in matching),
        "mean_retrieved_events": _mean(len(item.get("retrieved_event_ids", [])) for item in matching),
        "mean_supporting_events": _mean(len(item.get("supporting_event_ids", [])) for item in matching),
    }


def _mean(values: Any) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


if __name__ == "__main__":
    raise SystemExit(main())
