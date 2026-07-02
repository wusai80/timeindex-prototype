"""Run TimeIndex on the LANL auth benchmark slice."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmarks.lanl.adapter import stream_events
from benchmarks.lanl.evidence import UNION, build_gold_evidence
from benchmarks.lanl.profile_dataset import profile_auth_dataset
from benchmarks.lanl.schema import default_auth_schema
from timeindex import DecisionIntent, TimeIndexConfig
from timeindex.construction import TimeIndex
from timeindex.retrieval import retrieve


DEFAULT_BUDGETS: tuple[int, ...] = (4, 8, 12, 20)


def run_benchmark(
    auth_path: str | Path,
    redteam_path: str | Path,
    *,
    output_dir: str | Path = "outputs/lanl",
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    positive_query_limit: int | None = None,
    negative_query_sample_size: int = 0,
    random_seed: int = 0,
    evidence_window: int = 86_400,
    max_hops: int = 2,
    streaming_indexing: bool = True,
) -> dict[str, Any]:
    """Run a minimal LANL auth benchmark and write artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    schema = default_auth_schema(source_file=str(auth_path), redteam_file=str(redteam_path))
    records = list(stream_events(auth_path, redteam_path, schema, sort_by_time=True))
    events = [record.event for record in records]
    gold_evidence = build_gold_evidence(events, policy=UNION, window=evidence_window, max_hops=max_hops)
    dataset_profile = profile_auth_dataset(auth_path, redteam_path=redteam_path)

    config = TimeIndexConfig()
    index = TimeIndex(config)
    selected_query_ids = _select_query_ids(
        events,
        positive_query_limit=positive_query_limit,
        negative_query_sample_size=negative_query_sample_size,
        random_seed=random_seed,
    )
    event_time_by_id = {event.event_id: _sortable_time(event.time) for event in events}
    results_path = output_path / "retrieval_results.jsonl"
    latency_values: list[float] = []
    result_count = 0

    with results_path.open("w", encoding="utf-8") as handle:
        if streaming_indexing:
            for event in events:
                index.insert(event)
                if event.event_id not in selected_query_ids:
                    continue
                intent = _build_query_intent(event)
                gold_ids = _normalize_gold_ids(
                    gold_evidence.get(event.event_id, ()),
                    event.event_id,
                    event_time_by_id,
                )
                for budget in budgets:
                    started = perf_counter()
                    evidence_objects = list(retrieve(index, event.event_id, intent, budget))
                    latency_ms = (perf_counter() - started) * 1000.0
                    latency_values.append(latency_ms)
                    row = _serialize_result(
                        query_event=event,
                        evidence_objects=evidence_objects,
                        budget=budget,
                        gold_event_ids=gold_ids,
                        latency_ms=latency_ms,
                        event_time_by_id=event_time_by_id,
                        index=index,
                    )
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    result_count += 1
        else:
            for event in events:
                index.insert(event)
            for query_id in selected_query_ids:
                event = next(event for event in events if event.event_id == query_id)
                intent = _build_query_intent(event)
                gold_ids = _normalize_gold_ids(
                    gold_evidence.get(query_id, ()),
                    query_id,
                    event_time_by_id,
                )
                for budget in budgets:
                    started = perf_counter()
                    evidence_objects = list(retrieve(index, query_id, intent, budget))
                    latency_ms = (perf_counter() - started) * 1000.0
                    latency_values.append(latency_ms)
                    row = _serialize_result(
                        query_event=event,
                        evidence_objects=evidence_objects,
                        budget=budget,
                        gold_event_ids=gold_ids,
                        latency_ms=latency_ms,
                        event_time_by_id=event_time_by_id,
                        index=index,
                    )
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    result_count += 1

    config_path = output_path / "config.json"
    profile_path = output_path / "dataset_profile.json"
    summary_path = output_path / "run_summary.json"
    config_path.write_text(
        json.dumps(
            {
                "auth_path": str(auth_path),
                "redteam_path": str(redteam_path),
                "budgets": list(budgets),
                "positive_query_limit": positive_query_limit,
                "negative_query_sample_size": negative_query_sample_size,
                "random_seed": random_seed,
                "evidence_window": evidence_window,
                "max_hops": max_hops,
                "streaming_indexing": streaming_indexing,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    profile_path.write_text(json.dumps(dataset_profile, indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "query_count": len(selected_query_ids),
                "retrieval_count": result_count,
                "mean_latency_ms": (sum(latency_values) / len(latency_values)) if latency_values else 0.0,
                "max_latency_ms": max(latency_values) if latency_values else 0.0,
                "results_path": str(results_path),
                "streaming_indexing": streaming_indexing,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "results_path": str(results_path),
        "config_path": str(config_path),
        "dataset_profile_path": str(profile_path),
        "run_summary_path": str(summary_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("auth_path")
    parser.add_argument("redteam_path")
    parser.add_argument("--output-dir", default="outputs/lanl")
    parser.add_argument("--positive-query-limit", type=int, default=None)
    parser.add_argument("--negative-query-sample-size", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--evidence-window", type=int, default=86_400)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--budgets", nargs="*", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument(
        "--batch-indexing",
        action="store_true",
        help="Build the full index before querying instead of the default streaming insert+query loop.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_benchmark(
        args.auth_path,
        args.redteam_path,
        output_dir=args.output_dir,
        budgets=tuple(args.budgets),
        positive_query_limit=args.positive_query_limit,
        negative_query_sample_size=args.negative_query_sample_size,
        random_seed=args.random_seed,
        evidence_window=args.evidence_window,
        max_hops=args.max_hops,
        streaming_indexing=not args.batch_indexing,
    )
    return 0


def _select_query_ids(
    events: list[Any],
    *,
    positive_query_limit: int | None,
    negative_query_sample_size: int,
    random_seed: int,
) -> list[str]:
    positives = [event.event_id for event in events if str(event.label) == "1"]
    negatives = [event.event_id for event in events if str(event.label) != "1"]
    if positive_query_limit is not None:
        positives = positives[: max(0, positive_query_limit)]
    if negative_query_sample_size > 0 and negatives:
        rng = random.Random(random_seed)
        negatives = rng.sample(negatives, k=min(len(negatives), max(0, negative_query_sample_size)))
    else:
        negatives = []
    return negatives + positives


def _build_query_intent(event: Any) -> DecisionIntent:
    aspects: set[str] = set()
    attrs = event.attrs
    if attrs.get("is_cross_host_auth"):
        aspects.add("credential_reuse")
    if attrs.get("is_new_dst_for_user"):
        aspects.add("new_host_access")
    if attrs.get("prior_user_event_count", 0) or attrs.get("prior_user_host_count", 0):
        aspects.add("rare_auth_path")
    if attrs.get("is_cross_host_auth") and attrs.get("is_new_dst_for_user"):
        aspects.add("lateral_movement")
    if not aspects:
        aspects.add("generic_evidence")
    return DecisionIntent(aspects=aspects, name=f"lanl:{event.event_id}")


def _serialize_result(
    *,
    query_event: Any,
    evidence_objects: list[Any],
    budget: int,
    gold_event_ids: list[str],
    latency_ms: float,
    event_time_by_id: dict[str, float],
    index: Any,
) -> dict[str, Any]:
    query_time = event_time_by_id.get(query_event.event_id, float("inf"))
    retrieved_event_ids: list[str] = []
    retrieved_object_types: list[str] = []
    retrieved_aspects: set[str] = set()
    for evidence in evidence_objects:
        retrieved_object_types.append(_evidence_type(evidence))
        retrieved_aspects.update(str(aspect) for aspect in getattr(evidence, "aspects", ()))
        for event_id in getattr(evidence, "event_ids", ()):
            event_id_text = str(event_id)
            if event_id_text == query_event.event_id:
                continue
            if event_time_by_id.get(event_id_text, float("inf")) >= query_time:
                continue
            if event_id_text not in retrieved_event_ids:
                retrieved_event_ids.append(event_id_text)
    return {
        "query_event_id": query_event.event_id,
        "query_label": query_event.label,
        "budget": int(budget),
        "retrieved_event_ids": retrieved_event_ids[:budget],
        "retrieved_object_types": retrieved_object_types,
        "retrieved_aspects": sorted(retrieved_aspects),
        "gold_event_ids": gold_event_ids,
        "latency_ms": latency_ms,
        "index_stats": {
            "indexed_event_count": len(index.event_store.list()),
            "ordinary_links": len(index.ordinary_links(query_event.event_id)),
            "skip_links": len(index.skip_links(query_event.event_id)),
            "chains": len(index.chains(query_event.event_id)),
        },
    }


def _normalize_gold_ids(
    gold_ids: Iterable[Any],
    query_event_id: str,
    event_time_by_id: dict[str, float],
) -> list[str]:
    query_time = event_time_by_id.get(query_event_id, float("inf"))
    normalized: list[str] = []
    for event_id in gold_ids:
        event_id_text = str(event_id)
        if event_id_text == query_event_id:
            continue
        if event_time_by_id.get(event_id_text, float("inf")) >= query_time:
            continue
        if event_id_text not in normalized:
            normalized.append(event_id_text)
    return normalized


def _sortable_time(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _evidence_type(evidence: Any) -> str:
    object_id = str(getattr(evidence, "object_id", "evidence"))
    if object_id.startswith("skip:"):
        return "skip"
    if object_id.startswith("ordinary:"):
        return "ordinary"
    if object_id.startswith("chain:"):
        return "chain"
    return "event"


if __name__ == "__main__":
    raise SystemExit(main())
