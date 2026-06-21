"""Run IBM AML baseline and TimeIndex ablation variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmarks.baselines import (
    chain_only_retrieval,
    nearest_neighbor_retrieval,
    recent_window_retrieval,
    same_entity_window_retrieval,
)
from benchmarks.ibm_aml.ablation_configs import AblationVariant, build_variant_config, default_variants
from benchmarks.ibm_aml.run_timeindex import build_gold_supporting_evidence, load_ibm_aml_events
from benchmarks.metrics import (
    context_efficiency,
    evidence_f1_at_budget,
    evidence_precision_at_budget,
    evidence_recall_at_budget,
    mean_latency_ms,
)
from timeindex.construction import TimeIndex
from timeindex.event import DecisionIntent, EvidenceObject, Event
from timeindex.retrieval import retrieve


def run_ablations(
    path: str | Path,
    output_dir: str | Path = "outputs/ibm_aml/ablations",
    budgets: tuple[int, ...] = (3, 5, 10, 20),
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Run the requested ablation suite and save per-variant outputs."""

    del max_rows
    events = load_ibm_aml_events(path)
    events.sort(key=lambda event: (_time_value(event.time), event.event_id))
    gold = build_gold_supporting_evidence(events)
    queries = [event for event in events if _is_positive_label(event.label)]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {"variants": {}, "query_count": len(queries), "event_count": len(events)}

    for variant in default_variants():
        variant_budgets = tuple(budget for budget in budgets if budget in variant.budgets)
        rows = _run_variant(variant, events, queries, gold, variant_budgets)
        _write_jsonl(output_path / f"{variant.name}.jsonl", rows)
        aggregate_rows.extend(_aggregate_rows(variant.name, rows, variant_budgets))
        summaries["variants"][variant.name] = {
            "budgets": list(variant_budgets),
            "rows": len(rows),
        }

    _write_csv(output_path / "aggregate.csv", aggregate_rows)
    summary_path = output_path / "ablation_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    return summaries


def _run_variant(
    variant: AblationVariant,
    events: list[Event],
    queries: list[Event],
    gold: dict[str, list[str]],
    budgets: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = _build_index(events, variant.name) if (variant.use_timeindex or variant.mode == "chain_only") else None
    event_lookup = {event.event_id: event for event in events}

    for query in queries:
        for budget in budgets:
            started = perf_counter()
            if variant.mode == "recent_window":
                retrieved = recent_window_retrieval(events, query, budget=budget, window=max(budget, 10))
            elif variant.mode == "same_entity_window":
                retrieved = same_entity_window_retrieval(events, query, budget=budget, window=max(budget, 10))
            elif variant.mode == "nearest_neighbor":
                retrieved = nearest_neighbor_retrieval(events, query, budget=budget)
            elif variant.mode == "chain_only":
                retrieved = chain_only_retrieval(index, query.event_id, budget=budget)
            else:
                retrieved = retrieve(
                    index,
                    query.event_id,
                    DecisionIntent(aspects=_query_aspects(query), name="ibm_aml"),
                    budget,
                )
            latency_ms = (perf_counter() - started) * 1000.0
            normalized = _normalize_objects(retrieved, query, event_lookup)
            rows.append(
                {
                    "variant": variant.name,
                    "budget": budget,
                    "query_event_id": query.event_id,
                    "query_label": query.label,
                    "retrieved_event_ids": [event_id for item in normalized for event_id in item["event_ids"]],
                    "retrieved_object_types": [item["type"] for item in normalized],
                    "retrieved_aspects": sorted({aspect for item in normalized for aspect in item["aspects"]}),
                    "gold_event_ids": sorted(gold.get(query.event_id, [])),
                    "latency_ms": latency_ms,
                    "budget_used": len(normalized),
                }
            )
    return rows


def _build_index(events: list[Event], variant_name: str) -> TimeIndex:
    config = build_variant_config(variant_name)
    index = TimeIndex(config)
    if not config.construction.enable_bridge_score:
        index.skip_candidate_index.add_chain_anchor = lambda *args, **kwargs: None  # type: ignore[method-assign]
    for event in events:
        index.insert(event)
    return index


def _normalize_objects(
    objects: list[Any],
    query: Event,
    event_lookup: dict[str, Event],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, ...]] = set()
    query_time = _time_value(query.time)

    for obj in objects:
        item = _as_evidence_dict(obj)
        event_ids = [
            event_id
            for event_id in item["event_ids"]
            if event_id in event_lookup and _time_value(event_lookup[event_id].time) < query_time
        ]
        signature = tuple(sorted(event_ids))
        if not event_ids or signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        item["event_ids"] = event_ids
        normalized.append(item)
    return normalized


def _as_evidence_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, EvidenceObject):
        return {
            "event_ids": [str(event_id) for event_id in obj.event_ids],
            "type": _object_type(obj.object_id),
            "summary": obj.summary,
            "aspects": sorted(str(aspect) for aspect in obj.aspects),
            "score": float(obj.cost),
        }
    if isinstance(obj, dict):
        return {
            "event_ids": [str(event_id) for event_id in obj.get("event_ids", [])],
            "type": str(obj.get("type", "event")),
            "summary": str(obj.get("summary", "")),
            "aspects": sorted(str(aspect) for aspect in obj.get("aspects", [])),
            "score": float(obj.get("score", 0.0)),
        }

    event_ids = list(getattr(obj, "event_ids", []))
    if not event_ids and hasattr(obj, "event_id"):
        event_ids = [str(obj.event_id)]
    return {
        "event_ids": [str(event_id) for event_id in event_ids],
        "type": getattr(obj, "type", obj.__class__.__name__.lower()),
        "summary": getattr(obj, "summary", ""),
        "aspects": sorted(str(aspect) for aspect in getattr(obj, "aspects", set())),
        "score": float(getattr(obj, "score", 0.0)),
    }


def _aggregate_rows(variant: str, rows: list[dict[str, Any]], budgets: tuple[int, ...]) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    for budget in budgets:
        bucket = [row for row in rows if row["budget"] == budget]
        aggregate.append(
            {
                "variant": variant,
                "budget": budget,
                "recall": _mean([evidence_recall_at_budget(row["retrieved_event_ids"], row["gold_event_ids"]) for row in bucket]),
                "precision": _mean([evidence_precision_at_budget(row["retrieved_event_ids"], row["gold_event_ids"]) for row in bucket]),
                "f1": _mean([evidence_f1_at_budget(row["retrieved_event_ids"], row["gold_event_ids"]) for row in bucket]),
                "latency": mean_latency_ms([float(row["latency_ms"]) for row in bucket]),
                "context_efficiency": _mean(
                    [
                        context_efficiency(
                            row["retrieved_event_ids"],
                            row["gold_event_ids"],
                            row["budget_used"],
                        )
                        for row in bucket
                    ]
                ),
            }
        )
    return aggregate


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _query_aspects(event: Event) -> set[str]:
    aspects: set[str] = set()
    amount = event.attrs.get("amount")
    if isinstance(amount, (int, float)) and float(amount) >= 1_000.0:
        aspects.add("large_transfer")
    if event.attrs.get("beneficiary_account") or event.attrs.get("dst_account"):
        aspects.add("beneficiary_novelty")
    if not aspects:
        aspects.add("generic_evidence")
    return aspects


def _object_type(object_id: str) -> str:
    if object_id.startswith("skip:"):
        return "skip"
    if object_id.startswith("ordinary:"):
        return "ordinary"
    return "chain"


def _is_positive_label(label: Any) -> bool:
    if isinstance(label, bool):
        return label
    text = str(label).strip().lower()
    return text in {"1", "true", "yes", "y", "laundering", "suspicious", "alert", "fraud"}


def _time_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="IBM AML CSV path")
    parser.add_argument("--output-dir", default="outputs/ibm_aml/ablations", help="Directory for ablation artifacts")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional max rows to load")
    args = parser.parse_args()
    summary = run_ablations(args.path, output_dir=args.output_dir, max_rows=args.max_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
