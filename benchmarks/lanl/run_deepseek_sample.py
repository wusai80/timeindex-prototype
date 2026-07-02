"""Run the DeepSeek temporal reviewer on a deterministic LANL sample set."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.ibm_aml.deepseek_agent import classify_query_with_deepseek
from benchmarks.lanl.verifier import verify_lanl_decision
from timeindex import DecisionIntent
from timeindex.event import EvidenceObject, Event, EventRecord
from timeindex.retrieval import retrieve
from timeindex.sqlite_backend import SqliteTimeIndexBackend


DEFAULT_OUTPUT_DIR = Path("outputs/lanl/deepseek_sample")


def run_lanl_deepseek_sample(
    index_path: str | Path,
    sample_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    budget: int = 8,
    adaptive_budget: int | None = None,
    adaptive_min_events: int = 4,
    adaptive_min_aspects: int = 2,
    positives: int | None = None,
    negatives: int | None = None,
    seed: int = 0,
    max_retries: int = 3,
    domain: str = "lanl",
    verifier_mode: str = "none",
) -> dict[str, Any]:
    """Run DeepSeek on a prepared LANL query set."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    progress_path = output_path / "progress.json"
    results_path = output_path / "results.json"

    query_entries = _load_query_entries(sample_path, positives=positives, negatives=negatives, seed=seed)
    index = SqliteTimeIndexBackend.open(index_path)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    retrieval_latencies: list[float] = []
    llm_latencies: list[float] = []
    effective_budgets: list[float] = []

    def write_progress(stage: str) -> None:
        payload = {
            "stage": stage,
            "index_path": str(index_path),
            "sample_path": str(sample_path),
            "budget": int(budget),
            "adaptive_budget": None if adaptive_budget is None else int(adaptive_budget),
            "adaptive_min_events": int(adaptive_min_events),
            "adaptive_min_aspects": int(adaptive_min_aspects),
            "positive_limit": None if positives is None else int(positives),
            "negative_limit": None if negatives is None else int(negatives),
            "seed": int(seed),
            "domain": str(domain),
            "verifier_mode": str(verifier_mode),
            "processed_queries": len(results),
            "total_queries": len(query_entries),
            "errors": errors,
            "summary": _classification_summary(results),
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
        for position, entry in enumerate(query_entries, start=1):
            query_id = str(entry["event_id"])
            query_record = index.get_event(query_id)
            if query_record is None:
                errors.append({"query_event_id": query_id, "error": "missing query record"})
                write_progress("running")
                continue

            intent = _build_query_intent(query_record.event)

            retrieval_started = time.perf_counter()
            initial_retrieval = _collect_retrieval(index, query_id, intent, budget)
            effective_budget = int(budget)
            if _should_expand_budget(
                initial_retrieval["event_ids"],
                initial_retrieval["aspects"],
                adaptive_budget=adaptive_budget,
                base_budget=budget,
                adaptive_min_events=adaptive_min_events,
                adaptive_min_aspects=adaptive_min_aspects,
            ):
                expanded_retrieval = _collect_retrieval(index, query_id, intent, int(adaptive_budget or budget))
                if len(expanded_retrieval["event_ids"]) >= len(initial_retrieval["event_ids"]):
                    initial_retrieval = expanded_retrieval
                    effective_budget = int(adaptive_budget or budget)
            retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000.0
            retrieval_latencies.append(retrieval_latency_ms)
            effective_budgets.append(float(effective_budget))

            decision = None
            last_error = None
            llm_latency_ms = 0.0
            for attempt in range(max(1, int(max_retries))):
                try:
                    llm_started = time.perf_counter()
                    decision = classify_query_with_deepseek(
                        query_record.event,
                        list(initial_retrieval["events"]),
                        list(initial_retrieval["aspects"]),
                        list(initial_retrieval["objects"]),
                        domain=domain,
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

            actual_positive = _is_positive_label(query_record.event.label)
            verification = verify_lanl_decision(
                query_record.event,
                decision,
                list(initial_retrieval["object_payloads"]),
                mode=verifier_mode if str(domain).strip().lower() == "lanl" else "none",
            )

            row = decision.to_dict()
            row.update(
                {
                    "llm_predicted_positive": bool(decision.predicted_positive),
                    "llm_confidence": float(decision.confidence),
                    "llm_rationale": str(decision.rationale),
                    "predicted_positive": bool(verification.predicted_positive),
                    "confidence": float(verification.confidence),
                    "rationale": str(verification.rationale),
                    "verifier_mode": str(verification.verifier_mode),
                    "verifier_applied": bool(verification.verifier_applied),
                    "verifier_overrode": bool(verification.verifier_overrode),
                    "verifier_reason": str(verification.verifier_reason),
                    "verifier_features": dict(verification.verifier_features),
                    "query_time": int(query_record.event.time),
                    "query_attrs": dict(query_record.event.attrs),
                    "query_metadata": {
                        key: value
                        for key, value in entry.items()
                        if key not in {"event_id", "label"}
                    },
                    "actual_positive": actual_positive,
                    "retrieval_budget": int(budget),
                    "effective_retrieval_budget": int(effective_budget),
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "llm_latency_ms": llm_latency_ms,
                    "correct": bool(verification.predicted_positive) == actual_positive,
                    "retrieved_object_count": len(initial_retrieval["objects"]),
                    "retrieved_object_types": list(initial_retrieval["object_types"]),
                    "retrieved_object_ids": list(initial_retrieval["object_ids"]),
                    "retrieved_objects": list(initial_retrieval["object_payloads"]),
                    "query_frontier_stats": dict(initial_retrieval["frontier_stats"]),
                }
            )
            results.append(row)
            print(
                f"[{position}/{len(query_entries)}] {query_id} label={query_record.event.label} "
                f"predicted={verification.predicted_positive} conf={verification.confidence:.2f} "
                f"retrieved={len(decision.retrieved_event_ids)} support={len(decision.supporting_event_ids)}",
                flush=True,
            )
            write_progress("running")

        summary = {
            "index_path": str(index_path),
            "sample_path": str(sample_path),
            "budget": int(budget),
            "adaptive_budget": None if adaptive_budget is None else int(adaptive_budget),
            "adaptive_min_events": int(adaptive_min_events),
            "adaptive_min_aspects": int(adaptive_min_aspects),
            "positive_limit": None if positives is None else int(positives),
            "negative_limit": None if negatives is None else int(negatives),
            "seed": int(seed),
            "domain": str(domain),
            "verifier_mode": str(verifier_mode),
            "summary": _classification_summary(results),
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
    parser.add_argument("sample_path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--adaptive-budget", type=int, default=None)
    parser.add_argument("--adaptive-min-events", type=int, default=4)
    parser.add_argument("--adaptive-min-aspects", type=int, default=2)
    parser.add_argument("--positives", type=int, default=None)
    parser.add_argument("--negatives", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--domain", default="lanl")
    parser.add_argument("--verifier-mode", default="none")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_lanl_deepseek_sample(
        args.index_path,
        args.sample_path,
        output_dir=args.output_dir,
        budget=args.budget,
        adaptive_budget=args.adaptive_budget,
        adaptive_min_events=args.adaptive_min_events,
        adaptive_min_aspects=args.adaptive_min_aspects,
        positives=args.positives,
        negatives=args.negatives,
        seed=args.seed,
        max_retries=args.max_retries,
        domain=args.domain,
        verifier_mode=args.verifier_mode,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_query_entries(
    sample_path: str | Path,
    *,
    positives: int | None,
    negatives: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    payload = json.loads(Path(sample_path).read_text(encoding="utf-8"))
    positive_rows = [dict(item, label="1") for item in payload.get("positives", [])]
    negative_rows = [dict(item, label="0") for item in payload.get("negatives", [])]
    rng = random.Random(int(seed))
    if positives is not None and len(positive_rows) > int(positives):
        positive_rows = rng.sample(positive_rows, int(positives))
        positive_rows.sort(key=lambda item: str(item.get("event_id", "")))
    if negatives is not None and len(negative_rows) > int(negatives):
        negative_rows = rng.sample(negative_rows, int(negatives))
        negative_rows.sort(key=lambda item: str(item.get("event_id", "")))
    return negative_rows + positive_rows


def _collect_retrieval(
    index: SqliteTimeIndexBackend,
    query_id: str,
    intent: DecisionIntent,
    budget: int,
) -> dict[str, Any]:
    evidence_objects = retrieve(index, query_id, intent, budget)
    retrieved_ids: list[str] = []
    retrieved_aspects: set[str] = set()
    retrieved_events: list[Event] = []
    normalized_objects: list[EvidenceObject] = []
    object_types: list[str] = []
    object_ids: list[str] = []
    object_payloads: list[dict[str, Any]] = []
    for evidence in evidence_objects:
        retrieved_aspects.update(str(aspect) for aspect in getattr(evidence, "aspects", ()))
        for event_id in getattr(evidence, "event_ids", ()):
            event_id_text = str(event_id)
            if event_id_text == query_id or event_id_text in retrieved_ids:
                continue
            retrieved_ids.append(event_id_text)
    limited_ids = retrieved_ids[:budget]
    allowed_id_set = set(limited_ids)
    for evidence in evidence_objects:
        object_id = str(getattr(evidence, "object_id", ""))
        normalized_event_ids = [
            event_id_text
            for event_id_text in (str(event_id) for event_id in getattr(evidence, "event_ids", ()))
            if event_id_text in allowed_id_set and event_id_text != query_id
        ]
        aspects = set(str(aspect) for aspect in getattr(evidence, "aspects", ()) if str(aspect))
        summary = str(getattr(evidence, "summary", "")).strip()
        cost = float(getattr(evidence, "cost", 0.0))
        normalized_objects.append(
            EvidenceObject(
                object_id=object_id,
                event_ids=normalized_event_ids,
                aspects=aspects,
                summary=summary,
                cost=cost,
            )
        )
        object_type = _evidence_type(evidence)
        object_types.append(object_type)
        object_ids.append(object_id)
        object_payloads.append(
            {
                "object_id": object_id,
                "type": object_type,
                "event_ids": list(normalized_event_ids),
                "aspects": sorted(aspects),
                "cost": cost,
                "summary": summary,
            }
        )
    for event_id in limited_ids:
        record = index.get_event(event_id)
        if record is not None:
            retrieved_events.append(record.event)
    frontier_stats = {
        "ordinary_incoming_links": len(index.edge_store.incoming(query_id)),
        "skip_incoming_links": len(index.skip_link_store.incoming(query_id)),
        "chain_summaries": len(index.chain_store.get_for_tail(query_id)),
    }
    return {
        "event_ids": limited_ids,
        "aspects": sorted(retrieved_aspects),
        "events": retrieved_events,
        "objects": normalized_objects,
        "object_types": object_types,
        "object_ids": object_ids,
        "object_payloads": object_payloads,
        "frontier_stats": frontier_stats,
    }


def _evidence_type(evidence: EvidenceObject) -> str:
    object_id = str(getattr(evidence, "object_id", ""))
    summary = str(getattr(evidence, "summary", ""))
    if object_id.startswith("skip:"):
        return "skip"
    if object_id.startswith("ordinary:"):
        return "event"
    if "chain" in summary.lower():
        return "chain"
    return "event"


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


def _build_query_intent(event: Event) -> DecisionIntent:
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


def _classification_summary(results: list[dict[str, Any]]) -> dict[str, float]:
    tp = sum(1 for row in results if row.get("actual_positive") and row.get("predicted_positive"))
    tn = sum(1 for row in results if not row.get("actual_positive") and not row.get("predicted_positive"))
    fp = sum(1 for row in results if not row.get("actual_positive") and row.get("predicted_positive"))
    fn = sum(1 for row in results if row.get("actual_positive") and not row.get("predicted_positive"))
    total = len(results)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "count": float(total),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": float(tp),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
    }


def _label_summary(results: list[dict[str, Any]], *, positive: bool) -> dict[str, float]:
    matching = [item for item in results if bool(item.get("actual_positive")) == positive]
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


def _is_positive_label(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "laundering", "suspicious", "fraud"}


def _mean(values: Any) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


if __name__ == "__main__":
    raise SystemExit(main())
