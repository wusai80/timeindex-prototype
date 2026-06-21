"""Run a deterministic stand-in agent over retrieved IBM AML contexts."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from benchmarks.baselines import (
    flow_chain_retrieval,
    nearest_neighbor_retrieval,
    recent_window_retrieval,
    same_entity_window_retrieval,
)
from benchmarks.ibm_aml.fake_agent import classify_query_from_retrieval, summarize_decisions
from benchmarks.ibm_aml.run_timeindex import build_gold_supporting_evidence, load_ibm_aml_events
from timeindex import DecisionIntent
from timeindex.event import Event
from timeindex.retrieval import retrieve
from timeindex.sqlite_backend import SqliteTimeIndexBackend


def run_fake_agent_experiment(
    csv_path: str | Path,
    *,
    output_dir: str | Path = "outputs/ibm_aml/fake_agent",
    retrieval_mode: str = "same_entity_window",
    budget: int = 5,
    query_limit: int = 20,
    window: int = 200,
    index_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the deterministic agent on a limited set of positive queries."""

    events = load_ibm_aml_events(csv_path)
    events.sort(key=lambda event: (_time_value(event.time), event.event_id))
    event_lookup = {event.event_id: event for event in events}
    gold = build_gold_supporting_evidence(events)
    positive_queries = [event for event in events if _is_positive_label(event.label)]
    selected_queries = positive_queries[: max(0, query_limit)]

    index = _load_cached_index(index_cache_path) if retrieval_mode == "timeindex" and index_cache_path else None
    decisions = []
    for query_event in selected_queries:
        retrieved_ids, retrieved_aspects = _retrieve_context(
            retrieval_mode,
            events,
            query_event,
            budget=budget,
            window=window,
            index=index,
        )
        decisions.append(
            classify_query_from_retrieval(
                query_event,
                retrieved_event_ids=retrieved_ids,
                retrieved_aspects=retrieved_aspects,
                event_lookup=event_lookup,
                gold_event_ids=gold.get(query_event.event_id, []),
            )
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    details_path = output_path / f"{retrieval_mode}_details.json"
    summary_path = output_path / f"{retrieval_mode}_summary.json"

    details_path.write_text(
        json.dumps([decision.to_dict() for decision in decisions], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = summarize_decisions(decisions)
    summary.update(
        {
            "retrieval_mode": retrieval_mode,
            "budget": float(budget),
            "query_limit": float(query_limit),
            "queries_available": float(len(positive_queries)),
            "queries_evaluated": float(len(decisions)),
            "details_path": str(details_path),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--output-dir", default="outputs/ibm_aml/fake_agent")
    parser.add_argument(
        "--retrieval-mode",
        default="same_entity_window",
        choices=("same_entity_window", "recent_window", "nearest_neighbor", "flow_chain", "timeindex"),
    )
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--query-limit", type=int, default=20)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--index-cache-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_fake_agent_experiment(
        args.csv_path,
        output_dir=args.output_dir,
        retrieval_mode=args.retrieval_mode,
        budget=args.budget,
        query_limit=args.query_limit,
        window=args.window,
        index_cache_path=args.index_cache_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _retrieve_context(
    retrieval_mode: str,
    events: list[Event],
    query_event: Event,
    *,
    budget: int,
    window: int,
    index: Any,
) -> tuple[list[str], list[str]]:
    if retrieval_mode == "same_entity_window":
        items = same_entity_window_retrieval(events, query_event, budget=budget, window=window)
        return _result_ids(items), []
    if retrieval_mode == "recent_window":
        items = recent_window_retrieval(events, query_event, budget=budget, window=window)
        return _result_ids(items), []
    if retrieval_mode == "nearest_neighbor":
        items = nearest_neighbor_retrieval(events, query_event, budget=budget)
        return _result_ids(items), []
    if retrieval_mode == "flow_chain":
        items = flow_chain_retrieval(events, query_event, budget=budget, max_hops=3)
        return _result_ids(items), []
    if retrieval_mode == "timeindex":
        if index is None:
            raise ValueError("retrieval_mode='timeindex' requires index_cache_path")
        intent = DecisionIntent(aspects={"large_transfer", "beneficiary_novelty", "generic_evidence"}, name="fake_agent")
        evidence_objects = retrieve(index, query_event.event_id, intent, budget)
        event_ids = []
        aspects: set[str] = set()
        for item in evidence_objects:
            aspects.update(getattr(item, "aspects", set()))
            for event_id in getattr(item, "event_ids", []):
                event_id_text = str(event_id)
                if event_id_text != query_event.event_id and event_id_text not in event_ids:
                    event_ids.append(event_id_text)
        return event_ids[:budget], sorted(aspects)
    raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")


def _result_ids(items: list[dict[str, Any]]) -> list[str]:
    event_ids: list[str] = []
    for item in items:
        for event_id in item.get("event_ids", []):
            event_id_text = str(event_id)
            if event_id_text not in event_ids:
                event_ids.append(event_id_text)
    return event_ids


def _load_cached_index(index_cache_path: str | Path | None) -> Any:
    if index_cache_path is None:
        return None
    resolved_path = Path(index_cache_path)
    if resolved_path.suffix.lower() in {".sqlite", ".db"}:
        return SqliteTimeIndexBackend.open(resolved_path)
    with resolved_path.open("rb") as handle:
        return pickle.load(handle)


def _is_positive_label(label: Any) -> bool:
    if isinstance(label, bool):
        return label
    if isinstance(label, (int, float)):
        return float(label) > 0.0
    return str(label or "").strip().lower() in {"1", "true", "yes", "y", "laundering", "suspicious", "fraud"}


def _time_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        digits = "".join(character for character in str(value) if character.isdigit())
        return float(digits) if digits else 0.0


if __name__ == "__main__":
    main()
