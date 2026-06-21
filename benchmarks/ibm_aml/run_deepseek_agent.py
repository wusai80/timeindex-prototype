"""Run the DeepSeek fraud review agent over retrieved IBM AML contexts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.ibm_aml.deepseek_agent import (
    DEFAULT_KEY_PATH,
    DEFAULT_MODEL,
    classify_query_with_deepseek,
    summarize_decisions,
)
from benchmarks.ibm_aml.run_fake_agent import _is_positive_label, _load_cached_index, _retrieve_context, _time_value
from benchmarks.ibm_aml.run_timeindex import build_gold_supporting_evidence, load_ibm_aml_events
from timeindex.event import Event


def run_deepseek_agent_experiment(
    csv_path: str | Path,
    *,
    output_dir: str | Path = "outputs/ibm_aml/deepseek_agent",
    retrieval_mode: str = "timeindex",
    budget: int = 5,
    query_limit: int = 20,
    window: int = 200,
    index_cache_path: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    key_path: str | Path = DEFAULT_KEY_PATH,
) -> dict[str, Any]:
    """Run the DeepSeek-backed agent on a limited set of positive queries."""

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
        retrieved_events = [event_lookup[event_id] for event_id in retrieved_ids if event_id in event_lookup]
        decision = classify_query_with_deepseek(
            query_event,
            retrieved_events,
            retrieved_aspects,
            model=model,
            key_path=key_path,
        )
        decision.raw_response["gold_event_ids"] = sorted(str(event_id) for event_id in gold.get(query_event.event_id, []))
        decisions.append(decision)

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
            "model": model,
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
    parser.add_argument("--output-dir", default="outputs/ibm_aml/deepseek_agent")
    parser.add_argument(
        "--retrieval-mode",
        default="timeindex",
        choices=("same_entity_window", "recent_window", "nearest_neighbor", "flow_chain", "timeindex"),
    )
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--query-limit", type=int, default=20)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--index-cache-path", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--key-path", default=str(DEFAULT_KEY_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_deepseek_agent_experiment(
        args.csv_path,
        output_dir=args.output_dir,
        retrieval_mode=args.retrieval_mode,
        budget=args.budget,
        query_limit=args.query_limit,
        window=args.window,
        index_cache_path=args.index_cache_path,
        model=args.model,
        key_path=args.key_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
