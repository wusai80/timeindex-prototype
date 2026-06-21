"""Build a SQLite TimeIndex cache directly from a sorted IBM AML CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.ibm_aml.adapter import convert_row_to_event
from timeindex.config import TimeIndexConfig
from timeindex.construction import TimeIndex
from timeindex.sqlite_backend import SqliteTimeIndexWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sorted-csv",
        default="data/ibm_aml/processed/trans_3000p2_first_5000000_sorted.csv",
        help="Sorted normalized CSV path.",
    )
    parser.add_argument(
        "--sqlite-path",
        default="outputs/ibm_aml/cache/timeindex_first_5000000_streaming.sqlite",
        help="Destination SQLite index path.",
    )
    parser.add_argument(
        "--progress-path",
        default="outputs/ibm_aml/cache/timeindex_first_5000000_streaming.progress.json",
        help="Progress JSON path.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Optional cap on indexed events. Zero means all rows.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10000,
        help="Commit every N indexed events.",
    )
    parser.add_argument("--active-history-size", type=int, default=5_000_000)
    parser.add_argument("--posting-list-size", type=int, default=256)
    parser.add_argument("--ordinary-fan-in", type=int, default=5)
    parser.add_argument("--skip-fan-in", type=int, default=3)
    parser.add_argument("--chain-summaries-per-family", type=int, default=5)
    parser.add_argument("--time-decay", type=float, default=1_000_000.0)
    parser.add_argument("--max-frontier-size", type=int, default=64)
    parser.add_argument("--max-branch-factor", type=int, default=3)
    parser.add_argument("--max-search-expansions", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--skip-competitive-ratio", type=float, default=0.9)
    parser.add_argument("--skip-candidate-pool-factor", type=int, default=3)
    parser.add_argument("--skip-summary-event-limit", type=int, default=4)
    parser.add_argument(
        "--expire-stale-items",
        action="store_true",
        help="Enable expiration during construction. Disabled by default to match prior 5M builds.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sorted_csv = Path(args.sorted_csv)
    sqlite_path = Path(args.sqlite_path)
    progress_path = Path(args.progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    config = _build_config(args)
    schema = _build_schema(sorted_csv)

    started = perf_counter()
    index = TimeIndex(config)
    indexed_events = 0
    max_events = max(0, int(args.max_events))
    flush_every = max(1, int(args.flush_every))

    with SqliteTimeIndexWriter(sqlite_path, config=config, overwrite=args.overwrite) as writer:
        with sorted_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader):
                if max_events and indexed_events >= max_events:
                    break
                row["__row_index__"] = row_index
                converted = convert_row_to_event(row, schema)
                record = index.insert(converted.event)
                event_id = record.event.event_id
                writer.write_event_snapshot(
                    record,
                    index.ordinary_links(event_id),
                    index.chains(event_id),
                    index.skip_links(event_id),
                )
                indexed_events += 1

                if indexed_events % flush_every == 0:
                    writer.write_metadata(
                        "build_progress",
                        _progress_payload(
                            sqlite_path,
                            sorted_csv,
                            indexed_events,
                            started,
                            stage="indexing",
                        ),
                    )
                    writer.flush()
                    _save_json(
                        progress_path,
                        _progress_payload(
                            sqlite_path,
                            sorted_csv,
                            indexed_events,
                            started,
                            stage="indexing",
                        ),
                    )

        summary = _progress_payload(
            sqlite_path,
            sorted_csv,
            indexed_events,
            started,
            stage="complete",
        )
        writer.write_metadata("build_summary", summary)
        writer.flush()
        _save_json(progress_path, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))


def _build_config(args: argparse.Namespace) -> TimeIndexConfig:
    config = TimeIndexConfig()
    config.stores.active_history_size = int(args.active_history_size)
    config.stores.posting_list_size = int(args.posting_list_size)
    config.stores.ordinary_fan_in = int(args.ordinary_fan_in)
    config.stores.skip_fan_in = int(args.skip_fan_in)
    config.stores.chain_summaries_per_family = int(args.chain_summaries_per_family)
    config.scoring.time_decay = float(args.time_decay)
    config.retrieval.max_frontier_size = int(args.max_frontier_size)
    config.retrieval.max_branch_factor = int(args.max_branch_factor)
    config.retrieval.max_search_expansions = int(args.max_search_expansions)
    config.retrieval.max_depth = int(args.max_depth)
    config.retrieval.skip_competitive_ratio = float(args.skip_competitive_ratio)
    config.construction.skip_candidate_pool_factor = int(args.skip_candidate_pool_factor)
    config.construction.skip_summary_event_limit = int(args.skip_summary_event_limit)
    config.construction.expire_stale_items = bool(args.expire_stale_items)
    return config


def _build_schema(sorted_csv: Path) -> dict[str, object]:
    return {
        "dataset_name": "ibm_aml",
        "source_file": str(sorted_csv),
        "transaction_id": "transaction_id",
        "timestamp": "timestamp",
        "src_account": "src_account",
        "dst_account": "dst_account",
        "amount": "amount",
        "currency": "currency",
        "src_bank": "src_bank",
        "dst_bank": "dst_bank",
        "payment_format": "payment_format",
        "label": "is_laundering",
        "type": "payment_format",
        "columns": {
            "transaction_id": "transaction_id",
            "timestamp": "timestamp",
            "src_account": "src_account",
            "dst_account": "dst_account",
            "amount": "amount",
            "currency": "currency",
            "src_bank": "src_bank",
            "dst_bank": "dst_bank",
            "payment_format": "payment_format",
            "label": "is_laundering",
            "type": "payment_format",
        },
    }


def _progress_payload(
    sqlite_path: Path,
    sorted_csv: Path,
    indexed_events: int,
    started: float,
    *,
    stage: str,
) -> dict[str, object]:
    elapsed = perf_counter() - started
    return {
        "stage": stage,
        "sqlite_path": str(sqlite_path),
        "sorted_csv": str(sorted_csv),
        "indexed_events": indexed_events,
        "elapsed_seconds": elapsed,
        "events_per_second": (indexed_events / elapsed) if elapsed > 0 else 0.0,
    }


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
