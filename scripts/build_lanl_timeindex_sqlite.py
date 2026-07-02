"""Build a persistent SQLite TimeIndex cache for a LANL auth slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.lanl.adapter import stream_events
from benchmarks.lanl.schema import default_auth_schema
from timeindex.config import TimeIndexConfig
from timeindex.construction import TimeIndex
from timeindex.sqlite_backend import SqliteTimeIndexWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("auth_path", help="LANL auth slice path.")
    parser.add_argument("redteam_path", help="LANL red-team slice path.")
    parser.add_argument(
        "--sqlite-path",
        default="outputs/lanl/cache/timeindex.sqlite",
        help="Destination SQLite cache path.",
    )
    parser.add_argument(
        "--progress-path",
        default="outputs/lanl/cache/timeindex.progress.json",
        help="Progress JSON path.",
    )
    parser.add_argument("--active-history-size", type=int, default=100000)
    parser.add_argument("--posting-list-size", type=int, default=256)
    parser.add_argument("--ordinary-fan-in", type=int, default=5)
    parser.add_argument("--skip-fan-in", type=int, default=3)
    parser.add_argument("--chain-summaries-per-family", type=int, default=5)
    parser.add_argument("--time-decay", type=float, default=1_000_000.0)
    parser.add_argument("--flush-every", type=int, default=10000)
    parser.add_argument(
        "--row-index-offset",
        type=int,
        default=0,
        help="Optional global row index offset used to preserve event ids for extracted tail slices.",
    )
    parser.add_argument(
        "--expire-batch-size",
        type=int,
        default=1000,
        help="Expire aged events in batches once overflow reaches this size. Zero means expire every event.",
    )
    parser.add_argument("--max-events", type=int, default=0, help="Optional event cap. Zero means all rows.")
    parser.add_argument(
        "--disable-expiration",
        action="store_true",
        help="Keep the full in-memory history. This uses much more memory and is disabled by default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auth_path = Path(args.auth_path)
    redteam_path = Path(args.redteam_path)
    sqlite_path = Path(args.sqlite_path)
    progress_path = Path(args.progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    config = _build_config(args)
    index = TimeIndex(config)
    schema = default_auth_schema(source_file=str(auth_path), redteam_file=str(redteam_path))
    max_events = max(0, int(args.max_events))
    flush_every = max(1, int(args.flush_every))

    indexed_events = 0
    with SqliteTimeIndexWriter(sqlite_path, config=config, overwrite=args.overwrite) as writer:
        # Extracted LANL slice files preserve auth.gz time order, so we can stream
        # directly without materializing and resorting the entire slice.
        for _, source_record in enumerate(
            stream_events(
                auth_path,
                redteam_path,
                schema,
                sort_by_time=False,
                row_index_offset=max(0, int(args.row_index_offset)),
            ),
            start=1,
        ):
            record = index.insert(source_record.event)
            event_id = record.event.event_id
            writer.write_event_snapshot(
                record,
                index.ordinary_links(event_id),
                index.chains(event_id),
                index.skip_links(event_id),
            )
            expired_ids = index.recent_expired_ids()
            if expired_ids:
                writer.expire_event_ids(expired_ids)
            indexed_events += 1
            if indexed_events % flush_every == 0:
                payload = _progress_payload(
                    auth_path=auth_path,
                    redteam_path=redteam_path,
                    sqlite_path=sqlite_path,
                    indexed_events=indexed_events,
                    started=started,
                    stage="indexing",
                )
                writer.write_metadata("build_progress", payload)
                writer.flush()
                _save_json(progress_path, payload)
            if max_events and indexed_events >= max_events:
                break

        payload = _progress_payload(
            auth_path=auth_path,
            redteam_path=redteam_path,
            sqlite_path=sqlite_path,
            indexed_events=indexed_events,
            started=started,
            stage="complete",
        )
        writer.write_metadata("build_summary", payload)
        writer.flush()
        _save_json(progress_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_config(args: argparse.Namespace) -> TimeIndexConfig:
    config = TimeIndexConfig()
    config.stores.active_history_size = int(args.active_history_size)
    config.stores.posting_list_size = int(args.posting_list_size)
    config.stores.ordinary_fan_in = int(args.ordinary_fan_in)
    config.stores.skip_fan_in = int(args.skip_fan_in)
    config.stores.chain_summaries_per_family = int(args.chain_summaries_per_family)
    config.scoring.time_decay = float(args.time_decay)
    config.construction.expire_stale_items = not bool(args.disable_expiration)
    config.construction.expire_batch_size = max(0, int(args.expire_batch_size))
    return config


def _progress_payload(
    *,
    auth_path: Path,
    redteam_path: Path,
    sqlite_path: Path,
    indexed_events: int,
    started: float,
    stage: str,
) -> dict[str, object]:
    elapsed = perf_counter() - started
    return {
        "stage": stage,
        "auth_path": str(auth_path),
        "redteam_path": str(redteam_path),
        "sqlite_path": str(sqlite_path),
        "indexed_events": indexed_events,
        "elapsed_seconds": elapsed,
        "events_per_second": (indexed_events / elapsed) if elapsed > 0.0 else 0.0,
    }


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
