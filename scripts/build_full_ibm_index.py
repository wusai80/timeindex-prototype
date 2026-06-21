"""Build a full IBM AML TimeIndex cache with resumable stages and logs."""

from __future__ import annotations

import argparse
import bz2
import csv
import json
import pickle
import sys
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.ibm_aml.adapter import convert_row_to_event
from timeindex.config import TimeIndexConfig
from timeindex.construction import TimeIndex
from timeindex.sqlite_backend import export_sqlite_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-path",
        default="data/ibm_aml/raw/trans_3000p2_list.txt.bz2",
        help="Path to the raw IBM AML bz2 archive.",
    )
    parser.add_argument(
        "--work-dir",
        default="outputs/ibm_aml/full_build",
        help="Directory for normalized data, sorted data, logs, and final cache.",
    )
    parser.add_argument(
        "--active-history-size",
        type=int,
        default=100_000,
        help="Bounded active history retained in the final in-memory index.",
    )
    parser.add_argument(
        "--posting-list-size",
        type=int,
        default=256,
        help="Posting list bound per lookup key.",
    )
    parser.add_argument(
        "--backend",
        choices=("pickle", "sqlite", "both"),
        default="both",
        help="Which cache format(s) to emit after construction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw_path)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = work_dir / "trans_3000p2_full_normalized.csv"
    sorted_path = work_dir / "trans_3000p2_full_sorted.csv"
    pickle_cache_path = work_dir / "timeindex_full.pkl"
    sqlite_cache_path = work_dir / "timeindex_full.sqlite"
    metadata_path = work_dir / "timeindex_full.json"
    progress_path = work_dir / "progress.json"

    progress = _load_json(progress_path)
    progress.setdefault("stages", {})

    if not normalized_path.exists():
        _normalize_raw_archive(raw_path, normalized_path, progress_path)
    progress["stages"]["normalized"] = str(normalized_path)
    _save_json(progress_path, progress)

    if not sorted_path.exists():
        _sort_normalized_csv(normalized_path, sorted_path, progress_path)
    progress["stages"]["sorted"] = str(sorted_path)
    _save_json(progress_path, progress)

    should_have_pickle = args.backend in {"pickle", "both"}
    should_have_sqlite = args.backend in {"sqlite", "both"}
    outputs_missing = (
        (should_have_pickle and not pickle_cache_path.exists())
        or (should_have_sqlite and not sqlite_cache_path.exists())
        or not metadata_path.exists()
    )
    if outputs_missing:
        metadata = _build_timeindex(sorted_path, pickle_cache_path, sqlite_cache_path, progress_path, args)
        _save_json(metadata_path, metadata)
    else:
        metadata = _load_json(metadata_path)

    print(json.dumps(metadata, indent=2, sort_keys=True))


def _normalize_raw_archive(raw_path: Path, normalized_path: Path, progress_path: Path) -> None:
    started = perf_counter()
    rows_written = 0
    with bz2.open(raw_path, "rt", encoding="utf-8", errors="replace") as src, normalized_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as dst:
        reader = csv.reader(src)
        writer = csv.DictWriter(
            dst,
            fieldnames=[
                "transaction_id",
                "timestamp",
                "src_bank",
                "src_account",
                "dst_bank",
                "dst_account",
                "amount",
                "currency",
                "amount_received",
                "receiving_currency",
                "payment_format",
                "is_laundering",
            ],
        )
        writer.writeheader()
        next(reader)
        for row_index, row in enumerate(reader):
            if len(row) < 11:
                continue
            writer.writerow(
                {
                    "transaction_id": f"tx-{row_index:09d}",
                    "timestamp": row[0],
                    "src_bank": row[1],
                    "src_account": row[2],
                    "dst_bank": row[3],
                    "dst_account": row[4],
                    "amount": row[7] or row[5],
                    "currency": row[8] or row[6],
                    "amount_received": row[5],
                    "receiving_currency": row[6],
                    "payment_format": row[9],
                    "is_laundering": row[10],
                }
            )
            rows_written += 1
            if rows_written % 500_000 == 0:
                _save_json(
                    progress_path,
                    {
                        "stage": "normalize",
                        "rows_written": rows_written,
                        "elapsed_seconds": perf_counter() - started,
                    },
                )
    _save_json(
        progress_path,
        {
            "stage": "normalize_complete",
            "rows_written": rows_written,
            "elapsed_seconds": perf_counter() - started,
        },
    )


def _sort_normalized_csv(normalized_path: Path, sorted_path: Path, progress_path: Path) -> None:
    import subprocess

    started = perf_counter()
    with normalized_path.open("r", encoding="utf-8", newline="") as src:
        header = src.readline()
    data_path = normalized_path.with_suffix(".body.csv")
    if not data_path.exists():
        with normalized_path.open("r", encoding="utf-8", newline="") as src, data_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as dst:
            next(src)
            for line in src:
                dst.write(line)

    sorted_body_path = sorted_path.with_suffix(".body.csv")
    command = [
        "sort",
        "-t,",
        "-k2,2",
        "-T",
        str(sorted_path.parent),
        "-S",
        "50%",
        str(data_path),
    ]
    with sorted_body_path.open("w", encoding="utf-8", newline="") as dst:
        subprocess.run(command, check=True, stdout=dst)

    with sorted_path.open("w", encoding="utf-8", newline="") as final_dst:
        final_dst.write(header)
        with sorted_body_path.open("r", encoding="utf-8", newline="") as sorted_src:
            for line in sorted_src:
                final_dst.write(line)

    _save_json(
        progress_path,
        {
            "stage": "sort_complete",
            "elapsed_seconds": perf_counter() - started,
            "sorted_path": str(sorted_path),
        },
    )


def _build_timeindex(
    sorted_path: Path,
    pickle_cache_path: Path,
    sqlite_cache_path: Path,
    progress_path: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    started = perf_counter()
    config = TimeIndexConfig()
    config.stores.active_history_size = int(args.active_history_size)
    config.stores.posting_list_size = int(args.posting_list_size)
    config.stores.ordinary_fan_in = 5
    config.stores.skip_fan_in = 3
    config.stores.chain_summaries_per_family = 5
    config.scoring.time_decay = 1_000_000.0

    index = TimeIndex(config)
    schema = {
        "dataset_name": "ibm_aml",
        "source_file": str(sorted_path),
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

    indexed_events = 0
    with sorted_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            row["__row_index__"] = row_index
            record = convert_row_to_event(row, schema)
            index.insert(record.event)
            indexed_events += 1
            if indexed_events % 500_000 == 0:
                _save_json(
                    progress_path,
                    {
                        "stage": "indexing",
                        "indexed_events": indexed_events,
                        "elapsed_seconds": perf_counter() - started,
                    },
                )

    index.key_directory._postings = dict(index.key_directory._postings)
    metadata = {
        "sorted_csv": str(sorted_path),
        "indexed_events": indexed_events,
        "active_history_size": config.stores.active_history_size,
        "posting_list_size": config.stores.posting_list_size,
        "elapsed_seconds": perf_counter() - started,
        "backend": args.backend,
    }
    if args.backend in {"pickle", "both"}:
        with pickle_cache_path.open("wb") as handle:
            pickle.dump(index, handle)
        metadata["pickle_cache_path"] = str(pickle_cache_path)
    if args.backend in {"sqlite", "both"}:
        export_sqlite_backend(index, sqlite_cache_path, overwrite=True)
        metadata["sqlite_cache_path"] = str(sqlite_cache_path)
    _save_json(progress_path, {"stage": "complete", **metadata})
    return metadata


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
