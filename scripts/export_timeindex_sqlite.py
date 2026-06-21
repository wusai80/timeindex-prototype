"""Convert a pickled TimeIndex cache into a SQLite-backed retrieval cache."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from timeindex.sqlite_backend import export_sqlite_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle_path", help="Path to an existing pickled TimeIndex cache.")
    parser.add_argument("sqlite_path", help="Destination SQLite cache path.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the target SQLite file if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pickle_path = Path(args.pickle_path)
    sqlite_path = Path(args.sqlite_path)

    started = perf_counter()
    with pickle_path.open("rb") as handle:
        index = pickle.load(handle)
    load_elapsed = perf_counter() - started

    export_started = perf_counter()
    export_sqlite_backend(index, sqlite_path, overwrite=args.overwrite)
    export_elapsed = perf_counter() - export_started

    print(
        json.dumps(
            {
                "pickle_path": str(pickle_path),
                "sqlite_path": str(sqlite_path),
                "load_seconds": load_elapsed,
                "export_seconds": export_elapsed,
                "total_seconds": load_elapsed + export_elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
