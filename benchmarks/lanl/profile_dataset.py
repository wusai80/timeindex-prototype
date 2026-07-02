"""Profile the LANL auth + red-team benchmark slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from benchmarks.lanl.adapter import load_lanl_auth, load_lanl_redteam
from benchmarks.lanl.schema import default_auth_schema


def profile_auth_dataset(
    auth_path: str | Path,
    *,
    redteam_path: str | Path | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Build a lightweight LANL auth dataset profile."""

    schema = default_auth_schema(
        source_file=str(auth_path),
        redteam_file="" if redteam_path is None else str(redteam_path),
    )
    rows = load_lanl_auth(auth_path, schema=schema, max_rows=max_rows)
    redteam = set() if redteam_path is None else load_lanl_redteam(redteam_path)

    times = [int(row.get(schema.time, 0)) for row in rows]
    src_users = [_normalize(row.get(schema.src_user)) for row in rows]
    src_computers = [_normalize(row.get(schema.src_computer)) for row in rows]
    dst_computers = [_normalize(row.get(schema.dst_computer)) for row in rows]
    events_per_user = _frequency(src_users)
    events_per_computer = _frequency(src_computers + dst_computers)
    positives = 0
    malformed = 0
    for row in rows:
        if not row.get(schema.src_user) or not row.get(schema.src_computer) or not row.get(schema.dst_computer):
            malformed += 1
        if redteam_path is not None:
            key = (
                int(row.get(schema.time, 0)),
                _normalize(row.get(schema.src_user)).split("@", 1)[0],
                _normalize(row.get(schema.src_computer)),
                _normalize(row.get(schema.dst_computer)),
            )
            positives += int(any(
                activity.time == key[0]
                and activity.src_user == key[1]
                and activity.src_computer == key[2]
                and activity.dst_computer == key[3]
                for activity in redteam
            ))

    return {
        "auth_path": str(auth_path),
        "redteam_path": None if redteam_path is None else str(redteam_path),
        "row_count": len(rows),
        "time_range": {
            "min": min(times) if times else None,
            "max": max(times) if times else None,
        },
        "unique_src_users": len({value for value in src_users if value}),
        "unique_src_computers": len({value for value in src_computers if value}),
        "unique_dst_computers": len({value for value in dst_computers if value}),
        "mean_events_per_user": mean(events_per_user.values()) if events_per_user else 0.0,
        "max_events_per_user": max(events_per_user.values()) if events_per_user else 0,
        "mean_events_per_computer": mean(events_per_computer.values()) if events_per_computer else 0.0,
        "max_events_per_computer": max(events_per_computer.values()) if events_per_computer else 0,
        "positive_event_count": positives,
        "positive_rate": (positives / len(rows)) if rows else 0.0,
        "malformed_rows": malformed,
        "detected_fields": dict(schema.columns),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("auth_path")
    parser.add_argument("--redteam-path", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = profile_auth_dataset(
        args.auth_path,
        redteam_path=args.redteam_path,
        max_rows=args.max_rows,
    )
    if args.json:
        print(json.dumps(profile, indent=2, sort_keys=True))
        return 0

    print("LANL Auth Dataset Profile")
    print(f"Auth path: {profile['auth_path']}")
    print(f"Red-team path: {profile['redteam_path']}")
    print(f"Rows: {profile['row_count']}")
    print(f"Time range: {profile['time_range']['min']} -> {profile['time_range']['max']}")
    print(f"Unique src users: {profile['unique_src_users']}")
    print(f"Unique src computers: {profile['unique_src_computers']}")
    print(f"Unique dst computers: {profile['unique_dst_computers']}")
    print(f"Positive events: {profile['positive_event_count']} ({profile['positive_rate']:.6f})")
    print(f"Malformed rows: {profile['malformed_rows']}")
    print("Detected fields:")
    for key, value in sorted(profile["detected_fields"].items()):
        print(f"  - {key}: {value}")
    return 0


def _frequency(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _normalize(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value).strip().lower()


if __name__ == "__main__":
    raise SystemExit(main())

