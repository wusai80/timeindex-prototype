"""Record build-progress throughput snapshots at fixed event-count milestones."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("progress_path", help="Path to the live progress JSON file.")
    parser.add_argument(
        "--output-path",
        required=True,
        help="JSONL file that receives milestone snapshots.",
    )
    parser.add_argument(
        "--milestone-step",
        type=int,
        default=500_000,
        help="Record one row each time indexed_events crosses this step.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="Polling interval while the build is running.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    progress_path = Path(args.progress_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    milestone_step = max(1, int(args.milestone_step))
    poll_seconds = max(1.0, float(args.poll_seconds))

    seen_milestones = _load_seen_milestones(output_path)
    next_milestone = milestone_step

    if seen_milestones:
        next_milestone = max(seen_milestones) + milestone_step

    while True:
        payload = _read_progress(progress_path)
        if payload is None:
            time.sleep(poll_seconds)
            continue

        indexed_events = int(payload.get("indexed_events", 0) or 0)
        stage = str(payload.get("stage", ""))

        while indexed_events >= next_milestone:
            snapshot = {
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "milestone_events": int(next_milestone),
                "indexed_events": indexed_events,
                "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0) or 0.0),
                "events_per_second": float(payload.get("events_per_second", 0.0) or 0.0),
                "stage": stage,
                "sqlite_path": payload.get("sqlite_path"),
                "sorted_csv": payload.get("sorted_csv"),
            }
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
            seen_milestones.add(next_milestone)
            next_milestone += milestone_step

        if stage == "complete":
            return 0

        time.sleep(poll_seconds)


def _load_seen_milestones(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    milestones: set[int] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            milestone = row.get("milestone_events")
            if isinstance(milestone, int):
                milestones.add(milestone)
    return milestones


def _read_progress(progress_path: Path) -> dict[str, object] | None:
    if not progress_path.exists():
        return None
    try:
        return json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
