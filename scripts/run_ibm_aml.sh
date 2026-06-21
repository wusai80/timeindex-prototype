#!/usr/bin/env bash

set -euo pipefail

CSV_PATH="${1:-data/ibm_aml/sample.csv}"
OUTPUT_DIR="${2:-outputs/ibm_aml}"

python benchmarks/ibm_aml/profile_dataset.py --path "${CSV_PATH}"
python benchmarks/ibm_aml/run_timeindex.py "${CSV_PATH}" --output-dir "${OUTPUT_DIR}"
python benchmarks/ibm_aml/run_ablations.py --path "${CSV_PATH}" --output-dir "${OUTPUT_DIR}/ablations"

AGGREGATE_PATH="${OUTPUT_DIR}/ablations/aggregate.csv"
PLOT_DIR="${OUTPUT_DIR}/plots"

AGGREGATE_PATH="${AGGREGATE_PATH}" PLOT_DIR="${PLOT_DIR}" python - <<'PY'
import csv
import os
from pathlib import Path

from benchmarks.plots import (
    plot_ablation_bars,
    plot_context_efficiency,
    plot_latency_vs_budget,
    plot_precision_at_budget,
    plot_recall_at_budget,
)

aggregate_path = Path(os.environ["AGGREGATE_PATH"])
if not aggregate_path.exists():
    raise SystemExit(f"missing aggregate file: {aggregate_path}")

with aggregate_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

full_rows = [row for row in rows if row["variant"] == "timeindex_full"]
plot_dir = Path(os.environ["PLOT_DIR"])
plot_recall_at_budget(full_rows, plot_dir / "recall")
plot_precision_at_budget(full_rows, plot_dir / "precision")
plot_context_efficiency(full_rows, plot_dir / "context_efficiency")
plot_latency_vs_budget(full_rows, plot_dir / "latency")
plot_ablation_bars(
    [
        {"name": row["variant"].replace("_", " "), "score": float(row["recall"])}
        for row in rows
        if row["budget"] == "5"
    ],
    plot_dir / "ablation_recall_b5",
)
PY
