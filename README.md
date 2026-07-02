# TimeIndex Prototype

TimeIndex is a prototype temporal evidence index for streaming decision support. The goal is not generic vector search or task-specific prompt tuning alone. The goal is to retrieve compact, causally valid historical evidence for a query event under a small context budget.

## Core Idea

TimeIndex is built around a few simple principles:

1. Events are represented as structured records with attributes, context, and optional text.
2. Ordinary links preserve local temporal continuity between related events.
3. Chain summaries compress longer multi-step histories.
4. Skip links provide direct access to distant but valuable evidence anchors.
5. Retrieval explores both ordinary and skip frontiers under a fixed budget.
6. Downstream agents or rules consume evidence objects, not the full raw history.

The main project target is:

1. Given a query event
2. Retrieve compact historical supporting evidence
3. Under a small budget
4. While preserving temporal causality
5. With low update cost and low retrieval latency

## Repository Layout

Top-level structure:

```text
timeindex/
tests/
benchmarks/
scripts/
data/
outputs/
README.md
agent.md
TODO.txt
change.txt
pyproject.toml
TimeIndex.pdf
```

Main directories:

1. `timeindex/`
   Core package: event schema, extraction, scoring, stores, construction, retrieval, SQLite backend, synthetic generators, evaluation.
2. `tests/`
   Package-level unit tests.
3. `benchmarks/`
   Dataset-specific adapters, evidence builders, runners, metrics, plots, baselines, and agent experiments.
4. `scripts/`
   Helpers for building SQLite indexes and monitoring long runs.
5. `data/`
   Local datasets only. Not tracked in Git.
6. `outputs/`
   Local experiment outputs, prompt dumps, reports, and cached indexes. Not tracked in Git.

Important files:

1. `README.md`
   Main repository overview.
2. `agent.md`
   Working guide for future coding agents.
3. `TODO.txt`
   Current backlog and next-step priorities.
4. `change.txt`
   Running research log and paper-facing notes.
5. `TimeIndex.pdf`
   Original concept document.

## Installation

Requirements:

1. Python 3.11+
2. `numpy`
3. `pytest`

Suggested setup:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Run the full test suite:

```bash
pytest
```

Focused sanity checks:

```bash
pytest -q tests/test_retrieval.py tests/test_construction.py
pytest -q benchmarks/ibm_aml/test_deepseek_agent.py
pytest -q benchmarks/lanl/test_verifier.py
```

## Package Summary

Main package files:

1. `timeindex/event.py`
   Shared dataclasses such as `Event`, `EventRecord`, `EvidenceObject`, `OrdinaryLink`, `ChainSummary`, and `SkipLink`.
2. `timeindex/config.py`
   Configuration objects and feature flags.
3. `timeindex/extractors.py`
   Lookup-key extraction, vector sketching, and aspect extraction.
4. `timeindex/scoring.py`
   Ordinary-link, skip-link, retrieval, and utility scores.
5. `timeindex/stores.py`
   In-memory stores and directories.
6. `timeindex/candidate_index.py`
   Skip-candidate indexing logic.
7. `timeindex/construction.py`
   Online index construction.
8. `timeindex/retrieval.py`
   Budgeted dual-frontier retrieval.
9. `timeindex/sqlite_backend.py`
   SQLite persistence and hot-cache retrieval backend.
10. `timeindex/synthetic.py`
    Synthetic transaction and log generators.
11. `timeindex/evaluation.py`
    Small evaluation helpers.

## Current Technical Status

High-level summary:

1. TimeIndex already works as a causal, budgeted temporal retrieval system.
2. The SQLite backend is necessary for larger local experiments.
3. Ordinary links and chains provide the stable retrieval backbone.
4. Skip links help most when local traversal cannot bootstrap or cannot reach distant bridge evidence.
5. Verifier logic is useful as a precision filter, especially on LANL.
6. The biggest remaining problem on real workloads is still evidence recall.

For the most recent research notes, see:

1. `change.txt`
2. `outputs/lanl/verifier_audit_50p_50n_v2/verifier_audit.json`
3. `outputs/lanl/skip_effectiveness_tail1m_b12.json`
4. `outputs/lanl/skip_effectiveness_tail1m_b12_diagnosis.json`

## IBM AML Benchmark

The IBM AML benchmark adapts transaction CSV files into `TimeIndex` events and evaluates budgeted temporal evidence retrieval.

Main benchmark question:

1. For a suspicious transaction query, can we retrieve compact historical supporting evidence under a small budget?

Expected local layout:

```text
data/
  ibm_aml/
    *.csv
```

Benchmark focus:

1. Evidence recall
2. Evidence precision
3. Context efficiency
4. Retrieval latency
5. Update throughput
6. Strict temporal causality: no returned event after the query

Useful commands:

```bash
python benchmarks/ibm_aml/profile_dataset.py --path data/ibm_aml/sample.csv
python benchmarks/ibm_aml/profile_dataset.py --path data/ibm_aml/
python benchmarks/ibm_aml/run_timeindex.py data/ibm_aml/sample.csv
python benchmarks/ibm_aml/run_ablations.py --path data/ibm_aml/sample.csv
python benchmarks/ibm_aml/run_sqlite_deepseek_sample.py outputs/ibm_aml/cache/timeindex_first_5000000_streaming_skipv2.sqlite
```

Relevant files:

1. `benchmarks/ibm_aml/schema.py`
2. `benchmarks/ibm_aml/adapter.py`
3. `benchmarks/ibm_aml/evidence.py`
4. `benchmarks/ibm_aml/run_timeindex.py`
5. `benchmarks/ibm_aml/run_sqlite_ablation_compare.py`
6. `benchmarks/ibm_aml/run_sqlite_deepseek_sample.py`
7. `benchmarks/ibm_aml/deepseek_agent.py`
8. `benchmarks/ibm_aml/two_agent.py`

Current honest status on AML:

1. Retrieval is already reasonably strong on larger AML slices.
2. Downstream prompt-based judging is usable but not final.
3. Skip-link benefit on AML is still weak and dataset-dependent.
4. AML is useful for scale and causal retrieval benchmarking, but not yet the strongest proof that skip links are essential.

Main output files:

1. `outputs/ibm_aml/retrieval_results.jsonl`
2. `outputs/ibm_aml/config.json`
3. `outputs/ibm_aml/dataset_profile.json`
4. `outputs/ibm_aml/run_summary.json`
5. `outputs/ibm_aml/ablations/*.jsonl`
6. `outputs/ibm_aml/ablations/aggregate.csv`

Minimal expected result:

1. On the built-in synthetic sanity path, `timeindex_full` should beat `chain_only` on evidence recall at small budgets because skip evidence can recover earlier accumulation context.

## LANL Authentication Benchmark

The LANL benchmark adapts the auth-only slice of the LANL cyber dataset to the same budgeted temporal retrieval setting.

Main benchmark question:

1. Can skip links recover distant lateral-movement context that ordinary traversal misses?

Expected local layout:

```text
data/
  lanl/
    auth.gz
    redteam
    auth_window_*.txt
    redteam_window_*.txt
```

Useful commands:

```bash
python benchmarks/lanl/profile_dataset.py data/lanl/auth.gz --redteam-path data/lanl/redteam
python scripts/build_lanl_timeindex_sqlite.py
python benchmarks/lanl/run_deepseek_sample.py outputs/lanl/cache/timeindex_tail1m_eval.sqlite outputs/lanl/query_set_tail1m_66p_66n.json --domain lanl --budget 8 --adaptive-budget 12
python benchmarks/lanl/run_sqlite_ablation_compare.py data/lanl/auth_window_744000_768999_tail1m.txt data/lanl/redteam_window_744000_768999_tail1m.txt outputs/lanl/cache/timeindex_tail1m_eval.sqlite
```

Relevant files:

1. `benchmarks/lanl/schema.py`
2. `benchmarks/lanl/adapter.py`
3. `benchmarks/lanl/evidence.py`
4. `benchmarks/lanl/run_timeindex.py`
5. `benchmarks/lanl/run_deepseek_sample.py`
6. `benchmarks/lanl/run_sqlite_ablation_compare.py`
7. `benchmarks/lanl/verifier.py`

Current honest status on LANL:

1. Skip links are selectively useful.
2. The best skip-link value appears in bootstrap-recovery and long-range bridge cases.
3. Many changed retrievals are only lateral substitutions, not major gains.
4. The current verifier meaningfully improves precision by removing weak corridor-like false positives.
5. Recall remains the main bottleneck.

## Skip-Link Effectiveness

Current LANL conclusion:

1. Skip links are useful, but not uniformly useful.
2. Their strongest value is in `bootstrap_recovery` and `long_range_bridge` cases where ordinary or no-skip retrieval misses older but structurally relevant history.
3. Many changed queries are only `lateral_swap` cases, where skip links alter the evidence set but do not materially improve temporal reach.
4. Skip links should be kept, but their ranking and gating policy still needs improvement so they fire mainly on bridge-worthy cases.

Current AML conclusion:

1. Skip links are implemented and efficient.
2. Their measurable retrieval advantage on AML is still weak.
3. This should be treated as an honest open optimization target, not hidden.

## Verifier Status

The LANL verifier is designed as a precision filter, not a second classifier.

It currently:

1. Preserves bridge-backed or chain-backed positives.
2. Removes weak skip-induced and corridor-like false positives.
3. Should always be evaluated using same-run `llm_predicted_positive` versus final `predicted_positive`, not separate reruns alone.

Current status:

1. The verifier is no longer the main problem.
2. The main remaining weakness is recall, not verifier safety.

## Synthetic Demo

The synthetic helpers provide a stable prototype harness before real datasets or heavier evaluation tooling.

```python
from timeindex.evaluation import evidence_recall
from timeindex.synthetic import (
    baseline_nearest_neighbor,
    baseline_recent_window,
    synthetic_transaction_stream,
)

events = synthetic_transaction_stream()
query = events[-1]
gold_event_ids = {"e1", "e2", "e3", "e5", "e6"}

recent = baseline_recent_window(events, query, budget=3)
nearest = baseline_nearest_neighbor(events, query, budget=3)

print("recent ids:", [event.event_id for event in recent])
print("nearest ids:", [event.event_id for event in nearest])
print("nearest recall:", evidence_recall(nearest, gold_event_ids))
```

Expected transaction storyline:

1. `e1`, `e3`: deposits into account `A`
2. `e2`, `e4`: routine outgoing payments
3. `e5`: new beneficiary `B`
4. `e6`: accumulated balance snapshot
5. `e7`: full-balance transfer from `A` to `B`

## Plotting

To generate IBM AML plots from `aggregate.csv`:

```python
import csv

from benchmarks.plots import (
    plot_ablation_bars,
    plot_context_efficiency,
    plot_latency_vs_budget,
    plot_precision_at_budget,
    plot_recall_at_budget,
)

with open("outputs/ibm_aml/ablations/aggregate.csv", "r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

full_rows = [row for row in rows if row["variant"] == "timeindex_full"]
plot_recall_at_budget(full_rows, "outputs/ibm_aml/plots/recall")
plot_precision_at_budget(full_rows, "outputs/ibm_aml/plots/precision")
plot_context_efficiency(full_rows, "outputs/ibm_aml/plots/context_efficiency")
plot_latency_vs_budget(full_rows, "outputs/ibm_aml/plots/latency")
plot_ablation_bars(
    [
        {"name": row["variant"].replace("_", " "), "score": float(row["recall"])}
        for row in rows
        if row["budget"] == "5"
    ],
    "outputs/ibm_aml/plots/ablation_recall_b5",
)
```

Each plot helper saves both `.png` and `.pdf`.

## Operational Notes

Git and local files:

1. `outputs/` is local-only
2. `data/ibm_aml/` and `data/lanl/` are local-only
3. local API key notes are local-only

Long-running jobs:

1. Index builds and large benchmark runs can take a long time.
2. Use the SQLite builders in `scripts/`.
3. Use progress JSON files or `scripts/watch_progress_milestones.py` to monitor runs.

Prompt-based evaluation:

1. Network and API behavior are not perfectly deterministic.
2. When comparing verifier behavior, use same-run `llm_predicted_positive` versus final `predicted_positive`.
3. Do not infer verifier benefit from separate reruns alone.

## Suggested Next Work

Best next directions:

1. Improve retrieval recall on LANL and AML.
2. Rank skip links more selectively so they fire mostly on bridge-worthy cases.
3. Refine downstream evidence presentation without overfitting to one dataset.
4. Benchmark on additional temporal datasets with stronger long-range structure.
5. Keep paper claims conservative and tied to measured retrieval behavior.

Current backlog is tracked in:

1. `TODO.txt`
2. `change.txt`
3. `agent.md`
