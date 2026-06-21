# IBM AML Benchmark

This benchmark adapts IBM AML-style transaction CSV files into `TimeIndex` events and evaluates budgeted temporal evidence retrieval. The focus is not classification-first. The main question is:

- for a suspicious transaction query, can we retrieve compact historical supporting evidence under a small budget?

## Dataset Placement

Expected layout:

```text
data/
  ibm_aml/
    *.csv
```

Place one or more IBM AML CSV files in `data/ibm_aml/`. If you are using a single file, examples below assume `data/ibm_aml/sample.csv`.

## Schema Summary

The adapter and profiler look for these canonical fields:

- `timestamp`: examples include `timestamp`, `time`, `step`, `datetime`, `date`
- `source_account`: examples include `src_account`, `source_account`, `from_account`, `account`
- `destination_account`: examples include `dst_account`, `target_account`, `to_account`, `beneficiary_account`
- `amount`: examples include `amount`, `amount_paid`, `payment_amount`, `amount_received`
- `currency`: examples include `currency`, `payment_currency`, `receiving_currency`
- `payment_format`: examples include `payment_format`, `payment system`, `format`
- `transaction_type`: examples include `type`, `transaction_type`, `payment_type`
- `source_bank`: examples include `src_bank`, `source_bank`, `from_bank`, `bankorig`
- `destination_bank`: examples include `dst_bank`, `target_bank`, `to_bank`, `bankdest`
- `laundering_label`: examples include `is_laundering`, `label`, `suspicious`
- `pattern_id`: examples include `pattern_id`, `typology`, `pattern`, `alert_id`, `group_id`

Resolved schema objects are produced by [schema.py](/Users/wusai/Documents/code/EventIndex/benchmarks/ibm_aml/schema.py) and consumed by [adapter.py](/Users/wusai/Documents/code/EventIndex/benchmarks/ibm_aml/adapter.py).

## Profiling

Profile one file or a whole directory:

```bash
python benchmarks/ibm_aml/profile_dataset.py --path data/ibm_aml/sample.csv
python benchmarks/ibm_aml/profile_dataset.py --path data/ibm_aml/
```

The profiler prints:

- detected time, source, destination, label, and pattern columns
- row count and column list
- missing values
- class imbalance
- unique account count
- transactions-per-account summary
- time range

## Running TimeIndex

Run the main benchmark:

```bash
python benchmarks/ibm_aml/run_timeindex.py data/ibm_aml/sample.csv
```

Optional knobs:

```bash
python benchmarks/ibm_aml/run_timeindex.py data/ibm_aml/sample.csv \
  --output-dir outputs/ibm_aml \
  --negative-query-sample-size 100 \
  --random-seed 0 \
  --budgets 3 5 10 20
```

By default, the runner queries laundering-labeled events only.

## Running Baselines

Baseline retrieval methods are exercised through the ablation runner:

- `recent_window`
- `same_entity_window`
- `nearest_neighbor`
- `chain_only`

## Running Ablations

Run the full ablation suite:

```bash
python benchmarks/ibm_aml/run_ablations.py --path data/ibm_aml/sample.csv
```

Variants:

- `recent_window`
- `same_entity_window`
- `nearest_neighbor`
- `chain_only`
- `timeindex_no_skip`
- `timeindex_no_bridge`
- `timeindex_no_aspect`
- `timeindex_full`

## Generating Plots

After ablations finish, generate plots from `outputs/ibm_aml/ablations/aggregate.csv`:

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

## Output Files

Main benchmark outputs:

- `outputs/ibm_aml/retrieval_results.jsonl`: per-query retrieved evidence ids, object types, aspects, gold ids, latency, and index stats
- `outputs/ibm_aml/config.json`: benchmark config snapshot
- `outputs/ibm_aml/dataset_profile.json`: dataset summary
- `outputs/ibm_aml/run_summary.json`: aggregate benchmark summary

Ablation outputs:

- `outputs/ibm_aml/ablations/<variant>.jsonl`: per-query results for one retrieval method
- `outputs/ibm_aml/ablations/aggregate.csv`: aggregate metrics by variant and budget
- `outputs/ibm_aml/ablations/ablation_summary.json`: run metadata

## Expected Sanity Result

On the synthetic sanity path, `timeindex_full` should improve Evidence Recall@B over `chain_only` at small budgets because skip links can surface earlier accumulation evidence that ordinary-link traversal alone may miss.
