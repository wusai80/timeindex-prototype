TimeIndex Prototype
===================

Overview
--------
TimeIndex is a prototype temporal evidence index for streaming decision support.
The goal is not generic vector search. The goal is to retrieve compact,
causally valid historical evidence for a query event under a small context budget.

Core ideas:

1. events are represented with structured attributes, context, and optional text
2. ordinary links preserve local temporal continuity
3. chain summaries compress multi-step histories
4. skip links provide direct access to distant but valuable evidence anchors
5. retrieval uses both ordinary and skip frontiers under a fixed budget
6. downstream agents or rules consume evidence objects, not the whole raw history

The repository currently contains:

1. the TimeIndex core package in `timeindex/`
2. synthetic examples and tests
3. an IBM AML benchmark harness
4. a LANL authentication benchmark harness
5. SQLite-backed indexing and retrieval utilities for larger runs
6. prompt-based and verifier-based downstream evaluation tooling


Repository layout
-----------------

Top-level directories:

1. `timeindex/`
   main package: event schema, extraction, scoring, stores, construction,
   retrieval, synthetic generators, evaluation
2. `tests/`
   package-level unit tests
3. `benchmarks/`
   dataset-specific adapters, evidence builders, runners, metrics, plots,
   baselines, and agent experiments
4. `scripts/`
   operational helpers for building SQLite indexes and monitoring long runs
5. `data/`
   local datasets only, not tracked in Git
6. `outputs/`
   local benchmark outputs, prompt dumps, reports, and cached indexes,
   also not tracked in Git

Important top-level files:

1. `README.md`
   current markdown overview
2. `README.txt`
   this plain-text operational overview
3. `change.txt`
   running research log and paper-facing notes
4. `agent.md`
   future coding-agent working guide
5. `pyproject.toml`
   package and dev dependency configuration
6. `TimeIndex.pdf`
   original concept document


Installation
------------

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

Run the test suite:

```bash
pytest
```

Focused sanity checks:

```bash
pytest -q tests/test_retrieval.py tests/test_construction.py
pytest -q benchmarks/ibm_aml/test_deepseek_agent.py
pytest -q benchmarks/lanl/test_verifier.py
```


TimeIndex package summary
-------------------------

Main package files:

1. `timeindex/event.py`
   shared dataclasses such as `Event`, `EventRecord`, `EvidenceObject`,
   `OrdinaryLink`, `ChainSummary`, and `SkipLink`
2. `timeindex/config.py`
   configuration objects and feature flags
3. `timeindex/extractors.py`
   lookup-key extraction, vector sketching, and aspect extraction
4. `timeindex/scoring.py`
   ordinary-link, skip-link, retrieval, and utility scores
5. `timeindex/stores.py`
   in-memory stores and directories
6. `timeindex/candidate_index.py`
   skip-candidate indexing logic
7. `timeindex/construction.py`
   online index construction
8. `timeindex/retrieval.py`
   budgeted dual-frontier retrieval
9. `timeindex/sqlite_backend.py`
   SQLite persistence and hot-cache retrieval backend
10. `timeindex/synthetic.py`
    synthetic transaction and log generators
11. `timeindex/evaluation.py`
    small evaluation helpers


IBM AML benchmark
-----------------

Purpose:

1. measure budgeted temporal evidence retrieval on synthetic financial data
2. focus on evidence recall, precision, context efficiency, latency,
   and update throughput
3. keep temporal causality strict: no retrieved evidence after the query

Expected local layout:

```text
data/
  ibm_aml/
    *.csv
```

Useful commands:

```bash
python benchmarks/ibm_aml/profile_dataset.py --path data/ibm_aml/sample.csv
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

Current honest status:

1. retrieval is already reasonably strong on larger AML slices
2. downstream prompt-based judging is usable but not final
3. skip-link benefit on AML is still weak and dataset-dependent
4. AML is good for scale and causal retrieval benchmarking, but not yet
   the strongest proof that skip links are essential


LANL authentication benchmark
-----------------------------

Purpose:

1. test TimeIndex on cyber-authentication event streams
2. evaluate whether skip links recover distant lateral movement context
3. test whether a lightweight verifier can filter weak skip-induced positives

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

Current honest status:

1. skip links are selectively useful on LANL
2. the best skip-link value appears in bootstrap-recovery and long-range bridge cases
3. many changed retrievals are only lateral substitutions, not major gains
4. the current verifier meaningfully improves precision by removing weak
   corridor-like false positives
5. recall remains the main bottleneck


Current research findings
-------------------------

High-level summary:

1. TimeIndex already works as a causal, budgeted temporal retrieval system
2. the SQLite backend is necessary for larger local experiments
3. ordinary links and chains provide the stable retrieval backbone
4. skip links help most when local traversal cannot bootstrap or cannot reach
   distant bridge evidence
5. verifier logic is useful as a precision filter, especially on LANL
6. the biggest remaining problem on real workloads is still evidence recall

For the most recent research notes, read:

1. `change.txt`
2. `outputs/lanl/verifier_audit_50p_50n_v2/verifier_audit.json`
3. `outputs/lanl/skip_effectiveness_tail1m_b12.json`
4. `outputs/lanl/skip_effectiveness_tail1m_b12_diagnosis.json`


Operational notes
-----------------

Git:

1. `outputs/` is local-only
2. `data/ibm_aml/` and `data/lanl/` are local-only
3. API key notes such as `key(not for uploaded to github).md` are local-only

Long-running jobs:

1. index builds and large benchmark runs can take a long time
2. use the SQLite builders in `scripts/`
3. use `scripts/watch_progress_milestones.py` or progress JSON files under
   `outputs/` to monitor runs

Prompt-based evaluation:

1. network and API behavior are not perfectly deterministic
2. when comparing verifier behavior, use same-run `llm_predicted_positive`
   versus final `predicted_positive`
3. do not infer verifier benefit from separate reruns alone


Suggested next work
-------------------

Best next directions:

1. improve retrieval recall on LANL and AML
2. rank skip links more selectively so they fire mostly on bridge-worthy cases
3. refine downstream evidence presentation without overfitting to one dataset
4. benchmark on additional temporal datasets with stronger long-range structure
5. keep paper claims conservative and tied to measured retrieval behavior

