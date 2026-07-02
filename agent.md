# Agent Guide for TimeIndex

Purpose
-------
This file is for future coding agents working in this repository.
Treat it as the local working contract for implementation, debugging,
benchmarking, and paper-support tasks.


Project mission
---------------
TimeIndex is an efficient temporal evidence index.
The primary goal is not generic RAG and not pure classification.

The core target is:

1. given a query event
2. retrieve compact historical supporting evidence
3. under a small budget
4. while preserving temporal causality
5. with low update cost and low retrieval latency

When making design decisions, optimize for:

1. causal correctness
2. retrieval usefulness
3. scalability
4. deterministic behavior where practical
5. readable code over clever code


What matters most
-----------------
If you are unsure what to optimize, prioritize in this order:

1. no future evidence after the query
2. bounded retrieval and bounded fan-in
3. retrieval-side evidence quality
4. index build/update throughput
5. downstream judge quality

Do not let downstream prompt tweaks redefine the project.
The repository is about temporal indexing first.


Repository working map
----------------------

Core package:

1. `timeindex/event.py`
2. `timeindex/config.py`
3. `timeindex/extractors.py`
4. `timeindex/scoring.py`
5. `timeindex/stores.py`
6. `timeindex/candidate_index.py`
7. `timeindex/construction.py`
8. `timeindex/retrieval.py`
9. `timeindex/sqlite_backend.py`
10. `timeindex/synthetic.py`
11. `timeindex/evaluation.py`

Benchmarks:

1. `benchmarks/ibm_aml/`
2. `benchmarks/lanl/`
3. `benchmarks/baselines.py`
4. `benchmarks/metrics.py`
5. `benchmarks/plots.py`

Operational scripts:

1. `scripts/build_streaming_timeindex_sqlite.py`
2. `scripts/build_lanl_timeindex_sqlite.py`
3. `scripts/watch_progress_milestones.py`

Research notes:

1. `change.txt`
2. `README.md`
3. `README.txt`


Current technical state
-----------------------

IBM AML:

1. retrieval works at scale on large SQLite indexes
2. skip links are implemented but not yet showing consistent retrieval gains
3. use AML for scale, causal retrieval, and budgeted evidence benchmarking
4. do not over-claim skip-link value on AML

LANL:

1. skip links are selectively useful
2. strongest gains are in bootstrap recovery and long-range bridge cases
3. a lightweight verifier now removes weak skip/corridor false positives
4. recall is still the main weakness

Current verifier status on LANL:

1. good precision filter
2. not the main bottleneck anymore
3. compare same-run `llm_predicted_positive` vs final `predicted_positive`
   when auditing it


Design principles
-----------------

Always preserve:

1. strict temporal causality
2. deterministic local logic unless external APIs are required
3. bounded memory behavior in stores and retrieval
4. cheap access to recent and structurally relevant events

Prefer:

1. explicit dataclasses
2. simple score decompositions
3. transparent heuristics
4. benchmark scripts that write JSON or CSV artifacts
5. instrumentation that helps explain why retrieval changed

Avoid:

1. heavy dependencies
2. algorithm changes that only improve one hand-picked example
3. paper claims that outrun the measured results
4. using future information anywhere in retrieval or evidence labeling


When changing core retrieval/index code
---------------------------------------

Before editing:

1. read the affected core files
2. check nearby tests
3. check `change.txt` for current intended behavior

After editing:

1. run focused unit tests
2. if retrieval semantics changed, inspect one or two real query cases
3. if skip logic changed, evaluate both helpful and harmful cases
4. update `change.txt` when the change affects paper-facing claims

Core regression checklist:

1. no self-predecessor links
2. no future evidence returned
3. ordinary and skip fan-in bounds still hold
4. chain summaries still bounded
5. SQLite backend still returns equivalent retrieval objects


When changing prompts or downstream judges
------------------------------------------

Remember:

1. prompt changes are noisy
2. separate LLM drift from retrieval or verifier effects
3. do not use separate reruns alone to claim verifier benefit

Best practice:

1. run one LLM pass
2. store `llm_predicted_positive`
3. apply verifier or post-processing on the same run
4. audit override quality directly

If a prompt change improves a few examples but weakens the project goal,
do not keep it.


When evaluating skip links
--------------------------

Use three questions:

1. does full retrieval differ from no-skip retrieval
2. does skip increase temporal reach
3. does the extra evidence help downstream judgment

Important distinction:

1. many skip changes are only lateral substitutions
2. the most valuable cases are bootstrap recovery and long-range bridge

Do not judge skip links only by object count.
Judge them by whether they recover otherwise inaccessible causal context.


Recommended commands
--------------------

Core tests:

```bash
pytest -q tests/test_retrieval.py tests/test_construction.py tests/test_scoring.py tests/test_stores.py
```

Prompt and verifier checks:

```bash
pytest -q benchmarks/ibm_aml/test_deepseek_agent.py benchmarks/lanl/test_verifier.py
```

IBM AML benchmark examples:

```bash
python benchmarks/ibm_aml/profile_dataset.py --path data/ibm_aml/sample.csv
python benchmarks/ibm_aml/run_ablations.py --path data/ibm_aml/sample.csv
```

LANL benchmark examples:

```bash
python benchmarks/lanl/profile_dataset.py data/lanl/auth.gz --redteam-path data/lanl/redteam
python benchmarks/lanl/run_deepseek_sample.py outputs/lanl/cache/timeindex_tail1m_eval.sqlite outputs/lanl/query_set_tail1m_66p_66n.json --domain lanl --budget 8 --adaptive-budget 12
```


Git and local-data rules
------------------------

Never commit:

1. `data/`
2. `outputs/`
3. local API key files
4. private prompt dumps unless explicitly intended

Before a sync:

1. check `git status --short`
2. verify `.gitignore` still excludes local data and outputs
3. stage only code, docs, tests, and scripts


How to write useful research updates
------------------------------------

Whenever you learn something important, record:

1. what changed
2. what file or artifact proves it
3. whether the result is positive, negative, or mixed
4. what claim is safe to make in the paper

Preferred file:

1. `change.txt`

Do not hide negative results.
Negative results are especially important for skip-link claims.


Good next-step heuristics
-------------------------

If stuck, choose one of these:

1. inspect false positives that survive the verifier
2. inspect false negatives with very weak retrieved context
3. compare `chain_only`, `no_skip`, and `full` retrieval
4. instrument score funnels rather than guessing
5. prefer a small controlled rerun over a very large noisy one


Bottom line
-----------

Be conservative, causal, and empirical.

The best contribution from this repository is:

1. a scalable temporal evidence index
2. with budgeted causal retrieval
3. whose skip links help on the right workloads
4. and whose behavior is measured honestly

