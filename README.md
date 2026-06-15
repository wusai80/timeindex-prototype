# TimeIndex Prototype

This repository contains a Python 3.11+ skeleton for the TimeIndex prototype described in [`TimeIndex.pdf`](./TimeIndex.pdf). The current package defines shared dataclasses, configuration objects, storage abstractions, and initial synthetic utilities for the online temporal evidence index, while most core indexing algorithms are still intentionally left for later implementation.

## Method Summary

TimeIndex is framed in the paper as a value-aware multi-chain temporal evidence index for streaming decisions:

1. Each incoming event is represented as `(id, time, type, attrs, ctx, text, label)`.
2. The extractor produces lookup keys, a compact vector sketch, and evidence aspects for that event.
3. The online index maintains six main components:
   `EventStore`, `KeyDirectory`, `EdgeStore`, `ChainStore`, `SkipCandidateIndex`, and `SkipLinkStore`.
4. Ordinary dependency links preserve local temporal continuity between related events.
5. Skip links preserve direct access to distant high-value evidence anchors that might be missed by chain-only traversal.
6. Retrieval uses a dual-frontier procedure that explores both ordinary links and skip links under a context budget.
7. Synthetic streams are intended for prototype experiments and ablations, and the repository now includes deterministic transaction and log examples plus lightweight baseline evaluators.

This scaffold currently includes:

- Python 3.11+
- `dataclasses` and type hints throughout
- `numpy` as the only runtime dependency
- `pytest` for tests
- deterministic synthetic examples and simple baselines
- core online indexing and retrieval algorithms still under active implementation

## Package Layout

```text
timeindex/
  __init__.py
  event.py
  config.py
  interfaces.py
  extractors.py
  scoring.py
  stores.py
  candidate_index.py
  construction.py
  retrieval.py
  synthetic.py
tests/
```

## Development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Synthetic Demo

The synthetic helpers are meant to give us a stable prototype harness before real datasets or heavier evaluation tooling.

```python
from timeindex.evaluation import evidence_recall
from timeindex.synthetic import (
    baseline_nearest_neighbor,
    baseline_recent_window,
    synthetic_transaction_stream,
)

events = synthetic_transaction_stream()
query = events[-1]  # e7: full-balance transfer to new beneficiary B
gold_event_ids = {"e1", "e2", "e3", "e5", "e6"}

recent = baseline_recent_window(events, query, budget=3)
nearest = baseline_nearest_neighbor(events, query, budget=3)

print("recent ids:", [event.event_id for event in recent])
print("nearest ids:", [event.event_id for event in nearest])
print("nearest recall:", evidence_recall(nearest, gold_event_ids))
```

Expected transaction storyline:

- `e1`, `e3`: deposits into account `A`
- `e2`, `e4`: routine outgoing payments
- `e5`: new beneficiary `B`
- `e6`: accumulated balance snapshot
- `e7`: full-balance transfer from `A` to `B`
