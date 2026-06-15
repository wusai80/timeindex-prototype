# TimeIndex Prototype

This repository contains a Python 3.11+ skeleton for the TimeIndex prototype described in [`TimeIndex.pdf`](./TimeIndex.pdf). The current package defines shared dataclasses, configuration objects, storage abstractions, and empty method stubs for the online temporal evidence index, without implementing the algorithms yet.

## Method Summary

TimeIndex is framed in the paper as a value-aware multi-chain temporal evidence index for streaming decisions:

1. Each incoming event is represented as `(id, time, type, attrs, ctx, text, label)`.
2. The extractor produces lookup keys, a compact vector sketch, and evidence aspects for that event.
3. The online index maintains six main components:
   `EventStore`, `KeyDirectory`, `EdgeStore`, `ChainStore`, `SkipCandidateIndex`, and `SkipLinkStore`.
4. Ordinary dependency links preserve local temporal continuity between related events.
5. Skip links preserve direct access to distant high-value evidence anchors that might be missed by chain-only traversal.
6. Retrieval uses a dual-frontier procedure that explores both ordinary links and skip links under a context budget.
7. Synthetic streams are intended for prototype experiments and ablations.

This scaffold intentionally stops at the interface layer:

- Python 3.11+
- `dataclasses` and type hints throughout
- `numpy` as the only runtime dependency
- `pytest` for tests
- no algorithmic implementation yet

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
