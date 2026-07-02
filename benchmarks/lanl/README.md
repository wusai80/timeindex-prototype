# LANL Auth Benchmark

This directory adapts the LANL multi-source cyber-security dataset to the TimeIndex prototype, starting with the auth-only slice.

## Expected raw files

Place the raw files in a local data directory such as:

```text
data/lanl/auth.txt.gz
data/lanl/redteam.txt.gz
```

The first implementation target is:

- `auth.txt.gz`
- `redteam.txt.gz`

## Current scope

This v1 slice maps LANL authentication rows into `EventRecord` objects and labels rows as positive when they match a red-team activity tuple by:

- `time`
- `src_user`
- `src_computer`
- `dst_computer`

## Files

- `schema.py`: canonical auth/red-team schema helpers
- `profile_dataset.py`: row-count and entity-distribution profiler
- `adapter.py`: auth and red-team parsing into `EventRecord`
- `evidence.py`: initial historical supporting-evidence policies
- `run_timeindex.py`: minimal retrieval benchmark runner

## Initial gold evidence policies

- `same_user_history`
- `same_computer_history`
- `lateral_chain`
- `novel_access_bootstrap`
- `union`

All policies enforce strict temporal causality: no support event may occur after the query event.

## Example commands

Profile the dataset:

```bash
python benchmarks/lanl/profile_dataset.py data/lanl/auth.txt.gz --redteam-path data/lanl/redteam.txt.gz
```

Run the minimal benchmark:

```bash
python benchmarks/lanl/run_timeindex.py data/lanl/auth.txt.gz data/lanl/redteam.txt.gz --output-dir outputs/lanl
```

## Planned next step

Once auth-only evaluation is stable, extend the benchmark with:

- `proc.txt.gz`
- `dns.txt.gz`
- `flows.txt.gz`

That will let TimeIndex retrieve mixed evidence objects across users, computers, processes, and network behavior.

