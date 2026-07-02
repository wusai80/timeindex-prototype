from __future__ import annotations

import json
from pathlib import Path

from benchmarks.lanl.run_timeindex import run_benchmark


def test_runner_works_on_tiny_lanl_slice(tmp_path: Path) -> None:
    auth_path, redteam_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "outputs"

    result = run_benchmark(
        auth_path,
        redteam_path,
        output_dir=output_dir,
        budgets=(4,),
        positive_query_limit=1,
        negative_query_sample_size=1,
        random_seed=7,
    )

    assert Path(result["results_path"]).exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "dataset_profile.json").exists()
    assert (output_dir / "run_summary.json").exists()


def test_output_jsonl_fields_exist(tmp_path: Path) -> None:
    auth_path, redteam_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "outputs"
    run_benchmark(auth_path, redteam_path, output_dir=output_dir, budgets=(4,), positive_query_limit=1)

    rows = _read_jsonl(output_dir / "retrieval_results.jsonl")
    assert rows
    required = {
        "query_event_id",
        "query_label",
        "budget",
        "retrieved_event_ids",
        "retrieved_object_types",
        "retrieved_aspects",
        "gold_event_ids",
        "latency_ms",
        "index_stats",
    }
    for row in rows:
        assert required <= row.keys()


def test_retrieved_events_are_before_query(tmp_path: Path) -> None:
    auth_path, redteam_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "outputs"
    run_benchmark(auth_path, redteam_path, output_dir=output_dir, budgets=(4,), positive_query_limit=1)

    time_by_id = {f"auth-{index:08d}": index + 1 for index in range(5)}
    for row in _read_jsonl(output_dir / "retrieval_results.jsonl"):
        query_time = time_by_id[row["query_event_id"]]
        assert all(time_by_id[event_id] < query_time for event_id in row["retrieved_event_ids"])


def test_budgets_are_respected(tmp_path: Path) -> None:
    auth_path, redteam_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "outputs"
    run_benchmark(auth_path, redteam_path, output_dir=output_dir, budgets=(4, 8), positive_query_limit=1)

    rows = _read_jsonl(output_dir / "retrieval_results.jsonl")
    assert {row["budget"] for row in rows} == {4, 8}
    for row in rows:
        assert len(row["retrieved_event_ids"]) <= row["budget"]


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    auth_path = tmp_path / "auth.txt"
    redteam_path = tmp_path / "redteam.txt"
    auth_path.write_text(
        "\n".join(
            [
                "1,alice@LANL,bob@LANL,c1,c2,Kerberos,Network,LogOn,Success",
                "2,alice@LANL,bob@LANL,c2,c3,Kerberos,Network,LogOn,Success",
                "3,alice@LANL,bob@LANL,c3,c4,Kerberos,Network,LogOn,Success",
                "4,charlie@LANL,svc@LANL,c7,c8,Kerberos,Network,LogOn,Success",
                "5,alice@LANL,admin@LANL,c4,c5,Kerberos,Network,LogOn,Success",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    redteam_path.write_text("5,alice@LANL,c4,c5\n", encoding="utf-8")
    return auth_path, redteam_path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
