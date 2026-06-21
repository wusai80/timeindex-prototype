from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from benchmarks.ibm_aml.run_timeindex import run_benchmark
from timeindex import Event


def test_runner_works_on_tiny_synthetic_csv(tmp_path) -> None:
    _install_test_modules()
    csv_path = _write_tiny_csv(tmp_path)
    output_dir = tmp_path / "outputs"

    result = run_benchmark(
        csv_path,
        output_dir=output_dir,
        budgets=(3, 5, 10, 20),
        negative_query_sample_size=1,
        random_seed=7,
    )

    assert Path(result["results_path"]).exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "dataset_profile.json").exists()
    assert (output_dir / "run_summary.json").exists()


def test_output_jsonl_fields_exist(tmp_path) -> None:
    _install_test_modules()
    csv_path = _write_tiny_csv(tmp_path)
    output_dir = tmp_path / "outputs"

    run_benchmark(csv_path, output_dir=output_dir, budgets=(3,))

    rows = _read_jsonl(output_dir / "retrieval_results.jsonl")
    assert rows
    required_fields = {
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
        assert required_fields <= row.keys()


def test_retrieved_events_are_before_query(tmp_path) -> None:
    _install_test_modules()
    csv_path = _write_tiny_csv(tmp_path)
    output_dir = tmp_path / "outputs"

    run_benchmark(csv_path, output_dir=output_dir, budgets=(3, 5))

    time_by_id = {
        "e1": 1,
        "e2": 2,
        "e3": 3,
        "e4": 4,
        "e5": 5,
    }
    for row in _read_jsonl(output_dir / "retrieval_results.jsonl"):
        query_time = time_by_id[row["query_event_id"]]
        assert all(time_by_id[event_id] < query_time for event_id in row["retrieved_event_ids"])


def test_budgets_are_respected(tmp_path) -> None:
    _install_test_modules()
    csv_path = _write_tiny_csv(tmp_path)
    output_dir = tmp_path / "outputs"

    run_benchmark(csv_path, output_dir=output_dir, budgets=(3, 5, 10, 20))

    rows = _read_jsonl(output_dir / "retrieval_results.jsonl")
    assert {row["budget"] for row in rows} == {3, 5, 10, 20}
    for row in rows:
        assert len(row["retrieved_event_ids"]) <= row["budget"]


def _install_test_modules() -> None:
    adapter_module = types.ModuleType("benchmarks.ibm_aml.adapter")
    evidence_module = types.ModuleType("benchmarks.ibm_aml.evidence")

    def load_ibm_aml_csv(csv_path: Path) -> list[Event]:
        rows = [
            ("e1", 1, "A", "B", 200, "0", False, 1),
            ("e2", 2, "A", "C", 400, "0", False, 1),
            ("e3", 3, "A", "D", 1500, "1", True, 2),
            ("e4", 4, "X", "Y", 50, "0", False, 1),
            ("e5", 5, "A", "E", 2200, "1", True, 4),
        ]
        del csv_path
        return [
            Event(
                event_id=event_id,
                time=timestamp,
                event_type="transaction",
                attrs={
                    "account_id": account_id,
                    "beneficiary_id": beneficiary_id,
                    "amount": amount,
                    "is_new_beneficiary": is_new_beneficiary,
                    "burst_count": burst_count,
                },
                label=label,
            )
            for event_id, timestamp, account_id, beneficiary_id, amount, label, is_new_beneficiary, burst_count in rows
        ]

    def build_gold_supporting_evidence(events: list[Event]) -> dict[str, list[str]]:
        del events
        return {
            "e3": ["e1", "e2"],
            "e5": ["e2", "e3"],
        }

    adapter_module.load_ibm_aml_csv = load_ibm_aml_csv
    evidence_module.build_gold_supporting_evidence = build_gold_supporting_evidence
    sys.modules["benchmarks.ibm_aml.adapter"] = adapter_module
    sys.modules["benchmarks.ibm_aml.evidence"] = evidence_module


def _write_tiny_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "ibm_aml_tiny.csv"
    csv_path.write_text(
        "\n".join(
            [
                "event_id,timestamp,account_id,beneficiary_id,amount,label,is_new_beneficiary,burst_count",
                "e1,1,A,B,200,0,false,1",
                "e2,2,A,C,400,0,false,1",
                "e3,3,A,D,1500,1,true,2",
                "e4,4,X,Y,50,0,false,1",
                "e5,5,A,E,2200,1,true,4",
            ]
        ),
        encoding="utf-8",
    )
    return csv_path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
