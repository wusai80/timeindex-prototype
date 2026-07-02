from __future__ import annotations

import json
from pathlib import Path

from benchmarks.lanl.run_sqlite_ablation_compare import run_sqlite_ablation_compare


def test_sqlite_ablation_compare_outputs_expected_files(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.txt"
    redteam_path = tmp_path / "redteam.txt"
    index_path = tmp_path / "fake.sqlite"
    output_dir = tmp_path / "outputs"

    auth_path.write_text(
        "\n".join(
            [
                "1,alice@LANL,alice@LANL,c1,c2,NTLM,Network,LogOn,Success",
                "2,alice@LANL,alice@LANL,c2,c3,NTLM,Network,LogOn,Success",
                "3,alice@LANL,admin@LANL,c3,c4,NTLM,Network,LogOn,Success",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    redteam_path.write_text("3,alice@LANL,c3,c4\n", encoding="utf-8")

    from timeindex.config import TimeIndexConfig
    from timeindex.construction import TimeIndex
    from timeindex.sqlite_backend import export_sqlite_backend
    from benchmarks.lanl.adapter import stream_events

    index = TimeIndex(TimeIndexConfig())
    for record in stream_events(auth_path, redteam_path):
        index.insert(record.event)
    export_sqlite_backend(index, index_path, overwrite=True)

    result = run_sqlite_ablation_compare(
        auth_path,
        redteam_path,
        index_path,
        output_dir=output_dir,
        budgets=(4,),
        positive_query_limit=1,
        negative_query_sample_size=1,
    )

    assert Path(result["results_path"]).exists()
    assert Path(result["aggregate_path"]).exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()

    rows = [json.loads(line) for line in (output_dir / "retrieval_results.jsonl").read_text().splitlines() if line.strip()]
    assert rows
    assert {row["variant"] for row in rows} == {"chain_only", "timeindex_no_skip", "timeindex_full"}
    assert all(len(row["retrieved_event_ids"]) <= row["budget"] for row in rows)
