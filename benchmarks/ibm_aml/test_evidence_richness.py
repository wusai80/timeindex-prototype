from __future__ import annotations

import json

from benchmarks.ibm_aml.evidence_richness import (
    analyze_evidence_richness,
    analyze_probe_json,
    summarize_richness,
)
from timeindex.event import Event


def test_analyze_evidence_richness_detects_touching_history() -> None:
    query = Event(
        event_id="q1",
        time=100.0,
        event_type="transfer",
        attrs={"src_account": "A", "dst_account": "B", "amount": 500.0},
    )
    retrieved = [
        Event(
            event_id="e1",
            time=10.0,
            event_type="deposit",
            attrs={"src_account": "X", "dst_account": "A", "amount": 300.0},
        ),
        Event(
            event_id="e2",
            time=20.0,
            event_type="transfer",
            attrs={"src_account": "A", "dst_account": "Y", "amount": 250.0},
        ),
        Event(
            event_id="e3",
            time=30.0,
            event_type="transfer",
            attrs={"src_account": "A", "dst_account": "Y", "amount": 260.0},
        ),
    ]

    report = analyze_evidence_richness(
        query,
        retrieved,
        retrieved_aspects={"large_transfer", "source_accumulation"},
        retrieved_object_types=["chain", "chain"],
    )

    assert report.touching_event_count == 3
    assert report.inbound_to_query_src_count == 1
    assert report.outbound_from_query_src_count == 2
    assert report.repeated_pair_count == 1
    assert report.richness_score > 0.0
    assert report.decision_ready is True


def test_summarize_richness_handles_empty_input() -> None:
    summary = summarize_richness([])

    assert summary["queries"] == 0.0
    assert summary["decision_ready_rate"] == 0.0


def test_analyze_probe_json_reads_saved_probe(tmp_path) -> None:
    csv_path = tmp_path / "events.csv"
    csv_path.write_text(
        "\n".join(
            [
                "transaction_id,timestamp,src_account,dst_account,amount,payment_format,is_laundering",
                "e1,2019/01/01 00:00,A,B,10,ACH,0",
                "e2,2019/01/01 00:05,X,A,25,ACH,0",
                "q1,2019/01/01 00:10,A,C,40,ACH,1",
            ]
        ),
        encoding="utf-8",
    )
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(
        json.dumps(
            {
                "report": [
                    {
                        "query_event_id": "q1",
                        "object_summaries": [{"kind": "chain"}],
                        "samples": [{"event_id": "e2"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = analyze_probe_json(probe_path, csv_path)

    assert payload["summary"]["queries"] == 1.0
    assert payload["reports"][0]["query_event_id"] == "q1"
    assert payload["reports"][0]["touching_event_count"] == 1
