from __future__ import annotations

from datetime import datetime, timedelta

from benchmarks.ibm_aml.run_sqlite_realcases import (
    _same_entity_gold,
    _stream_realcase_gold,
    _weak_accumulation_gold,
)


def test_same_entity_gold_merges_unique_ids_in_order() -> None:
    history = {
        "A": [(datetime(2019, 1, 1, 1), "e1"), (datetime(2019, 1, 1, 2), "e2")],
        "B": [(datetime(2019, 1, 1, 3), "e2"), (datetime(2019, 1, 1, 4), "e3")],
    }

    assert _same_entity_gold(history, "A", "B") == ["e1", "e2", "e3"]


def test_weak_accumulation_gold_uses_recent_incoming_until_threshold() -> None:
    incoming = [
        (datetime(2019, 1, 1, 1), "e1", 20.0),
        (datetime(2019, 1, 1, 2), "e2", 25.0),
        (datetime(2019, 1, 1, 3), "e3", 40.0),
    ]

    assert _weak_accumulation_gold(incoming, threshold_amount=60.0) == ["e2", "e3"]
    assert _weak_accumulation_gold(incoming, threshold_amount=1000.0) == []


def test_stream_realcase_gold_builds_same_entity_and_accumulation_support(tmp_path) -> None:
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text(
        "\n".join(
            [
                "transaction_id,timestamp,src_bank,src_account,dst_bank,dst_account,amount,currency,amount_received,receiving_currency,payment_format,is_laundering",
                "e1,2019/01/01 00:00,b1,X,b2,A,30,USD,30,USD,ACH,0",
                "e2,2019/01/01 01:00,b1,Y,b2,A,40,USD,40,USD,ACH,0",
                "e3,2019/01/01 02:00,b1,A,b2,B,90,USD,90,USD,ACH,1",
                "e4,2019/01/01 03:00,b1,B,b2,C,20,USD,20,USD,ACH,1",
            ]
        ),
        encoding="utf-8",
    )

    queries = _stream_realcase_gold(
        csv_path,
        same_entity_window=timedelta(hours=24),
        accumulation_window=timedelta(hours=24),
        accumulation_threshold=0.75,
        limit_queries=None,
    )

    assert [query.query_event_id for query in queries] == ["e3", "e4"]
    assert queries[0].weak_accumulation_ids == ["e1", "e2"]
    assert queries[1].same_entity_ids == ["e3"]
