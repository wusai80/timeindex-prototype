from __future__ import annotations

from pathlib import Path

from benchmarks.ibm_aml.adapter import convert_row_to_event, load_ibm_aml_csv, stream_events
from timeindex.event import EventRecord


def test_adapter_creates_event_record_objects(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "aml.csv",
        [
            "transaction_id,timestamp,src_account,dst_account,amount,currency,src_bank,dst_bank,payment_format,is_laundering",
            "tx-1,2024-01-01T00:00:00,alice,bob,125.5,USD,bank-a,bank-b,wire,1",
        ],
    )

    rows = load_ibm_aml_csv(path)
    record = convert_row_to_event(rows[0], {"dataset_name": "ibm_aml", "source_file": str(path)})

    assert isinstance(record, EventRecord)
    assert record.event.event_id == "tx-1"
    assert record.event.attrs["src_account"] == "alice"
    assert record.event.ctx["source_file"] == str(path)
    assert "alice" in (record.event.text or "")


def test_events_are_sorted_by_time(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "aml.csv",
        [
            "transaction_id,timestamp,src_account,dst_account,amount",
            "tx-2,2024-01-02T00:00:00,a,b,10",
            "tx-1,2024-01-01T00:00:00,a,b,20",
        ],
    )

    records = list(stream_events(path, {"source_file": str(path)}, sort_by_time=True))

    assert [record.event.event_id for record in records] == ["tx-1", "tx-2"]
    assert records[0].event.time <= records[1].event.time


def test_label_is_correctly_mapped(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "aml.csv",
        [
            "transaction_id,timestamp,is_laundering",
            "tx-9,2024-01-01T00:00:00,1",
        ],
    )

    record = next(stream_events(path, {"source_file": str(path)}))

    assert record.event.label == "1"


def test_missing_optional_fields_do_not_crash(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "aml.csv",
        [
            "transaction_id,timestamp,amount",
            "tx-3,not-a-date,42",
        ],
    )

    record = next(stream_events(path, {"source_file": str(path)}))

    assert record.event.event_id == "tx-3"
    assert record.event.attrs["src_bank"] is None
    assert record.event.attrs["payment_format"] is None
    assert isinstance(record.event.time, (int, float))


def test_amount_is_numeric_if_possible(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "aml.csv",
        [
            "transaction_id,timestamp,amount",
            "tx-4,2024-01-01T00:00:00,\"1,250.75\"",
        ],
    )

    record = next(stream_events(path, {"source_file": str(path)}))

    assert record.event.attrs["amount"] == 1250.75
    assert isinstance(record.event.attrs["amount"], float)


def _write_csv(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
