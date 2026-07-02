from __future__ import annotations

import gzip
from pathlib import Path

from benchmarks.lanl.adapter import (
    convert_auth_row_to_event,
    load_lanl_auth,
    load_lanl_redteam,
    stream_events,
)
from benchmarks.lanl.schema import default_auth_schema
from timeindex.event import EventRecord


def test_adapter_creates_event_record_objects(tmp_path: Path) -> None:
    auth_path, redteam_path = _write_lanl_files(tmp_path)
    schema = default_auth_schema(source_file=str(auth_path), redteam_file=str(redteam_path))

    rows = load_lanl_auth(auth_path, schema=schema)
    redteam = load_lanl_redteam(redteam_path)
    record = convert_auth_row_to_event(rows[0], redteam, schema)

    assert isinstance(record, EventRecord)
    assert record.event.event_type == "authentication"
    assert record.event.attrs["src_user"] == "alice"
    assert "alice" in (record.event.text or "")


def test_events_are_sorted_by_time(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.txt"
    redteam_path = tmp_path / "redteam.txt"
    auth_path.write_text(
        "\n".join(
            [
                "2,alice@LANL,bob@LANL,c1,c2,Kerberos,Network,LogOn,Success",
                "1,alice@LANL,bob@LANL,c1,c3,Kerberos,Network,LogOn,Success",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    redteam_path.write_text("", encoding="utf-8")

    records = list(stream_events(auth_path, redteam_path))
    assert [record.event.time for record in records] == [1, 2]


def test_label_is_correctly_mapped_from_redteam(tmp_path: Path) -> None:
    auth_path, redteam_path = _write_lanl_files(tmp_path)
    record = next(stream_events(auth_path, redteam_path))
    assert record.event.label == "1"


def test_missing_optional_fields_do_not_crash(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.txt"
    redteam_path = tmp_path / "redteam.txt"
    auth_path.write_text("5,alice@LANL,bob@LANL,c1,c2,Kerberos,Network,,\n", encoding="utf-8")
    redteam_path.write_text("", encoding="utf-8")

    record = next(stream_events(auth_path, redteam_path))
    assert record.event.attrs["auth_orientation"] is None
    assert record.event.attrs["success"] is False


def test_gzip_auth_input_is_supported(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.txt.gz"
    redteam_path = tmp_path / "redteam.txt.gz"
    with gzip.open(auth_path, "wt", encoding="utf-8") as handle:
        handle.write("1,alice@LANL,bob@LANL,c1,c2,Kerberos,Network,LogOn,Success\n")
    with gzip.open(redteam_path, "wt", encoding="utf-8") as handle:
        handle.write("1,alice@LANL,c1,c2\n")

    records = list(stream_events(auth_path, redteam_path))
    assert len(records) == 1
    assert records[0].event.label == "1"


def test_row_index_offset_preserves_global_event_ids(tmp_path: Path) -> None:
    auth_path, redteam_path = _write_lanl_files(tmp_path)
    records = list(stream_events(auth_path, redteam_path, sort_by_time=False, row_index_offset=100))
    assert records[0].event.event_id == "auth-00000100"
    assert records[1].event.event_id == "auth-00000101"


def _write_lanl_files(tmp_path: Path) -> tuple[Path, Path]:
    auth_path = tmp_path / "auth.txt"
    redteam_path = tmp_path / "redteam.txt"
    auth_path.write_text(
        "\n".join(
            [
                "1,alice@LANL,bob@LANL,c1,c2,Kerberos,Network,LogOn,Success",
                "3,alice@LANL,bob@LANL,c2,c4,Kerberos,Network,LogOn,Success",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    redteam_path.write_text("1,alice@LANL,c1,c2\n", encoding="utf-8")
    return auth_path, redteam_path
