"""Evidence-richness analysis for IBM AML retrieval outputs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.ibm_aml.adapter import convert_row_to_event
from benchmarks.ibm_aml.schema import detect_schema
from timeindex.event import Event


@dataclass(slots=True)
class EvidenceRichnessReport:
    """Decision-readiness summary for one query and its retrieved context."""

    query_event_id: str
    retrieved_event_count: int
    touching_event_count: int
    touching_src_count: int
    touching_dst_count: int
    distinct_counterparties: int
    inbound_to_query_src_count: int
    outbound_from_query_src_count: int
    inbound_to_query_src_amount: float
    outbound_from_query_src_amount: float
    repeated_pair_count: int
    aspect_diversity: int
    time_span_minutes: float
    chain_object_count: int
    skip_object_count: int
    richness_score: float
    decision_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_evidence_richness(
    query_event: Event,
    retrieved_events: list[Event],
    *,
    retrieved_aspects: set[str] | None = None,
    retrieved_object_types: list[str] | None = None,
    readiness_threshold: float = 0.45,
) -> EvidenceRichnessReport:
    """Measure whether retrieved history is rich enough for downstream judgment."""

    query_src = _text(query_event.attrs.get("src_account"))
    query_dst = _text(query_event.attrs.get("dst_account"))
    query_entities = {value for value in (query_src, query_dst) if value}

    touching_src_count = 0
    touching_dst_count = 0
    touching_event_count = 0
    distinct_counterparties: set[str] = set()
    inbound_to_query_src_count = 0
    outbound_from_query_src_count = 0
    inbound_to_query_src_amount = 0.0
    outbound_from_query_src_amount = 0.0
    pair_counts: dict[tuple[str, str], int] = {}
    event_times: list[float] = []

    for event in retrieved_events:
        src = _text(event.attrs.get("src_account"))
        dst = _text(event.attrs.get("dst_account"))
        entities = {value for value in (src, dst) if value}
        touched_src = bool(query_src and query_src in entities)
        touched_dst = bool(query_dst and query_dst in entities)
        if touched_src:
            touching_src_count += 1
        if touched_dst:
            touching_dst_count += 1
        if touched_src or touched_dst:
            touching_event_count += 1
        for entity in entities - query_entities:
            distinct_counterparties.add(entity)

        if query_src and dst == query_src:
            inbound_to_query_src_count += 1
            inbound_to_query_src_amount += _amount(event)
        if query_src and src == query_src:
            outbound_from_query_src_count += 1
            outbound_from_query_src_amount += _amount(event)

        if src and dst:
            pair = (src, dst)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        event_time = _event_time(event.time)
        if event_time is not None:
            event_times.append(event_time)

    repeated_pair_count = sum(1 for count in pair_counts.values() if count > 1)
    aspects = set(retrieved_aspects or ())
    object_types = list(retrieved_object_types or [])
    time_span_minutes = 0.0
    if len(event_times) >= 2:
        time_span_minutes = (max(event_times) - min(event_times)) / 60.0

    richness_score = _clip01(
        0.20 * _safe_fraction(touching_event_count, max(1, len(retrieved_events)))
        + 0.20 * _safe_fraction(len(distinct_counterparties), 4)
        + 0.20 * _safe_fraction(inbound_to_query_src_count + outbound_from_query_src_count, 4)
        + 0.15 * _safe_fraction(repeated_pair_count, 3)
        + 0.15 * _safe_fraction(len(aspects), 4)
        + 0.10 * _safe_fraction(time_span_minutes, 180.0)
    )

    return EvidenceRichnessReport(
        query_event_id=query_event.event_id,
        retrieved_event_count=len(retrieved_events),
        touching_event_count=touching_event_count,
        touching_src_count=touching_src_count,
        touching_dst_count=touching_dst_count,
        distinct_counterparties=len(distinct_counterparties),
        inbound_to_query_src_count=inbound_to_query_src_count,
        outbound_from_query_src_count=outbound_from_query_src_count,
        inbound_to_query_src_amount=round(inbound_to_query_src_amount, 2),
        outbound_from_query_src_amount=round(outbound_from_query_src_amount, 2),
        repeated_pair_count=repeated_pair_count,
        aspect_diversity=len(aspects),
        time_span_minutes=round(time_span_minutes, 2),
        chain_object_count=object_types.count("chain"),
        skip_object_count=object_types.count("skip"),
        richness_score=round(richness_score, 4),
        decision_ready=richness_score >= readiness_threshold,
    )


def summarize_richness(reports: list[EvidenceRichnessReport]) -> dict[str, float]:
    """Aggregate evidence-richness reports."""

    if not reports:
        return {
            "queries": 0.0,
            "decision_ready_rate": 0.0,
            "mean_richness_score": 0.0,
            "mean_touching_event_count": 0.0,
            "mean_distinct_counterparties": 0.0,
        }

    return {
        "queries": float(len(reports)),
        "decision_ready_rate": sum(1 for report in reports if report.decision_ready) / len(reports),
        "mean_richness_score": sum(report.richness_score for report in reports) / len(reports),
        "mean_touching_event_count": sum(report.touching_event_count for report in reports) / len(reports),
        "mean_distinct_counterparties": sum(report.distinct_counterparties for report in reports) / len(reports),
    }


def analyze_probe_json(
    probe_path: str | Path,
    csv_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze a saved positive-probe JSON file against the CSV event stream."""

    probe_payload = json.loads(Path(probe_path).read_text(encoding="utf-8"))
    event_lookup = _load_events_by_id(csv_path)
    reports: list[EvidenceRichnessReport] = []

    for row in probe_payload.get("report", []):
        query_event = event_lookup.get(str(row["query_event_id"]))
        if query_event is None:
            continue
        retrieved_events = [
            event_lookup[str(sample["event_id"])]
            for sample in row.get("samples", [])
            if str(sample["event_id"]) in event_lookup
        ]
        report = analyze_evidence_richness(
            query_event,
            retrieved_events,
            retrieved_aspects=set(),
            retrieved_object_types=_object_types_from_probe_row(row),
        )
        reports.append(report)

    payload = {
        "summary": summarize_richness(reports),
        "reports": [report.to_dict() for report in reports],
    }
    if output_path is not None:
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-path", required=True)
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--output-path", required=False, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = analyze_probe_json(args.probe_path, args.csv_path, output_path=args.output_path)
    print(json.dumps(payload, indent=2))
    return 0


def _load_events_by_id(csv_path: str | Path) -> dict[str, Event]:
    path = Path(csv_path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        schema = detect_schema(reader.fieldnames or [])
        rows = [convert_row_to_event(row, schema) for row in reader]
    return {row.event.event_id: row.event for row in rows}


def _text(value: Any) -> str:
    return str(value).strip() if value not in (None, "") else ""


def _amount(event: Event) -> float:
    for key in ("amount", "transaction_amount", "value"):
        value = event.attrs.get(key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _event_time(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_fraction(value: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return min(float(value) / float(denominator), 1.0)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _object_types_from_probe_row(row: dict[str, Any]) -> list[str]:
    object_summaries = list(row.get("object_summaries", []))
    if object_summaries:
        return [str(item.get("kind", "chain")) for item in object_summaries]

    chain_count = int(row.get("chain_objects", 0) or 0)
    skip_count = int(row.get("skip_objects", 0) or 0)
    return ["chain"] * max(chain_count, 0) + ["skip"] * max(skip_count, 0)


if __name__ == "__main__":
    raise SystemExit(main())
