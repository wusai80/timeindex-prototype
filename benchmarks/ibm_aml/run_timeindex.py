"""Run the TimeIndex benchmark on IBM AML event streams."""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import random
import time
from pathlib import Path
from typing import Any

from timeindex import DecisionIntent, Event, EventQuery, EventRecord
from timeindex.construction import TimeIndex
from timeindex.retrieval import retrieve

from benchmarks.ibm_aml.configs import DEFAULT_BUDGETS, build_benchmark_config


def run_benchmark(
    csv_path: str | Path,
    *,
    output_dir: str | Path = "outputs/ibm_aml",
    budgets: list[int] | tuple[int, ...] = DEFAULT_BUDGETS,
    query_laundering_only: bool = True,
    negative_query_sample_size: int = 0,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Run IBM AML retrieval and write output artifacts."""

    config = build_benchmark_config(
        csv_path,
        output_dir=output_dir,
        budgets=budgets,
        query_laundering_only=query_laundering_only,
        negative_query_sample_size=negative_query_sample_size,
        random_seed=random_seed,
    )
    output_path = config.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_events = load_ibm_aml_events(config.csv_path)
    gold_evidence = build_gold_supporting_evidence(dataset_events)
    dataset_events.sort(key=_event_sort_key)

    index = TimeIndex(config.timeindex)
    event_time_by_id = {event.event_id: _sortable_time(event.time) for event in dataset_events}
    dataset_profile = _build_dataset_profile(dataset_events, gold_evidence)
    selected_query_ids = _select_query_event_ids(dataset_events, config)

    results_path = output_path / "retrieval_results.jsonl"
    result_count = 0
    latency_values: list[float] = []

    with results_path.open("w", encoding="utf-8") as handle:
        for event in dataset_events:
            index.insert(event)
            if event.event_id not in selected_query_ids:
                continue

            intent = _build_query_intent(event)
            gold_ids = _normalize_gold_ids(gold_evidence.get(event.event_id, ()), event.event_id, event_time_by_id)
            for budget in config.budgets:
                query = EventQuery(event=event, intent=intent, budget=budget)
                start = time.perf_counter()
                evidence_objects = list(retrieve(index, event.event_id, intent, budget))
                latency_ms = (time.perf_counter() - start) * 1000.0
                latency_values.append(latency_ms)
                result = _serialize_result(
                    query_event=event,
                    evidence_objects=evidence_objects,
                    budget=budget,
                    gold_event_ids=gold_ids,
                    latency_ms=latency_ms,
                    index=index,
                    event_time_by_id=event_time_by_id,
                )
                handle.write(json.dumps(result, sort_keys=True) + "\n")
                result_count += 1

    _write_json(output_path / "config.json", config.to_dict())
    _write_json(output_path / "dataset_profile.json", dataset_profile)
    run_summary = {
        "csv_path": str(config.csv_path),
        "output_dir": str(output_path),
        "query_count": len(selected_query_ids),
        "retrieval_count": result_count,
        "budgets": list(config.budgets),
        "mean_latency_ms": (sum(latency_values) / len(latency_values)) if latency_values else 0.0,
        "max_latency_ms": max(latency_values) if latency_values else 0.0,
        "query_laundering_only": config.query_laundering_only,
        "negative_query_sample_size": config.negative_query_sample_size,
    }
    _write_json(output_path / "run_summary.json", run_summary)
    return {
        "config": config.to_dict(),
        "dataset_profile": dataset_profile,
        "run_summary": run_summary,
        "results_path": str(results_path),
    }


def load_ibm_aml_events(csv_path: str | Path) -> list[Event]:
    """Load IBM AML events through adapter.py, with a lightweight fallback."""

    adapter_module = _optional_module("benchmarks.ibm_aml.adapter", "adapter")
    if adapter_module is not None:
        stream_events = getattr(adapter_module, "stream_events", None)
        if callable(stream_events):
            schema = _resolve_schema(csv_path)
            loaded = stream_events(Path(csv_path), schema, sort_by_time=True)
            return [_coerce_event(item) for item in loaded]
        for function_name in (
            "load_ibm_aml_csv",
            "load_csv",
            "load_events",
            "read_events",
        ):
            loader = getattr(adapter_module, function_name, None)
            if callable(loader):
                loaded = loader(Path(csv_path))
                return [_coerce_event(item) for item in loaded]
        adapter_cls = getattr(adapter_module, "IBMAMLAdapter", None)
        if adapter_cls is not None:
            adapter = adapter_cls()
            for method_name in ("load_csv", "load_events", "read_events"):
                method = getattr(adapter, method_name, None)
                if callable(method):
                    loaded = method(Path(csv_path))
                    return [_coerce_event(item) for item in loaded]

    return _load_events_from_csv_fallback(Path(csv_path))


def build_gold_supporting_evidence(events: list[Event]) -> dict[str, list[str]]:
    """Build gold evidence using evidence.py, with a lightweight fallback."""

    evidence_module = _optional_module("benchmarks.ibm_aml.evidence", "evidence")
    if evidence_module is not None:
        for function_name in (
            "build_gold_supporting_evidence",
            "build_gold_evidence",
            "build_gold_map",
        ):
            builder = getattr(evidence_module, function_name, None)
            if callable(builder):
                built = _call_gold_builder(builder, events)
                return _normalize_gold_mapping(built)
        evidence_cls = getattr(evidence_module, "EvidenceBuilder", None)
        if evidence_cls is not None:
            builder = evidence_cls()
            for method_name in ("build_gold_supporting_evidence", "build_gold_evidence", "build"):
                method = getattr(builder, method_name, None)
                if callable(method):
                    built = _call_gold_builder(method, events)
                    return _normalize_gold_mapping(built)

    return _fallback_gold_evidence(events)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TimeIndex on IBM AML data.")
    parser.add_argument("csv_path")
    parser.add_argument("--output-dir", default="outputs/ibm_aml")
    parser.add_argument("--negative-query-sample-size", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--budgets",
        nargs="*",
        type=int,
        default=list(DEFAULT_BUDGETS),
    )
    parser.add_argument(
        "--include-non-laundering-queries",
        action="store_true",
        help="Query non-laundering events too when no negative sampling limit is used.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_benchmark(
        args.csv_path,
        output_dir=args.output_dir,
        budgets=args.budgets,
        query_laundering_only=not args.include_non_laundering_queries,
        negative_query_sample_size=args.negative_query_sample_size,
        random_seed=args.random_seed,
    )
    return 0


def _optional_module(*names: str) -> Any | None:
    for name in names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
    return None


def _coerce_event(item: Any) -> Event:
    if isinstance(item, Event):
        return item
    if isinstance(item, EventRecord):
        return item.event
    if isinstance(item, dict):
        attrs = dict(item.get("attrs", {}))
        ctx = dict(item.get("ctx", {}))
        reserved = {"event_id", "time", "event_type", "attrs", "ctx", "text", "label"}
        for key, value in item.items():
            if key not in reserved:
                attrs.setdefault(key, value)
        return Event(
            event_id=str(item["event_id"]),
            time=item["time"],
            event_type=str(item.get("event_type", "transaction")),
            attrs=attrs,
            ctx=ctx,
            text=item.get("text"),
            label=None if item.get("label") is None else str(item["label"]),
        )
    raise TypeError(f"Unsupported adapter output: {type(item)!r}")


def _load_events_from_csv_fallback(csv_path: Path) -> list[Event]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        events: list[Event] = []
        for row_number, row in enumerate(reader):
            raw = {key: value for key, value in row.items() if key is not None}
            event_id = str(raw.pop("event_id", raw.pop("tx_id", row_number)))
            raw_time = raw.pop("timestamp", raw.pop("time", row_number))
            event_type = str(raw.pop("event_type", raw.pop("type", "transaction")))
            label = raw.get("label") or raw.get("laundering_label") or raw.get("is_laundering")
            attrs = {key: _maybe_number(value) for key, value in raw.items()}
            events.append(
                Event(
                    event_id=event_id,
                    time=_maybe_number(raw_time),
                    event_type=event_type,
                    attrs=attrs,
                    label=None if label is None else str(label),
                )
            )
    return events


def _fallback_gold_evidence(events: list[Event]) -> dict[str, list[str]]:
    predecessors_by_sender: dict[str, list[str]] = {}
    gold: dict[str, list[str]] = {}
    for event in sorted(events, key=_event_sort_key):
        sender = _first_non_empty(
            event.attrs.get("account_id"),
            event.attrs.get("sender_account"),
            event.attrs.get("source_account"),
        )
        prior = list(predecessors_by_sender.get(str(sender), ())) if sender is not None else []
        gold[event.event_id] = prior[-3:]
        if sender is not None:
            predecessors_by_sender.setdefault(str(sender), []).append(event.event_id)
    return gold


def _normalize_gold_mapping(mapping: Any) -> dict[str, list[str]]:
    if isinstance(mapping, dict):
        normalized: dict[str, list[str]] = {}
        for key, value in mapping.items():
            if isinstance(value, dict):
                event_ids = value.get("event_ids", value.get("gold_event_ids", ()))
            else:
                event_ids = value
            normalized[str(key)] = [str(item) for item in event_ids]
        return normalized
    raise TypeError("Gold evidence builder must return a mapping.")


def _call_gold_builder(builder: Any, events: list[Event]) -> Any:
    signature = inspect.signature(builder)
    parameters = list(signature.parameters)
    if len(parameters) <= 1:
        return builder(events)
    return builder(
        [_event_to_mapping(event) for event in events],
        policy="same-entity-laundering-window",
        window=50,
        max_hops=2,
        amount_threshold=0.8,
    )


def _build_dataset_profile(events: list[Event], gold_evidence: dict[str, list[str]]) -> dict[str, Any]:
    laundering_events = [event for event in events if _is_laundering_label(event.label)]
    return {
        "event_count": len(events),
        "laundering_event_count": len(laundering_events),
        "non_laundering_event_count": len(events) - len(laundering_events),
        "event_type_counts": _count_values(event.event_type for event in events),
        "gold_query_count": sum(1 for event_id in gold_evidence if gold_evidence.get(event_id)),
        "min_time": min((_sortable_time(event.time) for event in events), default=0),
        "max_time": max((_sortable_time(event.time) for event in events), default=0),
    }


def _resolve_schema(csv_path: str | Path) -> Any:
    schema_module = _optional_module("benchmarks.ibm_aml.schema", "schema")
    if schema_module is None:
        return {"source_file": str(csv_path), "dataset_name": Path(csv_path).stem}

    header = _read_header(Path(csv_path))
    for function_name in ("detect_schema", "infer_schema", "resolve_schema"):
        resolver = getattr(schema_module, function_name, None)
        if callable(resolver):
            return resolver(header)
    return {
        "source_file": str(csv_path),
        "dataset_name": Path(csv_path).stem,
    }


def _select_query_event_ids(events: list[Event], config: Any) -> set[str]:
    laundering_ids = [event.event_id for event in events if _is_laundering_label(event.label)]
    non_laundering_ids = [event.event_id for event in events if not _is_laundering_label(event.label)]
    selected = list(laundering_ids) if config.query_laundering_only else [event.event_id for event in events]
    if config.negative_query_sample_size > 0 and non_laundering_ids:
        rng = random.Random(config.random_seed)
        sample_size = min(config.negative_query_sample_size, len(non_laundering_ids))
        selected.extend(rng.sample(non_laundering_ids, sample_size))
    return set(selected)


def _event_to_mapping(event: Event) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "timestamp": event.time,
        "src_account": _first_non_empty(
            event.attrs.get("src_account"),
            event.attrs.get("source_account"),
            event.attrs.get("account_id"),
        ),
        "dst_account": _first_non_empty(
            event.attrs.get("dst_account"),
            event.attrs.get("beneficiary_id"),
            event.attrs.get("beneficiary_account"),
        ),
        "amount": _first_non_empty(
            event.attrs.get("amount"),
            event.attrs.get("transaction_amount"),
            event.attrs.get("value"),
            0.0,
        ),
        "is_laundering": _is_laundering_label(event.label),
        "pattern_id": _first_non_empty(
            event.attrs.get("pattern_id"),
            event.attrs.get("typology_id"),
            event.attrs.get("group_id"),
        ),
    }


def _build_query_intent(event: Event) -> DecisionIntent:
    attrs = event.attrs
    aspects: set[str] = set()
    if attrs.get("amount") or attrs.get("transaction_amount") or attrs.get("value"):
        aspects.add("large_transfer")
    if attrs.get("is_new_beneficiary") or attrs.get("new_beneficiary"):
        aspects.add("beneficiary_novelty")
    if attrs.get("burst_count") or attrs.get("recent_event_count"):
        aspects.add("temporal_burst")
    if not aspects:
        aspects.add("generic_evidence")
    return DecisionIntent(aspects=aspects, name="ibm_aml_query")


def _serialize_result(
    *,
    query_event: Event,
    evidence_objects: list[Any],
    budget: int,
    gold_event_ids: list[str],
    latency_ms: float,
    index: TimeIndex,
    event_time_by_id: dict[str, float],
) -> dict[str, Any]:
    query_time = _sortable_time(query_event.time)
    causal_event_ids: list[str] = []
    retrieved_aspects: set[str] = set()
    retrieved_object_types: list[str] = []

    for evidence in evidence_objects:
        retrieved_aspects.update(getattr(evidence, "aspects", ()))
        retrieved_object_types.append(_object_type(evidence))
        for event_id in getattr(evidence, "event_ids", ()):
            if event_id == query_event.event_id:
                continue
            if event_time_by_id.get(event_id, float("inf")) >= query_time:
                continue
            if event_id not in causal_event_ids:
                causal_event_ids.append(event_id)

    return {
        "query_event_id": query_event.event_id,
        "query_label": query_event.label,
        "budget": int(budget),
        "retrieved_event_ids": causal_event_ids[:budget],
        "retrieved_object_types": retrieved_object_types,
        "retrieved_aspects": sorted(retrieved_aspects),
        "gold_event_ids": gold_event_ids,
        "latency_ms": latency_ms,
        "index_stats": {
            "indexed_event_count": len(index.event_store.list()),
            "query_incoming_ordinary_links": len(index.ordinary_links(query_event.event_id)),
            "query_incoming_skip_links": len(index.skip_links(query_event.event_id)),
            "query_chain_summaries": len(index.chains(query_event.event_id)),
        },
    }


def _object_type(evidence: Any) -> str:
    object_id = str(getattr(evidence, "object_id", "evidence"))
    if object_id.startswith("skip:"):
        return "skip"
    if object_id.startswith("ordinary:"):
        return "ordinary"
    return "chain"


def _normalize_gold_ids(gold_ids: list[str] | tuple[str, ...] | set[str], query_event_id: str, event_time_by_id: dict[str, float]) -> list[str]:
    query_time = event_time_by_id.get(query_event_id, float("inf"))
    normalized: list[str] = []
    for event_id in gold_ids:
        event_id_text = str(event_id)
        if event_id_text == query_event_id:
            continue
        if event_time_by_id.get(event_id_text, float("inf")) >= query_time:
            continue
        if event_id_text not in normalized:
            normalized.append(event_id_text)
    return normalized


def _event_sort_key(event: Event) -> tuple[float, str]:
    return (_sortable_time(event.time), event.event_id)


def _sortable_time(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        digits = "".join(character for character in str(value) if character.isdigit())
        return float(digits) if digits else 0.0


def _read_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def _is_laundering_label(label: Any) -> bool:
    if isinstance(label, bool):
        return label
    if isinstance(label, (int, float)):
        return float(label) > 0
    normalized = str(label or "").strip().lower()
    return normalized in {"1", "true", "yes", "y", "laundering", "suspicious", "fraud"}


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _maybe_number(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped == "":
        return value
    try:
        if "." in stripped:
            return float(stripped)
        return int(stripped)
    except ValueError:
        return value


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


if __name__ == "__main__":
    raise SystemExit(main())
