from __future__ import annotations

from types import SimpleNamespace

from benchmarks.ibm_aml.run_sqlite_deepseek_sample import (
    DEFAULT_POSITIVE_MIN_INSERTION_ORDER,
    _collect_retrieval,
    _resolve_cutoffs,
    _should_expand_budget,
)
from timeindex.event import Event, EventMetadata, EventRecord, EvidenceObject


def test_should_expand_budget_for_thin_retrieval() -> None:
    assert _should_expand_budget(
        ["e1", "e2"],
        ["large_transfer"],
        adaptive_budget=16,
        base_budget=8,
        adaptive_min_events=4,
        adaptive_min_aspects=2,
    )


def test_should_expand_budget_for_low_aspect_diversity() -> None:
    assert _should_expand_budget(
        ["e1", "e2", "e3", "e4"],
        ["large_transfer"],
        adaptive_budget=16,
        base_budget=8,
        adaptive_min_events=4,
        adaptive_min_aspects=2,
    )


def test_should_not_expand_when_budget_is_not_larger() -> None:
    assert not _should_expand_budget(
        ["e1", "e2"],
        ["large_transfer"],
        adaptive_budget=8,
        base_budget=8,
        adaptive_min_events=4,
        adaptive_min_aspects=2,
    )


def test_should_not_expand_for_rich_retrieval() -> None:
    assert not _should_expand_budget(
        ["e1", "e2", "e3", "e4", "e5"],
        ["large_transfer", "beneficiary_novelty"],
        adaptive_budget=16,
        base_budget=8,
        adaptive_min_events=4,
        adaptive_min_aspects=2,
    )


def test_resolve_cutoffs_supports_late_positives_only() -> None:
    assert _resolve_cutoffs(
        min_insertion_order=None,
        positive_min_insertion_order=4_000_000,
        negative_min_insertion_order=None,
    ) == (4_000_000, None)


def test_resolve_cutoffs_uses_default_late_positive_sampler() -> None:
    assert _resolve_cutoffs(
        min_insertion_order=None,
        positive_min_insertion_order=None,
        negative_min_insertion_order=None,
    ) == (DEFAULT_POSITIVE_MIN_INSERTION_ORDER, None)


def test_collect_retrieval_keeps_structured_objects_aligned_with_prompt_events(monkeypatch) -> None:
    query_event = Event(
        event_id="q1",
        time=10,
        event_type="wire",
        attrs={"src_account": "A", "dst_account": "B", "amount": 900.0},
    )
    support_a = Event(
        event_id="e1",
        time=7,
        event_type="deposit",
        attrs={"src_account": "X", "dst_account": "A", "amount": 500.0},
    )
    support_b = Event(
        event_id="e2",
        time=8,
        event_type="transfer",
        attrs={"src_account": "A", "dst_account": "Y", "amount": 700.0},
    )
    support_c = Event(
        event_id="e3",
        time=9,
        event_type="transfer",
        attrs={"src_account": "Y", "dst_account": "B", "amount": 650.0},
    )

    class StubIndex:
        def __init__(self) -> None:
            self.records = {
                "q1": EventRecord(event=query_event, metadata=EventMetadata(insertion_order=10)),
                "e1": EventRecord(event=support_a, metadata=EventMetadata(insertion_order=7)),
                "e2": EventRecord(event=support_b, metadata=EventMetadata(insertion_order=8)),
                "e3": EventRecord(event=support_c, metadata=EventMetadata(insertion_order=9)),
            }

        def get_event(self, event_id: str) -> EventRecord | None:
            return self.records.get(event_id)

    fake_objects = [
        EvidenceObject(
            object_id="skip:e0->q1",
            event_ids=["e1", "e2", "e3"],
            aspects={"generic_evidence", "large_transfer"},
            summary="skip bridge",
            cost=4.75,
        )
    ]

    monkeypatch.setattr(
        "benchmarks.ibm_aml.run_sqlite_deepseek_sample.retrieve",
        lambda index, query_id, intent, budget: fake_objects,
    )

    payload = _collect_retrieval(StubIndex(), "q1", 2)

    assert payload["event_ids"] == ["e1", "e2"]
    assert [event.event_id for event in payload["events"]] == ["e1", "e2"]
    assert len(payload["objects"]) == 1
    assert payload["objects"][0].event_ids == ["e1", "e2"]
    assert payload["objects"][0].summary == "skip bridge"
