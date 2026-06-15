from __future__ import annotations

from dataclasses import dataclass

from timeindex.evaluation import evidence_recall
from timeindex.event import Event, OrdinaryLink
from timeindex.synthetic import (
    baseline_chain_only,
    baseline_nearest_neighbor,
    baseline_recent_window,
    synthetic_log_stream,
    synthetic_transaction_stream,
)


def test_synthetic_transaction_stream_has_expected_gold_evidence() -> None:
    events = synthetic_transaction_stream()

    assert [event.event_id for event in events] == ["e1", "e2", "e3", "e4", "e5", "e6", "e7"]
    query = events[-1]
    gold_ids = {"e1", "e2", "e3", "e5", "e6"}

    assert query.event_id == "e7"
    assert query.label == "full_balance_transfer"
    assert query.attrs["beneficiary_account"] == "B"
    assert query.attrs["balance_after"] == 0.0
    assert gold_ids.issubset({event.event_id for event in events[:-1]})


def test_synthetic_log_stream_is_deterministic() -> None:
    events = synthetic_log_stream()

    assert [event.event_id for event in events] == ["l1", "l2", "l3", "l4", "l5"]
    assert [event.label for event in events] == [
        "deployment",
        "config_change",
        "upstream_error",
        "resource_saturation",
        "timeout",
    ]


def test_baseline_recent_window_returns_latest_predecessors() -> None:
    events = synthetic_transaction_stream()

    selected = baseline_recent_window(events, "e7", budget=3)

    assert [event.event_id for event in selected] == ["e6", "e5", "e4"]


def test_baseline_nearest_neighbor_prefers_beneficiary_and_account_overlap() -> None:
    events = synthetic_transaction_stream()

    selected = baseline_nearest_neighbor(events, "e7", budget=3)
    selected_ids = [event.event_id for event in selected]

    assert "e5" in selected_ids
    assert "e6" in selected_ids
    assert "e7" not in selected_ids


def test_baseline_chain_only_walks_ordinary_predecessors_only() -> None:
    events = synthetic_transaction_stream()
    index = _FakeIndex(
        events=events,
        incoming={
            "e7": [OrdinaryLink(predecessor_id="e6", successor_id="e7", score=0.90)],
            "e6": [OrdinaryLink(predecessor_id="e5", successor_id="e6", score=0.80)],
            "e5": [OrdinaryLink(predecessor_id="e4", successor_id="e5", score=0.70)],
        },
    )

    selected = baseline_chain_only(index, "e7", budget=3)

    assert [event.event_id for event in selected] == ["e6", "e5", "e4"]


def test_evidence_recall_handles_events_and_ids() -> None:
    events = synthetic_transaction_stream()
    gold_ids = {"e1", "e2", "e3", "e5", "e6"}

    recall = evidence_recall([events[0], events[1], "e5"], gold_ids)

    assert recall == 3 / 5


@dataclass
class _FakeIndex:
    events: list[Event]
    incoming: dict[str, list[OrdinaryLink]]

    def ordinary_links(self, event_id: str) -> list[OrdinaryLink]:
        return list(self.incoming.get(event_id, []))

    def get_event(self, event_id: str) -> Event | None:
        for event in self.events:
            if event.event_id == event_id:
                return event
        return None
