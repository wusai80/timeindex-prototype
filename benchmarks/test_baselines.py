from __future__ import annotations

from dataclasses import dataclass, field

from benchmarks.baselines import (
    chain_only_retrieval,
    flow_chain_retrieval,
    nearest_neighbor_retrieval,
    random_retrieval,
    recent_window_retrieval,
    same_entity_window_retrieval,
)
from timeindex.event import Event, EventRecord, OrdinaryLink
from timeindex.synthetic import synthetic_transaction_stream


def test_every_baseline_respects_budget() -> None:
    events = synthetic_transaction_stream()
    index = _FakeIndex(events=events)

    assert len(recent_window_retrieval(events, "e7", budget=2, window=5)) <= 2
    assert len(same_entity_window_retrieval(events, "e7", budget=2, window=6)) <= 2
    assert len(flow_chain_retrieval(events, "e7", budget=2, max_hops=3)) <= 2
    assert len(nearest_neighbor_retrieval(events, "e7", budget=2)) <= 2
    assert len(chain_only_retrieval(index, "e7", budget=2)) <= 2
    assert len(random_retrieval(events, "e7", budget=2, seed=7)) <= 2


def test_no_baseline_returns_future_events() -> None:
    events = synthetic_transaction_stream()
    query = events[-1]
    index = _FakeIndex(events=events)

    results = [
        recent_window_retrieval(events, query, budget=3, window=5),
        same_entity_window_retrieval(events, query, budget=3, window=6),
        flow_chain_retrieval(events, query, budget=3, max_hops=3),
        nearest_neighbor_retrieval(events, query, budget=3),
        chain_only_retrieval(index, query.event_id, budget=3),
        random_retrieval(events, query, budget=3, seed=9),
    ]

    future_ids = {event.event_id for event in events if event.time > query.time}
    future_ids.add(query.event_id)
    for result in results:
        returned_ids = {event_id for item in result for event_id in item["event_ids"]}
        assert returned_ids.isdisjoint(future_ids)


def test_chain_only_does_not_use_skip_links() -> None:
    events = synthetic_transaction_stream()
    index = _FakeIndex(
        events=events,
        incoming={
            "e7": [OrdinaryLink(predecessor_id="e6", successor_id="e7", score=0.9)],
            "e6": [OrdinaryLink(predecessor_id="e5", successor_id="e6", score=0.8)],
            "e5": [OrdinaryLink(predecessor_id="e4", successor_id="e5", score=0.7)],
        },
    )

    selected = chain_only_retrieval(index, "e7", budget=3)

    assert [item["event_ids"][0] for item in selected] == ["e6", "e5", "e4"]
    assert index.skip_link_calls == 0


def test_nearest_neighbor_is_deterministic() -> None:
    events = synthetic_transaction_stream()

    first = nearest_neighbor_retrieval(events, "e7", budget=3)
    second = nearest_neighbor_retrieval(events, "e7", budget=3)

    assert first == second


def test_flow_chain_returns_interacting_history_across_different_accounts() -> None:
    events = [
        Event(event_id="e1", time=1, event_type="transfer", attrs={"src_account": "X", "dst_account": "A"}),
        Event(event_id="e2", time=2, event_type="transfer", attrs={"src_account": "A", "dst_account": "B"}),
        Event(event_id="e3", time=3, event_type="transfer", attrs={"src_account": "B", "dst_account": "C"}),
        Event(event_id="q", time=4, event_type="transfer", attrs={"src_account": "C", "dst_account": "D"}),
    ]

    selected = flow_chain_retrieval(events, "q", budget=3, max_hops=3)

    assert [item["event_ids"][0] for item in selected] == ["e3", "e2", "e1"]


@dataclass
class _FakeIndex:
    events: list[Event]
    incoming: dict[str, list[OrdinaryLink]] = field(default_factory=dict)
    skip_link_calls: int = 0

    def ordinary_links(self, event_id: str) -> list[OrdinaryLink]:
        return list(self.incoming.get(event_id, []))

    def skip_links(self, event_id: str) -> list[object]:
        self.skip_link_calls += 1
        return [object()]

    def get_event(self, event_id: str) -> EventRecord | None:
        for event in self.events:
            if event.event_id == event_id:
                return EventRecord(event=event)
        return None
