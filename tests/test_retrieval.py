from __future__ import annotations

from dataclasses import dataclass

from timeindex.config import RetrievalConfig, ScoringConfig, TimeIndexConfig
from timeindex.event import ChainSummary, DecisionIntent, Event, EventMetadata, EventQuery, EventRecord, OrdinaryLink, SkipLink
from timeindex.retrieval import DualFrontierRetriever, retrieve


class FakeEventStore:
    def __init__(self, records: dict[str, EventRecord]) -> None:
        self.records = records

    def get(self, event_id: str) -> EventRecord | None:
        return self.records.get(event_id)


class FakeEdgeStore:
    def __init__(self, incoming_map: dict[str, list[OrdinaryLink]]) -> None:
        self.incoming_map = incoming_map

    def incoming(self, event_id: str) -> list[OrdinaryLink]:
        return list(self.incoming_map.get(event_id, []))


class FakeSkipLinkStore:
    def __init__(self, incoming_map: dict[str, list[SkipLink]]) -> None:
        self.incoming_map = incoming_map

    def incoming(self, event_id: str) -> list[SkipLink]:
        return list(self.incoming_map.get(event_id, []))


class FakeChainStore:
    def __init__(self, summaries_by_tail: dict[str, list[ChainSummary]]) -> None:
        self.summaries_by_tail = summaries_by_tail

    def get_for_tail(self, event_id: str) -> list[ChainSummary]:
        return list(self.summaries_by_tail.get(event_id, []))


@dataclass
class FakeIndex:
    event_store: FakeEventStore
    edge_store: FakeEdgeStore
    chain_store: FakeChainStore
    skip_link_store: FakeSkipLinkStore
    config: TimeIndexConfig

    def get_event(self, event_id: str) -> EventRecord | None:
        return self.event_store.get(event_id)


def _record(event_id: str, time: int, aspects: set[str]) -> EventRecord:
    return EventRecord(
        event=Event(event_id=event_id, time=time, event_type="transaction"),
        aspects=set(aspects),
        metadata=EventMetadata(insertion_order=time),
    )


def _build_index() -> FakeIndex:
    records = {
        "e1": _record("e1", 1, {"source_accumulation"}),
        "e2": _record("e2", 2, {"source_accumulation"}),
        "e3": _record("e3", 3, {"source_accumulation"}),
        "e5": _record("e5", 5, {"beneficiary_novelty"}),
        "e6": _record("e6", 6, {"large_transfer"}),
        "e7": _record("e7", 7, {"full_balance_transfer"}),
    }
    edge_store = FakeEdgeStore(
        {
            "e7": [OrdinaryLink(predecessor_id="e6", successor_id="e7", score=0.95)],
            "e6": [OrdinaryLink(predecessor_id="e5", successor_id="e6", score=0.80)],
            "e5": [OrdinaryLink(predecessor_id="e3", successor_id="e5", score=0.75)],
            "e3": [OrdinaryLink(predecessor_id="e2", successor_id="e3", score=0.70)],
            "e2": [OrdinaryLink(predecessor_id="e1", successor_id="e2", score=0.65)],
        }
    )
    chain_store = FakeChainStore(
        {
            "e7": [
                ChainSummary(
                    chain_id="chain:e6:e7",
                    family="transaction",
                    head_id="e6",
                    tail_id="e7",
                    representative_event_ids=["e6", "e7"],
                    aspects={"large_transfer", "full_balance_transfer"},
                    summary="Recent high-value transfer chain",
                    cost=1.0,
                )
            ],
            "e6": [
                ChainSummary(
                    chain_id="chain:e5:e6",
                    family="transaction",
                    head_id="e5",
                    tail_id="e6",
                    representative_event_ids=["e5", "e6"],
                    aspects={"beneficiary_novelty", "large_transfer"},
                    summary="Novel beneficiary followed by large transfer",
                    cost=1.0,
                )
            ],
        }
    )
    skip_link_store = FakeSkipLinkStore(
        {
            "e7": [
                SkipLink(
                    from_id="e1",
                    to_id="e7",
                    skip_value=0.95,
                    aspects={"source_accumulation", "full_balance_transfer"},
                    summary="Early balance buildup explains the final drain",
                    representative_event_ids=["e1", "e2", "e3"],
                    cost=1.0,
                )
            ]
        }
    )
    config = TimeIndexConfig(
        retrieval=RetrievalConfig(return_summaries=True, allow_skip_expansion=True, default_budget=3),
        synthetic=None,  # type: ignore[arg-type]
    )
    config.scoring = ScoringConfig(retrieval_stop_threshold=0.01)
    return FakeIndex(
        event_store=FakeEventStore(records),
        edge_store=edge_store,
        chain_store=chain_store,
        skip_link_store=skip_link_store,
        config=config,
    )


def _chain_only_baseline(index: FakeIndex, query_event_id: str, budget: int) -> list[str]:
    current_id = query_event_id
    collected: list[str] = []
    remaining = budget
    while remaining > 0:
        links = index.edge_store.incoming(current_id)
        if not links:
            break
        link = links[0]
        collected.append(link.predecessor_id)
        current_id = link.predecessor_id
        remaining -= 1
    return collected


def test_retrieval_respects_budget() -> None:
    index = _build_index()
    intent = DecisionIntent(aspects={"source_accumulation", "full_balance_transfer"})

    results = retrieve(index, "e7", intent, budget=2)

    assert results
    assert sum(item.cost for item in results) <= 2


def test_retrieval_returns_ordinary_evidence() -> None:
    index = _build_index()
    intent = DecisionIntent(aspects={"large_transfer", "beneficiary_novelty"})
    retriever = DualFrontierRetriever(
        index.event_store,
        index.edge_store,
        index.chain_store,
        index.skip_link_store,
        index.config.retrieval,
    )

    results = retriever.retrieve(EventQuery(event=index.get_event("e7").event, intent=intent, budget=2))

    assert any(result.object_id.startswith("chain:") for result in results)
    assert any("large_transfer" in result.aspects for result in results)


def test_retrieval_returns_skip_evidence() -> None:
    index = _build_index()
    intent = DecisionIntent(aspects={"source_accumulation", "full_balance_transfer"})

    results = retrieve(index, "e7", intent, budget=3)

    skip_results = [result for result in results if result.object_id.startswith("skip:")]
    assert skip_results
    assert skip_results[0].summary == "Early balance buildup explains the final drain"


def test_dual_frontier_beats_chain_only_local_myopia() -> None:
    index = _build_index()
    intent = DecisionIntent(aspects={"source_accumulation", "full_balance_transfer"})

    baseline_event_ids = _chain_only_baseline(index, "e7", budget=2)
    results = retrieve(index, "e7", intent, budget=3)
    skip_results = [result for result in results if result.object_id.startswith("skip:")]

    assert "e1" not in baseline_event_ids
    assert skip_results
    assert skip_results[0].summary.startswith("Early balance buildup")
