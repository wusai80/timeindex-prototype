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


def _lanl_record(
    event_id: str,
    time: int,
    *,
    src_user: str,
    src_host: str,
    dst_host: str,
    aspects: set[str],
    auth_type: str = "NTLM",
) -> EventRecord:
    return EventRecord(
        event=Event(
            event_id=event_id,
            time=time,
            event_type="auth",
            attrs={
                "src_user": src_user,
                "src_computer": src_host,
                "dst_computer": dst_host,
                "auth_type": auth_type,
            },
        ),
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
    assert skip_results[0].event_ids[0] == "e1"


def test_retrieval_can_discover_skip_from_expanded_ordinary_predecessor() -> None:
    records = {
        "s1": _record("s1", 1, {"source_accumulation"}),
        "p1": _record("p1", 2, {"beneficiary_novelty"}),
        "q1": _record("q1", 3, {"full_balance_transfer"}),
    }
    edge_store = FakeEdgeStore(
        {
            "q1": [OrdinaryLink(predecessor_id="p1", successor_id="q1", score=0.95)],
        }
    )
    chain_store = FakeChainStore({})
    skip_link_store = FakeSkipLinkStore(
        {
            "p1": [
                SkipLink(
                    from_id="s1",
                    to_id="p1",
                    skip_value=0.9,
                    aspects={"source_accumulation", "full_balance_transfer"},
                    summary="Earlier buildup attached to the predecessor",
                    representative_event_ids=["s1", "p1"],
                    cost=1.0,
                )
            ]
        }
    )
    index = FakeIndex(
        event_store=FakeEventStore(records),
        edge_store=edge_store,
        chain_store=chain_store,
        skip_link_store=skip_link_store,
        config=TimeIndexConfig(
            retrieval=RetrievalConfig(return_summaries=True, allow_skip_expansion=True, default_budget=3),
            synthetic=None,  # type: ignore[arg-type]
        ),
    )
    index.config.scoring = ScoringConfig(retrieval_stop_threshold=0.01)
    intent = DecisionIntent(aspects={"source_accumulation", "full_balance_transfer"})

    results = retrieve(index, "q1", intent, budget=2)

    skip_results = [result for result in results if result.object_id.startswith("skip:")]
    assert skip_results
    assert skip_results[0].summary == "Earlier buildup attached to the predecessor"
    assert "s1" in skip_results[0].event_ids


def test_retrieval_demotes_generic_redundant_skip_candidates() -> None:
    records = {
        "s1": _record("s1", 1, {"generic_evidence"}),
        "p1": _record("p1", 2, {"large_transfer"}),
        "q1": _record("q1", 3, {"large_transfer"}),
    }
    edge_store = FakeEdgeStore(
        {
            "q1": [OrdinaryLink(predecessor_id="p1", successor_id="q1", score=0.95)],
        }
    )
    chain_store = FakeChainStore({})
    skip_link_store = FakeSkipLinkStore(
        {
            "p1": [
                SkipLink(
                    from_id="s1",
                    to_id="p1",
                    skip_value=0.9,
                    aspects={"generic_evidence", "large_transfer"},
                    summary="Generic redundant skip",
                    representative_event_ids=["s1", "p1"],
                    cost=1.25,
                )
            ]
        }
    )
    index = FakeIndex(
        event_store=FakeEventStore(records),
        edge_store=edge_store,
        chain_store=chain_store,
        skip_link_store=skip_link_store,
        config=TimeIndexConfig(
            retrieval=RetrievalConfig(return_summaries=True, allow_skip_expansion=True, default_budget=1),
            synthetic=None,  # type: ignore[arg-type]
        ),
    )
    index.config.scoring = ScoringConfig(retrieval_stop_threshold=0.01)
    intent = DecisionIntent(aspects={"large_transfer"})

    results = retrieve(index, "q1", intent, budget=1)

    assert results
    assert all(not result.object_id.startswith("skip:") for result in results)


def test_dual_frontier_beats_chain_only_local_myopia() -> None:
    index = _build_index()
    intent = DecisionIntent(aspects={"source_accumulation", "full_balance_transfer"})

    baseline_event_ids = _chain_only_baseline(index, "e7", budget=2)
    results = retrieve(index, "e7", intent, budget=3)
    skip_results = [result for result in results if result.object_id.startswith("skip:")]

    assert "e1" not in baseline_event_ids
    assert skip_results
    assert skip_results[0].summary.startswith("Early balance buildup")


def test_retrieval_excludes_query_event_and_future_events() -> None:
    records = {
        "past": _record("past", 1, {"source_accumulation"}),
        "future": _record("future", 9, {"source_accumulation"}),
        "q": _record("q", 5, {"full_balance_transfer"}),
    }
    edge_store = FakeEdgeStore(
        {
            "q": [
                OrdinaryLink(predecessor_id="past", successor_id="q", score=0.9),
                OrdinaryLink(predecessor_id="future", successor_id="q", score=0.8),
            ]
        }
    )
    chain_store = FakeChainStore(
        {
            "q": [
                ChainSummary(
                    chain_id="chain:past:q",
                    family="transaction",
                    head_id="past",
                    tail_id="q",
                    representative_event_ids=["past", "q", "future"],
                    aspects={"source_accumulation", "full_balance_transfer"},
                    summary="Mixed chain with invalid members",
                    cost=1.0,
                )
            ]
        }
    )
    skip_link_store = FakeSkipLinkStore(
        {
            "q": [
                SkipLink(
                    from_id="future",
                    to_id="q",
                    skip_value=0.95,
                    aspects={"source_accumulation"},
                    summary="Future skip should be filtered",
                    representative_event_ids=["future"],
                    cost=1.0,
                )
            ]
        }
    )
    index = FakeIndex(
        event_store=FakeEventStore(records),
        edge_store=edge_store,
        chain_store=chain_store,
        skip_link_store=skip_link_store,
        config=TimeIndexConfig(
            retrieval=RetrievalConfig(return_summaries=True, allow_skip_expansion=True, default_budget=3),
            synthetic=None,  # type: ignore[arg-type]
        ),
    )
    index.config.scoring = ScoringConfig(retrieval_stop_threshold=0.01)

    results = retrieve(index, "q", DecisionIntent(aspects={"source_accumulation"}), budget=3)

    retrieved_ids = [event_id for result in results for event_id in result.event_ids]
    assert "q" not in retrieved_ids
    assert "future" not in retrieved_ids
    assert "past" in retrieved_ids


def test_lanl_retrieval_enriches_fanout_objects_with_specific_aspects() -> None:
    records = {
        "p1": _lanl_record("p1", 1, src_user="u1", src_host="c1", dst_host="c2", aspects={"credential_reuse"}),
        "p2": _lanl_record("p2", 2, src_user="u1", src_host="c1", dst_host="c3", aspects={"credential_reuse"}),
        "q": _lanl_record("q", 3, src_user="u1", src_host="c1", dst_host="c4", aspects={"new_host_access"}),
    }
    edge_store = FakeEdgeStore(
        {
            "q": [OrdinaryLink(predecessor_id="p2", successor_id="q", score=0.95)],
            "p2": [OrdinaryLink(predecessor_id="p1", successor_id="p2", score=0.90)],
        }
    )
    chain_store = FakeChainStore(
        {
            "q": [
                ChainSummary(
                    chain_id="chain:p1:q",
                    family="auth",
                    head_id="p2",
                    tail_id="q",
                    representative_event_ids=["p2", "p1"],
                    aspects={"credential_reuse"},
                    summary="Prior LANL auth chain",
                    cost=1.0,
                )
            ]
        }
    )
    skip_link_store = FakeSkipLinkStore({})
    index = FakeIndex(
        event_store=FakeEventStore(records),
        edge_store=edge_store,
        chain_store=chain_store,
        skip_link_store=skip_link_store,
        config=TimeIndexConfig(
            retrieval=RetrievalConfig(return_summaries=True, allow_skip_expansion=True, default_budget=3),
            synthetic=None,  # type: ignore[arg-type]
        ),
    )
    index.config.scoring = ScoringConfig(retrieval_stop_threshold=0.01)

    results = retrieve(index, "q", DecisionIntent(aspects={"credential_reuse"}), budget=3)

    assert results
    assert any("lanl_source_host_fanout" in result.aspects for result in results)
    fanout_result = next(result for result in results if "lanl_source_host_fanout" in result.aspects)
    assert "pattern=lanl_source_host_fanout" in fanout_result.summary
    assert "query_host_touch=true" in fanout_result.summary


def test_lanl_skip_summary_marks_detached_history() -> None:
    records = {
        "s1": _lanl_record("s1", 1, src_user="u9", src_host="c9", dst_host="c10", aspects={"credential_reuse"}),
        "q": _lanl_record("q", 4, src_user="u1", src_host="c1", dst_host="c4", aspects={"new_host_access"}),
    }
    edge_store = FakeEdgeStore({})
    chain_store = FakeChainStore({})
    skip_link_store = FakeSkipLinkStore(
        {
            "q": [
                SkipLink(
                    from_id="s1",
                    to_id="q",
                    skip_value=0.9,
                    aspects={"credential_reuse"},
                    summary="Detached skip candidate",
                    representative_event_ids=["s1"],
                    cost=1.0,
                )
            ]
        }
    )
    index = FakeIndex(
        event_store=FakeEventStore(records),
        edge_store=edge_store,
        chain_store=chain_store,
        skip_link_store=skip_link_store,
        config=TimeIndexConfig(
            retrieval=RetrievalConfig(return_summaries=True, allow_skip_expansion=True, default_budget=2),
            synthetic=None,  # type: ignore[arg-type]
        ),
    )
    index.config.scoring = ScoringConfig(retrieval_stop_threshold=0.01)

    results = retrieve(index, "q", DecisionIntent(aspects={"credential_reuse"}), budget=2)

    assert results
    skip_results = [result for result in results if result.object_id.startswith("skip:")]
    assert skip_results
    assert "lanl_detached_history" in skip_results[0].aspects
    assert "detached_from_query=true" in skip_results[0].summary
