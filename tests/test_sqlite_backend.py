from __future__ import annotations

from dataclasses import dataclass

from benchmarks.ibm_aml.run_fake_agent import _load_cached_index
from timeindex.config import RetrievalConfig, ScoringConfig, TimeIndexConfig
from timeindex.event import ChainSummary, DecisionIntent, Event, EventMetadata, EventRecord, OrdinaryLink, SkipLink
from timeindex.retrieval import retrieve
from timeindex.sqlite_backend import SqliteTimeIndexBackend, export_sqlite_backend


class SimpleEventStore:
    def __init__(self, records: dict[str, EventRecord], order: list[str]) -> None:
        self._records = records
        self._insertion_order = order

    def list(self) -> list[EventRecord]:
        return [self._records[event_id] for event_id in self._insertion_order]


class SimpleEdgeStore:
    def __init__(self, incoming_map: dict[str, list[OrdinaryLink]]) -> None:
        self.incoming_map = incoming_map

    def incoming(self, event_id: str) -> list[OrdinaryLink]:
        return list(self.incoming_map.get(event_id, []))


class SimpleSkipLinkStore:
    def __init__(self, incoming_map: dict[str, list[SkipLink]]) -> None:
        self.incoming_map = incoming_map

    def incoming(self, event_id: str) -> list[SkipLink]:
        return list(self.incoming_map.get(event_id, []))


class SimpleChainStore:
    def __init__(self, summaries_by_tail: dict[str, list[ChainSummary]]) -> None:
        self.summaries_by_tail = summaries_by_tail

    def get_for_tail(self, event_id: str) -> list[ChainSummary]:
        return list(self.summaries_by_tail.get(event_id, []))


@dataclass
class SimpleIndex:
    event_store: SimpleEventStore
    edge_store: SimpleEdgeStore
    chain_store: SimpleChainStore
    skip_link_store: SimpleSkipLinkStore
    config: TimeIndexConfig

    def get_event(self, event_id: str) -> EventRecord | None:
        return self.event_store._records.get(event_id)


def _record(event_id: str, time: int, aspects: set[str], **attrs) -> EventRecord:
    return EventRecord(
        event=Event(
            event_id=event_id,
            time=time,
            event_type="transaction",
            attrs=attrs or {},
            ctx={"dataset": "test"},
        ),
        lookup_keys={f"type:transaction", *{f"entity:{key}:{value}" for key, value in attrs.items()}},
        aspects=set(aspects),
        metadata=EventMetadata(insertion_order=time),
    )


def _build_index() -> SimpleIndex:
    records = {
        "e1": _record("e1", 1, {"source_accumulation"}, src_account="A0", dst_account="A1"),
        "e2": _record("e2", 2, {"source_accumulation"}, src_account="A1", dst_account="A2"),
        "e5": _record("e5", 5, {"beneficiary_novelty"}, src_account="A2", dst_account="A3"),
        "e6": _record("e6", 6, {"large_transfer"}, src_account="A3", dst_account="A4"),
        "e7": _record("e7", 7, {"full_balance_transfer"}, src_account="A4", dst_account="A5"),
    }
    edge_store = SimpleEdgeStore(
        {
            "e7": [OrdinaryLink(predecessor_id="e6", successor_id="e7", score=0.95)],
            "e6": [OrdinaryLink(predecessor_id="e5", successor_id="e6", score=0.85)],
        }
    )
    chain_store = SimpleChainStore(
        {
            "e7": [
                ChainSummary(
                    chain_id="chain:e6:e7",
                    family="transaction",
                    head_id="e6",
                    tail_id="e7",
                    representative_event_ids=["e6", "e7"],
                    aspects={"large_transfer", "full_balance_transfer"},
                    summary="Recent transfer chain",
                    cost=1.0,
                )
            ]
        }
    )
    skip_link_store = SimpleSkipLinkStore(
        {
            "e7": [
                SkipLink(
                    from_id="e1",
                    to_id="e7",
                    skip_value=0.93,
                    aspects={"source_accumulation", "full_balance_transfer"},
                    summary="Early buildup explains the final drain",
                    representative_event_ids=["e1", "e2"],
                    cost=1.0,
                )
            ]
        }
    )
    config = TimeIndexConfig(retrieval=RetrievalConfig(hot_cache_size=8))
    config.scoring = ScoringConfig(retrieval_stop_threshold=0.01)
    return SimpleIndex(
        event_store=SimpleEventStore(records, ["e1", "e2", "e5", "e6", "e7"]),
        edge_store=edge_store,
        chain_store=chain_store,
        skip_link_store=skip_link_store,
        config=config,
    )


def test_sqlite_backend_export_and_retrieve(tmp_path) -> None:
    index = _build_index()
    sqlite_path = tmp_path / "timeindex.sqlite"
    export_sqlite_backend(index, sqlite_path, overwrite=True)

    backend = SqliteTimeIndexBackend.open(sqlite_path, hot_cache_size=8)
    try:
        results = retrieve(
            backend,
            "e7",
            DecisionIntent(aspects={"source_accumulation", "full_balance_transfer"}),
            budget=3,
        )
    finally:
        backend.close()

    assert any(result.object_id.startswith("skip:") for result in results)
    assert any("e1" in result.event_ids for result in results)


def test_sqlite_backend_hot_cache_records_hits(tmp_path) -> None:
    index = _build_index()
    sqlite_path = tmp_path / "timeindex.sqlite"
    export_sqlite_backend(index, sqlite_path, overwrite=True)

    backend = SqliteTimeIndexBackend.open(sqlite_path, hot_cache_size=8)
    try:
        assert backend.get_event("e7") is not None
        assert backend.get_event("e7") is not None
        assert backend.edge_store.incoming("e7")
        assert backend.edge_store.incoming("e7")
        stats = backend.cache_stats()
    finally:
        backend.close()

    assert stats["events"]["hits"] >= 1
    assert stats["ordinary_links"]["hits"] >= 1


def test_load_cached_index_supports_sqlite(tmp_path) -> None:
    index = _build_index()
    sqlite_path = tmp_path / "timeindex.sqlite"
    export_sqlite_backend(index, sqlite_path, overwrite=True)

    backend = _load_cached_index(sqlite_path)
    try:
        assert isinstance(backend, SqliteTimeIndexBackend)
        assert backend.get_event("e1") is not None
    finally:
        backend.close()
