"""SQLite-backed retrieval backend for TimeIndex."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, fields
import json
from pathlib import Path
import sqlite3
from typing import Any, Generic, TypeVar

from .config import (
    ConstructionConfig,
    ExtractorConfig,
    RetrievalConfig,
    ScoringConfig,
    StoreConfig,
    SyntheticConfig,
    TimeIndexConfig,
)
from .event import ChainSummary, Event, EventMetadata, EventRecord, OrdinaryLink, SkipLink


T = TypeVar("T")


class HotCache(Generic[T]):
    """Small in-memory LRU cache for frequently reused retrieval data."""

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max(1, int(max_entries))
        self._items: OrderedDict[str, T] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> T | None:
        if key not in self._items:
            self.misses += 1
            return None
        self.hits += 1
        value = self._items.pop(key)
        self._items[key] = value
        return value

    def put(self, key: str, value: T) -> T:
        if key in self._items:
            self._items.pop(key)
        self._items[key] = value
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)
        return value

    def get_or_load(self, key: str, loader: Any) -> T:
        cached = self.get(key)
        if cached is not None:
            return cached
        return self.put(key, loader())

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._items),
            "hits": self.hits,
            "misses": self.misses,
            "max_entries": self.max_entries,
        }


class SqliteTimeIndexWriter:
    """Incrementally persist retrieval-facing index state into SQLite."""

    def __init__(
        self,
        path: str | Path,
        *,
        config: TimeIndexConfig | None = None,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path)
        if overwrite and self.path.exists():
            self.path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        _configure_connection(self.connection)
        _create_schema(self.connection)
        self.connection.execute("BEGIN")
        self.write_config(config or TimeIndexConfig())

    def write_config(self, config: TimeIndexConfig) -> None:
        self.write_metadata("config_json", _config_to_dict(config))

    def write_metadata(self, key: str, value: Any) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata (key, value_json) VALUES (?, ?)",
            (str(key), _encode_json(value)),
        )

    def write_event_snapshot(
        self,
        record: EventRecord,
        ordinary_links: Sequence[OrdinaryLink],
        chain_summaries: Sequence[ChainSummary],
        skip_links: Sequence[SkipLink],
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO events (
                event_id,
                time_text,
                event_type,
                attrs_json,
                ctx_json,
                text_value,
                label_value,
                lookup_keys_json,
                aspects_json,
                rarity,
                surprise,
                insertion_order,
                expired,
                labels_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _event_row(record),
        )
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO ordinary_links (successor_id, predecessor_id, score)
            VALUES (?, ?, ?)
            """,
            [_ordinary_link_row(link) for link in ordinary_links],
        )
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO chain_summaries (
                chain_id,
                family,
                head_id,
                tail_id,
                representative_event_ids_json,
                source_entities_json,
                destination_entities_json,
                aspects_json,
                dependency_confidence,
                summary,
                cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_chain_row(summary) for summary in chain_summaries],
        )
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO skip_links (
                to_id,
                from_id,
                skip_value,
                segment_confidence,
                source_entities_json,
                destination_entities_json,
                aspects_json,
                summary,
                representative_event_ids_json,
                cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_skip_row(link) for link in skip_links],
        )

    def expire_event_ids(self, event_ids: Sequence[str]) -> None:
        expired_ids = [str(event_id) for event_id in event_ids if str(event_id)]
        if not expired_ids:
            return
        placeholders = ", ".join("?" for _ in expired_ids)
        self.connection.execute(
            f"UPDATE events SET expired = 1 WHERE event_id IN ({placeholders})",
            expired_ids,
        )
        self.connection.execute(
            f"DELETE FROM ordinary_links WHERE successor_id IN ({placeholders}) OR predecessor_id IN ({placeholders})",
            [*expired_ids, *expired_ids],
        )
        self.connection.execute(
            f"DELETE FROM chain_summaries WHERE tail_id IN ({placeholders}) OR head_id IN ({placeholders})",
            [*expired_ids, *expired_ids],
        )
        self.connection.execute(
            f"DELETE FROM skip_links WHERE to_id IN ({placeholders}) OR from_id IN ({placeholders})",
            [*expired_ids, *expired_ids],
        )

    def flush(self) -> None:
        self.connection.commit()
        self.connection.execute("BEGIN")

    def close(self) -> None:
        try:
            self.connection.commit()
        finally:
            self.connection.close()

    def __enter__(self) -> SqliteTimeIndexWriter:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class SqliteEventStore:
    """Event record access with a hot in-memory buffer."""

    def __init__(self, backend: SqliteTimeIndexBackend) -> None:
        self.backend = backend

    def get(self, event_id: str) -> EventRecord | None:
        return self.backend._event_cache.get_or_load(event_id, lambda: self._load(event_id))

    def _load(self, event_id: str) -> EventRecord | None:
        row = self.backend.connection.execute(
            """
            SELECT
                event_id,
                time_text,
                event_type,
                attrs_json,
                ctx_json,
                text_value,
                label_value,
                lookup_keys_json,
                aspects_json,
                rarity,
                surprise,
                insertion_order,
                expired,
                labels_json
            FROM events
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()
        if row is None or int(row["expired"] or 0) != 0:
            return None
        event = Event(
            event_id=str(row["event_id"]),
            time=_decode_scalar(row["time_text"]),
            event_type=str(row["event_type"]),
            attrs=_decode_json_object(row["attrs_json"]),
            ctx=_decode_json_object(row["ctx_json"]),
            text=row["text_value"],
            label=row["label_value"],
        )
        metadata = EventMetadata(
            rarity=float(row["rarity"] or 0.0),
            surprise=float(row["surprise"] or 0.0),
            insertion_order=int(row["insertion_order"]) if row["insertion_order"] is not None else None,
            expired=bool(int(row["expired"] or 0)),
            labels=_decode_json_object(row["labels_json"]),
        )
        return EventRecord(
            event=event,
            lookup_keys=set(_decode_json_list(row["lookup_keys_json"])),
            aspects=set(_decode_json_list(row["aspects_json"])),
            metadata=metadata,
        )


class SqliteEdgeStore:
    """Ordinary incoming-link access with a hot buffer."""

    def __init__(self, backend: SqliteTimeIndexBackend) -> None:
        self.backend = backend

    def incoming(self, event_id: str) -> Sequence[OrdinaryLink]:
        return self.backend._ordinary_cache.get_or_load(event_id, lambda: self._load(event_id))

    def _load(self, event_id: str) -> list[OrdinaryLink]:
        rows = self.backend.connection.execute(
            """
            SELECT predecessor_id, successor_id, score
            FROM ordinary_links
            WHERE successor_id = ?
            ORDER BY score DESC, predecessor_id ASC, successor_id ASC
            """,
            (event_id,),
        ).fetchall()
        return [
            OrdinaryLink(
                predecessor_id=str(row["predecessor_id"]),
                successor_id=str(row["successor_id"]),
                score=float(row["score"]),
            )
            for row in rows
        ]


class SqliteChainStore:
    """Chain-summary access with a hot buffer."""

    def __init__(self, backend: SqliteTimeIndexBackend) -> None:
        self.backend = backend

    def get_for_tail(self, event_id: str) -> Sequence[ChainSummary]:
        return self.backend._chain_cache.get_or_load(event_id, lambda: self._load(event_id))

    def _load(self, event_id: str) -> list[ChainSummary]:
        rows = self.backend.connection.execute(
            """
            SELECT
                chain_id,
                family,
                head_id,
                tail_id,
                representative_event_ids_json,
                source_entities_json,
                destination_entities_json,
                aspects_json,
                dependency_confidence,
                summary,
                cost
            FROM chain_summaries
            WHERE tail_id = ?
            ORDER BY family ASC, dependency_confidence DESC, chain_id ASC
            """,
            (event_id,),
        ).fetchall()
        return [
            ChainSummary(
                chain_id=str(row["chain_id"]),
                family=str(row["family"]),
                head_id=str(row["head_id"]),
                tail_id=str(row["tail_id"]),
                representative_event_ids=_decode_json_list(row["representative_event_ids_json"]),
                source_entities=set(_decode_json_list(row["source_entities_json"])),
                destination_entities=set(_decode_json_list(row["destination_entities_json"])),
                aspects=set(_decode_json_list(row["aspects_json"])),
                dependency_confidence=float(row["dependency_confidence"] or 0.0),
                summary=str(row["summary"] or ""),
                cost=float(row["cost"] or 0.0),
            )
            for row in rows
        ]


class SqliteSkipLinkStore:
    """Skip-link access with a hot buffer."""

    def __init__(self, backend: SqliteTimeIndexBackend) -> None:
        self.backend = backend

    def incoming(self, event_id: str) -> Sequence[SkipLink]:
        return self.backend._skip_cache.get_or_load(event_id, lambda: self._load(event_id))

    def _load(self, event_id: str) -> list[SkipLink]:
        rows = self.backend.connection.execute(
            """
            SELECT
                from_id,
                to_id,
                skip_value,
                segment_confidence,
                source_entities_json,
                destination_entities_json,
                aspects_json,
                summary,
                representative_event_ids_json,
                cost
            FROM skip_links
            WHERE to_id = ?
            ORDER BY skip_value DESC, from_id ASC, to_id ASC
            """,
            (event_id,),
        ).fetchall()
        return [
            SkipLink(
                from_id=str(row["from_id"]),
                to_id=str(row["to_id"]),
                skip_value=float(row["skip_value"]),
                segment_confidence=float(row["segment_confidence"] or 0.0),
                source_entities=set(_decode_json_list(row["source_entities_json"])),
                destination_entities=set(_decode_json_list(row["destination_entities_json"])),
                aspects=set(_decode_json_list(row["aspects_json"])),
                summary=str(row["summary"] or ""),
                representative_event_ids=_decode_json_list(row["representative_event_ids_json"]),
                cost=float(row["cost"] or 0.0),
            )
            for row in rows
        ]


class SqliteTimeIndexBackend:
    """Disk-backed retrieval view over a built TimeIndex."""

    def __init__(self, path: str | Path, *, hot_cache_size: int | None = None) -> None:
        self.path = Path(path)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.row_factory = sqlite3.Row
        self._configure_connection()
        self.config = self._load_config()

        cache_size = hot_cache_size
        if cache_size is None:
            cache_size = getattr(self.config.retrieval, "hot_cache_size", 2048)
        cache_size = max(64, int(cache_size))
        branch_cache_size = max(32, cache_size // 4)
        self._event_cache: HotCache[EventRecord | None] = HotCache(cache_size)
        self._ordinary_cache: HotCache[list[OrdinaryLink]] = HotCache(branch_cache_size)
        self._chain_cache: HotCache[list[ChainSummary]] = HotCache(branch_cache_size)
        self._skip_cache: HotCache[list[SkipLink]] = HotCache(branch_cache_size)

        self.event_store = SqliteEventStore(self)
        self.edge_store = SqliteEdgeStore(self)
        self.chain_store = SqliteChainStore(self)
        self.skip_link_store = SqliteSkipLinkStore(self)

    @classmethod
    def open(cls, path: str | Path, *, hot_cache_size: int | None = None) -> SqliteTimeIndexBackend:
        return cls(path, hot_cache_size=hot_cache_size)

    def get_event(self, event_id: str) -> EventRecord | None:
        return self.event_store.get(event_id)

    def cache_stats(self) -> dict[str, dict[str, int]]:
        return {
            "events": self._event_cache.stats(),
            "ordinary_links": self._ordinary_cache.stats(),
            "chains": self._chain_cache.stats(),
            "skip_links": self._skip_cache.stats(),
        }

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SqliteTimeIndexBackend:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _configure_connection(self) -> None:
        _configure_connection(self.connection)

    def _load_config(self) -> TimeIndexConfig:
        row = self.connection.execute(
            "SELECT value_json FROM metadata WHERE key = 'config_json'"
        ).fetchone()
        if row is None:
            return TimeIndexConfig()
        payload = json.loads(str(row["value_json"]))
        return _config_from_dict(payload)


def export_sqlite_backend(
    index: Any,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export the retrieval-facing portion of an index into SQLite."""

    target = Path(path)
    if overwrite and target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)

    writer = SqliteTimeIndexWriter(target, config=getattr(index, "config", None), overwrite=overwrite)
    try:
        for record in _iter_event_records(index):
            event_id = record.event.event_id
            writer.write_event_snapshot(
                record,
                _incoming_links(getattr(index, "edge_store", None), event_id),
                _chain_summaries(getattr(index, "chain_store", None), event_id),
                [link for link in _incoming_links(getattr(index, "skip_link_store", None), event_id) if isinstance(link, SkipLink)],
            )
        writer.flush()
    finally:
        writer.close()
    return target


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            time_text TEXT NOT NULL,
            event_type TEXT NOT NULL,
            attrs_json TEXT NOT NULL,
            ctx_json TEXT NOT NULL,
            text_value TEXT,
            label_value TEXT,
            lookup_keys_json TEXT NOT NULL,
            aspects_json TEXT NOT NULL,
            rarity REAL NOT NULL,
            surprise REAL NOT NULL,
            insertion_order INTEGER,
            expired INTEGER NOT NULL,
            labels_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ordinary_links (
            successor_id TEXT NOT NULL,
            predecessor_id TEXT NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (successor_id, predecessor_id)
        );

        CREATE TABLE IF NOT EXISTS chain_summaries (
            chain_id TEXT PRIMARY KEY,
            family TEXT NOT NULL,
            head_id TEXT NOT NULL,
            tail_id TEXT NOT NULL,
            representative_event_ids_json TEXT NOT NULL,
            source_entities_json TEXT NOT NULL,
            destination_entities_json TEXT NOT NULL,
            aspects_json TEXT NOT NULL,
            dependency_confidence REAL NOT NULL,
            summary TEXT NOT NULL,
            cost REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS skip_links (
            to_id TEXT NOT NULL,
            from_id TEXT NOT NULL,
            skip_value REAL NOT NULL,
            segment_confidence REAL NOT NULL,
            source_entities_json TEXT NOT NULL,
            destination_entities_json TEXT NOT NULL,
            aspects_json TEXT NOT NULL,
            summary TEXT NOT NULL,
            representative_event_ids_json TEXT NOT NULL,
            cost REAL NOT NULL,
            PRIMARY KEY (to_id, from_id)
        );

        CREATE INDEX IF NOT EXISTS idx_events_insertion_order ON events (insertion_order);
        CREATE INDEX IF NOT EXISTS idx_ordinary_successor ON ordinary_links (successor_id, score DESC);
        CREATE INDEX IF NOT EXISTS idx_chain_tail ON chain_summaries (tail_id, family, dependency_confidence DESC);
        CREATE INDEX IF NOT EXISTS idx_skip_target ON skip_links (to_id, skip_value DESC);
        """
    )


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")


def _write_metadata(connection: sqlite3.Connection, index: Any) -> None:
    config = getattr(index, "config", None)
    payload = _config_to_dict(config if config is not None else TimeIndexConfig())
    connection.execute(
        "INSERT OR REPLACE INTO metadata (key, value_json) VALUES (?, ?)",
        ("config_json", _encode_json(payload)),
    )


def _write_events(connection: sqlite3.Connection, index: Any) -> list[str]:
    records = list(_iter_event_records(index))
    rows = []
    event_ids: list[str] = []
    for record in records:
        event = record.event
        metadata = record.metadata
        event_ids.append(event.event_id)
        rows.append(
            _event_row(record)
        )
    connection.executemany(
        """
        INSERT OR REPLACE INTO events (
            event_id,
            time_text,
            event_type,
            attrs_json,
            ctx_json,
            text_value,
            label_value,
            lookup_keys_json,
            aspects_json,
            rarity,
            surprise,
            insertion_order,
            expired,
            labels_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return event_ids


def _write_ordinary_links(connection: sqlite3.Connection, index: Any, event_ids: Sequence[str]) -> None:
    rows = []
    for event_id in event_ids:
        for link in _incoming_links(getattr(index, "edge_store", None), event_id):
            rows.append(_ordinary_link_row(link))
    connection.executemany(
        """
        INSERT OR REPLACE INTO ordinary_links (successor_id, predecessor_id, score)
        VALUES (?, ?, ?)
        """,
        rows,
    )


def _write_chain_summaries(connection: sqlite3.Connection, index: Any, event_ids: Sequence[str]) -> None:
    rows = []
    chain_store = getattr(index, "chain_store", None)
    if chain_store is None:
        return
    seen_chain_ids: set[str] = set()
    for event_id in event_ids:
        summaries = _chain_summaries(chain_store, event_id)
        for summary in summaries:
            if summary.chain_id in seen_chain_ids:
                continue
            seen_chain_ids.add(summary.chain_id)
            rows.append(
                _chain_row(summary)
            )
    connection.executemany(
        """
        INSERT OR REPLACE INTO chain_summaries (
            chain_id,
            family,
            head_id,
            tail_id,
            representative_event_ids_json,
            source_entities_json,
            destination_entities_json,
            aspects_json,
            dependency_confidence,
            summary,
            cost
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _write_skip_links(connection: sqlite3.Connection, index: Any, event_ids: Sequence[str]) -> None:
    rows = []
    for event_id in event_ids:
        for link in _incoming_links(getattr(index, "skip_link_store", None), event_id):
            if not isinstance(link, SkipLink):
                continue
            rows.append(
                _skip_row(link)
            )
    connection.executemany(
        """
        INSERT OR REPLACE INTO skip_links (
            to_id,
            from_id,
            skip_value,
            segment_confidence,
            source_entities_json,
            destination_entities_json,
            aspects_json,
            summary,
            representative_event_ids_json,
            cost
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _iter_event_records(index: Any) -> Iterator[EventRecord]:
    store = getattr(index, "event_store", None)
    if store is None:
        return iter(())

    listed = getattr(store, "list", None)
    if callable(listed):
        for record in listed():
            if isinstance(record, EventRecord):
                yield record
        return

    ordered_records = getattr(store, "_records", None)
    insertion_order = getattr(store, "_insertion_order", None)
    if isinstance(ordered_records, dict) and isinstance(insertion_order, Sequence):
        for event_id in insertion_order:
            record = ordered_records.get(event_id)
            if isinstance(record, EventRecord):
                yield record
        return

    records = getattr(store, "records", None)
    order = getattr(store, "order", None)
    if isinstance(records, dict) and isinstance(order, Sequence):
        for event_id in order:
            record = records.get(event_id)
            if isinstance(record, EventRecord):
                yield record
        return

    if isinstance(records, dict):
        for event_id in sorted(records):
            record = records[event_id]
            if isinstance(record, EventRecord):
                yield record


def _incoming_links(store: Any, event_id: str) -> Sequence[OrdinaryLink] | Sequence[SkipLink]:
    if store is None:
        return []
    method = getattr(store, "incoming", None)
    if callable(method):
        return list(method(event_id))
    mapping = getattr(store, "_incoming", None) or getattr(store, "incoming_map", None)
    if isinstance(mapping, dict):
        return list(mapping.get(event_id, ()))
    return []


def _chain_summaries(store: Any, tail_id: str) -> Sequence[ChainSummary]:
    method = getattr(store, "get_for_tail", None)
    if callable(method):
        return list(method(tail_id))
    return []


def _event_row(record: EventRecord) -> tuple[Any, ...]:
    event = record.event
    metadata = record.metadata
    return (
        event.event_id,
        _encode_json(event.time),
        event.event_type,
        _encode_json(dict(event.attrs)),
        _encode_json(dict(event.ctx)),
        event.text,
        event.label,
        _encode_json(sorted(str(value) for value in record.lookup_keys)),
        _encode_json(sorted(str(value) for value in record.aspects)),
        float(metadata.rarity),
        float(metadata.surprise),
        metadata.insertion_order,
        1 if metadata.expired else 0,
        _encode_json(dict(metadata.labels)),
    )


def _ordinary_link_row(link: OrdinaryLink) -> tuple[Any, ...]:
    return (link.successor_id, link.predecessor_id, float(link.score))


def _chain_row(summary: ChainSummary) -> tuple[Any, ...]:
    return (
        summary.chain_id,
        summary.family,
        summary.head_id,
        summary.tail_id,
        _encode_json(list(summary.representative_event_ids)),
        _encode_json(sorted(str(value) for value in summary.source_entities)),
        _encode_json(sorted(str(value) for value in summary.destination_entities)),
        _encode_json(sorted(str(value) for value in summary.aspects)),
        float(summary.dependency_confidence),
        summary.summary,
        float(summary.cost),
    )


def _skip_row(link: SkipLink) -> tuple[Any, ...]:
    return (
        link.to_id,
        link.from_id,
        float(link.skip_value),
        float(link.segment_confidence),
        _encode_json(sorted(str(value) for value in link.source_entities)),
        _encode_json(sorted(str(value) for value in link.destination_entities)),
        _encode_json(sorted(str(value) for value in link.aspects)),
        link.summary,
        _encode_json(list(link.representative_event_ids)),
        float(link.cost),
    )




def _config_from_dict(payload: dict[str, Any]) -> TimeIndexConfig:
    return TimeIndexConfig(
        extractor=_build_dataclass(ExtractorConfig, payload.get("extractor")),
        scoring=_build_dataclass(ScoringConfig, payload.get("scoring")),
        stores=_build_dataclass(StoreConfig, payload.get("stores")),
        construction=_build_dataclass(ConstructionConfig, payload.get("construction")),
        retrieval=_build_dataclass(RetrievalConfig, payload.get("retrieval")),
        synthetic=_build_dataclass(SyntheticConfig, payload.get("synthetic")),
    )


def _build_dataclass(cls: type[T], payload: Any) -> T:
    if not isinstance(payload, dict):
        return cls()
    allowed = {field.name for field in fields(cls)}
    kwargs = {key: value for key, value in payload.items() if key in allowed}
    return cls(**kwargs)


def _config_to_dict(config: Any) -> dict[str, Any]:
    return {
        "extractor": _dataclass_like_to_dict(getattr(config, "extractor", None), ExtractorConfig()),
        "scoring": _dataclass_like_to_dict(getattr(config, "scoring", None), ScoringConfig()),
        "stores": _dataclass_like_to_dict(getattr(config, "stores", None), StoreConfig()),
        "construction": _dataclass_like_to_dict(getattr(config, "construction", None), ConstructionConfig()),
        "retrieval": _dataclass_like_to_dict(getattr(config, "retrieval", None), RetrievalConfig()),
        "synthetic": _dataclass_like_to_dict(getattr(config, "synthetic", None), SyntheticConfig()),
    }


def _dataclass_like_to_dict(value: Any, default: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields(type(default)):
        fallback = getattr(default, field.name)
        current = getattr(value, field.name, fallback)
        payload[field.name] = current
    return payload


def _encode_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _decode_json_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    decoded = json.loads(str(value))
    if not isinstance(decoded, list):
        return []
    return [str(item) for item in decoded]


def _decode_json_object(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    decoded = json.loads(str(value))
    return decoded if isinstance(decoded, dict) else {}


def _decode_scalar(value: Any) -> Any:
    if value in (None, ""):
        return value
    return json.loads(str(value))
