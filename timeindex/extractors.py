"""Extractor stubs for the TimeIndex prototype."""

from __future__ import annotations

from .config import ExtractorConfig
from .event import Event, EventRecord


class EventRepresentationExtractor:
    """Stub extractor for lookup keys, sketches, and evidence aspects."""

    def __init__(self, config: ExtractorConfig) -> None:
        self.config = config

    def extract(self, event: Event) -> EventRecord:
        raise NotImplementedError("Event representation extraction is not implemented yet.")
