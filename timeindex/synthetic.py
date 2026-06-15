"""Synthetic data generation stubs for the TimeIndex prototype."""

from __future__ import annotations

from collections.abc import Sequence

from .config import SyntheticConfig
from .event import DecisionIntent, Event


class SyntheticStreamGenerator:
    """Stub generator for synthetic event streams and query intents."""

    def __init__(self, config: SyntheticConfig) -> None:
        self.config = config

    def generate_events(self) -> Sequence[Event]:
        raise NotImplementedError("Synthetic event generation is not implemented yet.")

    def generate_intents(self) -> Sequence[DecisionIntent]:
        raise NotImplementedError("Synthetic intent generation is not implemented yet.")
