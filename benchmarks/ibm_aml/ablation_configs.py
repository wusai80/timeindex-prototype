"""Configuration helpers for IBM AML ablation runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from timeindex.config import TimeIndexConfig


@dataclass(frozen=True, slots=True)
class AblationVariant:
    """Single ablation variant definition."""

    name: str
    budgets: tuple[int, ...]
    mode: str
    description: str
    use_timeindex: bool = False


DEFAULT_BUDGETS: tuple[int, ...] = (3, 5, 10, 20)


def default_variants() -> list[AblationVariant]:
    """Return the standard IBM AML ablation suite."""

    return [
        AblationVariant("recent_window", DEFAULT_BUDGETS, "recent_window", "Most recent historical events."),
        AblationVariant("same_entity_window", DEFAULT_BUDGETS, "same_entity_window", "Recent events sharing source or destination account."),
        AblationVariant("nearest_neighbor", DEFAULT_BUDGETS, "nearest_neighbor", "Vector-similar historical events."),
        AblationVariant("chain_only", DEFAULT_BUDGETS, "chain_only", "TimeIndex ordinary-link traversal only."),
        AblationVariant("timeindex_no_skip", DEFAULT_BUDGETS, "timeindex", "TimeIndex with skip links disabled.", use_timeindex=True),
        AblationVariant("timeindex_no_bridge", DEFAULT_BUDGETS, "timeindex", "TimeIndex without chain-anchor bridge updates.", use_timeindex=True),
        AblationVariant("timeindex_no_aspect", DEFAULT_BUDGETS, "timeindex", "TimeIndex without aspect candidate retrieval.", use_timeindex=True),
        AblationVariant("timeindex_full", DEFAULT_BUDGETS, "timeindex", "Full TimeIndex configuration.", use_timeindex=True),
    ]


def build_variant_config(variant_name: str, base_config: TimeIndexConfig | None = None) -> TimeIndexConfig:
    """Create a TimeIndex config adjusted for a specific ablation."""

    config = deepcopy(base_config) if base_config is not None else TimeIndexConfig()
    config.construction.enable_skip_links = True
    config.construction.enable_bridge_score = True
    config.construction.enable_aspect_candidates = True
    config.construction.enable_rare_candidates = True
    config.construction.enable_correlation_candidates = True

    if variant_name == "timeindex_no_skip":
        config.construction.enable_skip_links = False
        config.stores.skip_fan_in = 0
    elif variant_name == "timeindex_no_bridge":
        config.construction.enable_bridge_score = False
    elif variant_name == "timeindex_no_aspect":
        config.construction.enable_aspect_candidates = False
        config.stores.aspect_candidates = 0
    elif variant_name == "timeindex_full":
        pass

    if not config.construction.enable_rare_candidates:
        config.stores.rarity_candidates = 0
    if not config.construction.enable_correlation_candidates:
        config.stores.correlation_candidates = 0
    if not config.construction.enable_aspect_candidates:
        config.stores.aspect_candidates = 0

    return config
