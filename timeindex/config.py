"""Configuration dataclasses for the TimeIndex prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExtractorConfig:
    """Configuration for key, sketch, and aspect extraction."""

    sketch_dim: int = 128
    text_token_weight: float = 0.2
    key_weight: float = 1.0
    time_bucket_width: int = 100
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoringConfig:
    """Configuration for local, anchor, skip, and retrieval scoring."""

    local_dependency_threshold: float = 0.35
    skip_threshold: float = 0.40
    anchor_threshold: float = 0.45
    retrieval_stop_threshold: float = 0.05
    time_decay: float = 100.0
    skip_lanl_temporal_gain_scale: float = 400.0
    skip_lanl_high_baseline_events: int = 20
    skip_lanl_high_baseline_hosts: int = 8
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StoreConfig:
    """Bounds for in-memory index structures."""

    active_history_size: int = 10_000
    posting_list_size: int = 100
    ordinary_fan_in: int = 5
    skip_fan_in: int = 3
    chain_summaries_per_family: int = 5
    anchor_candidates: int = 20
    correlation_candidates: int = 20
    rarity_candidates: int = 20
    intent_candidates: int = 20
    aspect_candidates: int = 20


@dataclass(slots=True)
class ConstructionConfig:
    """Configuration for online index construction."""

    maintain_outgoing_links: bool = True
    expire_stale_items: bool = True
    expire_batch_size: int = 0
    enable_skip_links: bool = True
    enable_bridge_score: bool = True
    enable_aspect_candidates: bool = True
    enable_rare_candidates: bool = True
    enable_correlation_candidates: bool = True
    skip_candidate_pool_factor: int = 3
    skip_summary_event_limit: int = 4
    skip_min_representative_events: int = 2
    skip_min_single_event_time_gap_seconds: float = 3600.0
    skip_min_single_event_order_gap: int = 4
    prefer_chain_skip_candidates: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalConfig:
    """Configuration for dual-frontier retrieval."""

    return_summaries: bool = True
    allow_skip_expansion: bool = True
    default_budget: int = 10
    priority_epsilon: float = 1e-8
    max_frontier_size: int = 64
    max_branch_factor: int = 3
    max_search_expansions: int = 64
    max_depth: int = 4
    skip_competitive_ratio: float = 0.9
    hot_cache_size: int = 2048
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SyntheticConfig:
    """Configuration for synthetic event stream generation."""

    seed: int = 0
    num_events: int = 100
    stream_type: str = "transaction"
    include_labels: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TimeIndexConfig:
    """Top-level configuration container."""

    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    stores: StoreConfig = field(default_factory=StoreConfig)
    construction: ConstructionConfig = field(default_factory=ConstructionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
