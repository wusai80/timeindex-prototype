from __future__ import annotations

from benchmarks.ibm_aml.run_sqlite_deepseek_sample import (
    DEFAULT_POSITIVE_MIN_INSERTION_ORDER,
    _resolve_cutoffs,
    _should_expand_budget,
)


def test_should_expand_budget_for_thin_retrieval() -> None:
    assert _should_expand_budget(
        ["e1", "e2"],
        ["large_transfer"],
        adaptive_budget=16,
        base_budget=8,
        adaptive_min_events=4,
        adaptive_min_aspects=2,
    )


def test_should_expand_budget_for_low_aspect_diversity() -> None:
    assert _should_expand_budget(
        ["e1", "e2", "e3", "e4"],
        ["large_transfer"],
        adaptive_budget=16,
        base_budget=8,
        adaptive_min_events=4,
        adaptive_min_aspects=2,
    )


def test_should_not_expand_when_budget_is_not_larger() -> None:
    assert not _should_expand_budget(
        ["e1", "e2"],
        ["large_transfer"],
        adaptive_budget=8,
        base_budget=8,
        adaptive_min_events=4,
        adaptive_min_aspects=2,
    )


def test_should_not_expand_for_rich_retrieval() -> None:
    assert not _should_expand_budget(
        ["e1", "e2", "e3", "e4", "e5"],
        ["large_transfer", "beneficiary_novelty"],
        adaptive_budget=16,
        base_budget=8,
        adaptive_min_events=4,
        adaptive_min_aspects=2,
    )


def test_resolve_cutoffs_supports_late_positives_only() -> None:
    assert _resolve_cutoffs(
        min_insertion_order=None,
        positive_min_insertion_order=4_000_000,
        negative_min_insertion_order=None,
    ) == (4_000_000, None)


def test_resolve_cutoffs_uses_default_late_positive_sampler() -> None:
    assert _resolve_cutoffs(
        min_insertion_order=None,
        positive_min_insertion_order=None,
        negative_min_insertion_order=None,
    ) == (DEFAULT_POSITIVE_MIN_INSERTION_ORDER, None)
