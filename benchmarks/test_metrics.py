from __future__ import annotations

from pathlib import Path

from benchmarks.metrics import (
    context_efficiency,
    decision_proxy_score,
    evidence_f1_at_budget,
    evidence_precision_at_budget,
    evidence_recall_at_budget,
    index_size_stats,
    mean_latency_ms,
    update_throughput_events_per_sec,
)
from benchmarks.plots import (
    plot_ablation_bars,
    plot_context_efficiency,
    plot_latency_vs_budget,
    plot_precision_at_budget,
    plot_recall_at_budget,
)


def test_metrics_handle_empty_gold_sets() -> None:
    assert evidence_recall_at_budget(["e1", "e2"], []) == 1.0
    assert evidence_precision_at_budget(["e1", "e2"], []) == 0.0
    assert evidence_f1_at_budget(["e1", "e2"], []) == 0.0
    assert context_efficiency(["e1", "e2"], [], budget_used=2) == 0.5


def test_metrics_handle_empty_retrievals() -> None:
    assert evidence_recall_at_budget([], ["e1", "e2"]) == 0.0
    assert evidence_precision_at_budget([], ["e1", "e2"]) == 0.0
    assert evidence_f1_at_budget([], ["e1", "e2"]) == 0.0
    assert context_efficiency([], ["e1", "e2"], budget_used=0) == 0.0


def test_aggregate_metrics_and_proxy_score_are_deterministic() -> None:
    proxy = decision_proxy_score(
        [
            {
                "summary": "Potential laundering chain with shell transfers",
                "amount": 70000,
                "entity_ids": ["acct-a", "acct-b"],
                "aspects": ["layering", "same_entity"],
            },
            {
                "summary": "Rapid round trip between repeated entities",
                "amount": 50000,
                "entity_ids": ["acct-b", "acct-c"],
                "aspects": ["recent_window", "nearest_neighbor"],
            },
        ]
    )

    assert mean_latency_ms([10, 20, 30]) == 20.0
    assert update_throughput_events_per_sec(num_events=20, elapsed_seconds=4) == 5.0
    assert update_throughput_events_per_sec(event_timestamps_s=[0.0, 0.5, 1.0, 1.5]) == 4 / 1.5
    assert index_size_stats([10, 20, 30]) == {
        "count": 3.0,
        "total": 60.0,
        "min": 10.0,
        "max": 30.0,
        "mean": 20.0,
        "median": 20.0,
        "std": (200 / 3) ** 0.5 / 1.0,
    }
    assert proxy["laundering_like_count"] == 2.0
    assert proxy["amount_sum"] == 120000.0
    assert proxy["entity_overlap"] == 2.0
    assert proxy["aspect_coverage"] == 4.0
    assert proxy["predicted_positive"] is True


def test_plots_run_on_tiny_fake_results_and_save_png_pdf(tmp_path: Path) -> None:
    results = [
        {"budget": 1, "recall": 0.25, "precision": 1.0, "context_efficiency": 0.25, "latency_ms": 2.0},
        {"budget": 2, "recall": 0.5, "precision": 0.75, "context_efficiency": 0.25, "latency_ms": 3.5},
    ]
    ablations = [
        {"name": "recent window", "score": 0.20},
        {"name": "same entity", "score": 0.35},
        {"name": "nearest neighbor", "score": 0.40},
        {"name": "chain-only", "score": 0.45},
        {"name": "TimeIndex no-skip", "score": 0.52},
        {"name": "TimeIndex full", "score": 0.61},
    ]

    outputs = [
        plot_recall_at_budget(results, tmp_path / "recall_budget"),
        plot_precision_at_budget(results, tmp_path / "precision_budget"),
        plot_context_efficiency(results, tmp_path / "context_efficiency_budget"),
        plot_latency_vs_budget(results, tmp_path / "latency_budget"),
        plot_ablation_bars(ablations, tmp_path / "ablations"),
    ]

    for png_path, pdf_path in outputs:
        assert png_path.exists()
        assert png_path.suffix == ".png"
        assert pdf_path.exists()
        assert pdf_path.suffix == ".pdf"
