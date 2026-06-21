"""Matplotlib-only plotting helpers for IBM AML benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_recall_at_budget(results: list[dict[str, Any]], output_path: str | Path) -> tuple[Path, Path]:
    return _plot_metric_vs_budget(results, "recall", "Recall@B vs Budget", "Recall@B", output_path)


def plot_precision_at_budget(results: list[dict[str, Any]], output_path: str | Path) -> tuple[Path, Path]:
    return _plot_metric_vs_budget(results, "precision", "Precision@B vs Budget", "Precision@B", output_path)


def plot_context_efficiency(results: list[dict[str, Any]], output_path: str | Path) -> tuple[Path, Path]:
    return _plot_metric_vs_budget(
        results,
        "context_efficiency",
        "Context Efficiency vs Budget",
        "Context Efficiency",
        output_path,
    )


def plot_latency_vs_budget(results: list[dict[str, Any]], output_path: str | Path) -> tuple[Path, Path]:
    return _plot_metric_vs_budget(results, "latency_ms", "Latency vs Budget", "Latency (ms)", output_path)


def plot_ablation_bars(ablation_results: list[dict[str, Any]], output_path: str | Path) -> tuple[Path, Path]:
    ordered_names = [
        "recent window",
        "same entity",
        "nearest neighbor",
        "chain-only",
        "TimeIndex no-skip",
        "TimeIndex full",
    ]
    score_by_name = {str(item["name"]): float(item["score"]) for item in ablation_results}
    labels = [name for name in ordered_names if name in score_by_name]
    labels.extend(name for name in score_by_name if name not in labels)
    values = [score_by_name[name] for name in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values)
    ax.set_title("Ablation Comparison")
    ax.set_xlabel("Method")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def _plot_metric_vs_budget(
    results: list[dict[str, Any]],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> tuple[Path, Path]:
    sorted_results = sorted(results, key=lambda item: float(item["budget"]))
    budgets = [float(item["budget"]) for item in sorted_results]
    values = [float(item[metric_key]) for item in sorted_results]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(budgets, values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Budget")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def _save_figure(fig: plt.Figure, output_path: str | Path) -> tuple[Path, Path]:
    base = Path(output_path)
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path
