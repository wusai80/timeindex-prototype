"""Configuration helpers for the IBM AML TimeIndex benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from timeindex.config import TimeIndexConfig


DEFAULT_BUDGETS: tuple[int, ...] = (3, 5, 10, 20)
DEFAULT_OUTPUT_DIR = Path("outputs/ibm_aml")


@dataclass(slots=True)
class IBMAMLBenchmarkConfig:
    """Benchmark-level options for running TimeIndex on IBM AML data."""

    csv_path: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    budgets: tuple[int, ...] = DEFAULT_BUDGETS
    query_laundering_only: bool = True
    negative_query_sample_size: int = 0
    random_seed: int = 0
    timeindex: TimeIndexConfig = field(default_factory=TimeIndexConfig)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["csv_path"] = str(self.csv_path)
        data["output_dir"] = str(self.output_dir)
        return data


def build_benchmark_config(
    csv_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    budgets: list[int] | tuple[int, ...] | None = None,
    query_laundering_only: bool = True,
    negative_query_sample_size: int = 0,
    random_seed: int = 0,
    timeindex_config: TimeIndexConfig | None = None,
) -> IBMAMLBenchmarkConfig:
    """Construct a normalized benchmark config."""

    budget_values = tuple(int(value) for value in (budgets or DEFAULT_BUDGETS))
    if not budget_values:
        budget_values = DEFAULT_BUDGETS
    effective_timeindex_config = timeindex_config or TimeIndexConfig()
    effective_timeindex_config.construction.maintain_outgoing_links = False
    return IBMAMLBenchmarkConfig(
        csv_path=Path(csv_path),
        output_dir=Path(output_dir),
        budgets=budget_values,
        query_laundering_only=query_laundering_only,
        negative_query_sample_size=max(0, int(negative_query_sample_size)),
        random_seed=int(random_seed),
        timeindex=effective_timeindex_config,
    )
