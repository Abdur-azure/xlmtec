"""
xlmtec.dashboard.comparator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Side-by-side comparison of multiple training runs.
Picks a winner based on best available metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xlmtec.dashboard.reader import RunInfo, RunReader


@dataclass
class ComparisonResult:
    """Result of comparing multiple runs."""

    runs: list[RunInfo]
    winner: RunInfo | None = None
    winner_reason: str = ""
    metric_used: str = ""

    @property
    def run_names(self) -> list[str]:
        return [r.name for r in self.runs]


class RunComparator:
    """Compare multiple training runs and pick a winner."""

    # Metrics in priority order — first available wins
    _METRIC_PRIORITY = [
        ("best_metric", "best metric"),
        ("best_eval_loss", "lowest eval loss"),
        ("final_train_loss", "lowest train loss"),
    ]

    def compare(self, run_dirs: list[Path]) -> ComparisonResult:
        """Read and compare multiple run directories.

        Args:
            run_dirs: List of output directories to compare.

        Returns:
            A :class:`ComparisonResult` with runs ranked and winner identified.

        Raises:
            ValueError: If fewer than 2 valid runs are found.
        """
        runs: list[RunInfo] = []
        errors: list[str] = []

        for d in run_dirs:
            try:
                runs.append(RunReader(d).read())
            except (FileNotFoundError, ValueError) as exc:
                errors.append(str(exc))

        if len(runs) < 1:
            raise ValueError("No valid runs found.\n" + "\n".join(errors))

        winner, reason, metric = self._pick_winner(runs)
        return ComparisonResult(
            runs=runs,
            winner=winner,
            winner_reason=reason,
            metric_used=metric,
        )

    def _pick_winner(self, runs: list[RunInfo]) -> tuple[RunInfo | None, str, str]:
        """Pick winner using first available metric in priority order."""
        for attr, label in self._METRIC_PRIORITY:
            values = [(r, getattr(r, attr, None)) for r in runs]
            valid = [(r, v) for r, v in values if v is not None]
            if not valid:
                continue

            # For loss metrics lower is better; for others higher is better
            if "loss" in attr:
                winner, val = min(valid, key=lambda x: x[1])
                return winner, f"{label}: {val:.4f}", label
            else:
                winner, val = max(valid, key=lambda x: x[1])
                return winner, f"{label}: {val:.4f}", label

        # Fallback — most steps completed
        winner = max(runs, key=lambda r: r.total_steps)
        return winner, f"most steps completed: {winner.total_steps}", "steps"

    def diff_configs(self, run_a: RunInfo, run_b: RunInfo) -> dict[str, tuple]:
        """Return fields that differ between two run configs.

        Returns:
            Dict of {field_path: (value_a, value_b)} for differing fields.
        """
        diffs = {}
        all_keys = set(run_a.config) | set(run_b.config)
        for key in sorted(all_keys):
            va = run_a.config.get(key)
            vb = run_b.config.get(key)
            if va != vb:
                diffs[key] = (va, vb)
        return diffs
