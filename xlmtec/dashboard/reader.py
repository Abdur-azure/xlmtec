"""
xlmtec.dashboard.reader
~~~~~~~~~~~~~~~~~~~~~~~~
Reads training run metadata from an output directory.

HF Trainer writes:
  output_dir/trainer_state.json   — epoch, step, log history, best metric
  output_dir/config.yaml          — PipelineConfig used for the run (if saved)
  output_dir/eval_results.json    — final evaluation metrics (optional)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunMetrics:
    """Metrics recorded at a single training step."""

    step: int
    epoch: float
    train_loss: float | None = None
    eval_loss: float | None = None
    eval_rouge1: float | None = None
    eval_rouge2: float | None = None
    eval_rougeL: float | None = None
    eval_bleu: float | None = None
    learning_rate: float | None = None


@dataclass
class RunInfo:
    """Full metadata for a single training run."""

    name: str  # Directory name e.g. "run1"
    path: Path
    total_steps: int = 0
    total_epochs: float = 0.0
    best_metric: float | None = None
    best_metric_name: str = ""
    best_model_checkpoint: str = ""
    train_runtime_seconds: float = 0.0
    train_samples_per_second: float = 0.0
    history: list[RunMetrics] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    final_eval: dict = field(default_factory=dict)

    @property
    def best_eval_loss(self) -> float | None:
        losses = [h.eval_loss for h in self.history if h.eval_loss is not None]
        return min(losses) if losses else None

    @property
    def final_train_loss(self) -> float | None:
        losses = [h.train_loss for h in self.history if h.train_loss is not None]
        return losses[-1] if losses else None

    @property
    def has_eval_metrics(self) -> bool:
        return any(h.eval_rouge1 is not None or h.eval_bleu is not None for h in self.history)


class RunReader:
    """Read a training run from an output directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)

    def read(self) -> RunInfo:
        """Parse all available metadata from the run directory.

        Raises:
            FileNotFoundError: If the directory does not exist.
            ValueError: If trainer_state.json is missing or unreadable.
        """
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        state_file = self.run_dir / "trainer_state.json"
        if not state_file.exists():
            raise ValueError(
                f"No trainer_state.json in {self.run_dir}\n"
                "Make sure this is a completed training run directory."
            )

        state = json.loads(state_file.read_text(encoding="utf-8"))
        history = self._parse_history(state.get("log_history", []))
        config = self._read_config()
        final_eval = self._read_eval_results()

        return RunInfo(
            name=self.run_dir.name,
            path=self.run_dir,
            total_steps=state.get("global_step", 0),
            total_epochs=float(state.get("epoch", 0)),
            best_metric=state.get("best_metric"),
            best_metric_name=state.get("best_metric_key", ""),
            best_model_checkpoint=state.get("best_model_checkpoint", ""),
            train_runtime_seconds=state.get("train_runtime", 0.0),
            train_samples_per_second=state.get("train_samples_per_second", 0.0),
            history=history,
            config=config,
            final_eval=final_eval,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_history(self, log_history: list[dict]) -> list[RunMetrics]:
        metrics = []
        for entry in log_history:
            metrics.append(
                RunMetrics(
                    step=entry.get("step", 0),
                    epoch=float(entry.get("epoch", 0)),
                    train_loss=entry.get("loss"),
                    eval_loss=entry.get("eval_loss"),
                    eval_rouge1=entry.get("eval_rouge1"),
                    eval_rouge2=entry.get("eval_rouge2"),
                    eval_rougeL=entry.get("eval_rougeL"),
                    eval_bleu=entry.get("eval_bleu"),
                    learning_rate=entry.get("learning_rate"),
                )
            )
        return metrics

    def _read_config(self) -> dict:
        for name in ("config.yaml", "config.json"):
            f = self.run_dir / name
            if f.exists():
                try:
                    if name.endswith(".yaml"):
                        import yaml

                        return yaml.safe_load(f.read_text(encoding="utf-8")) or {}
                    return json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return {}

    def _read_eval_results(self) -> dict:
        f = self.run_dir / "eval_results.json"
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}
