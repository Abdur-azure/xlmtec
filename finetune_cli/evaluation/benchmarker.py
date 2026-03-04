"""
Benchmarking pipeline — computes metrics before and after fine-tuning
and produces a structured comparison report.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional
import time

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.types import EvaluationConfig, EvaluationMetric
from ..utils.logging import get_logger
from .metrics import MetricRegistry, Metric


logger = get_logger(__name__)


# ============================================================================
# RESULT TYPES
# ============================================================================


@dataclass(frozen=True)
class EvaluationResult:
    """Scores for a single model evaluation pass."""

    model_label: str
    """Human-readable label, e.g. 'base' or 'fine-tuned'."""

    scores: Dict[str, float]
    """metric_name → score."""

    num_samples: int
    """Number of samples evaluated."""

    elapsed_seconds: float
    """Wall-clock time for this evaluation."""


@dataclass(frozen=True)
class BenchmarkReport:
    """Before/after comparison report."""

    baseline: EvaluationResult
    finetuned: EvaluationResult

    @property
    def improvements(self) -> Dict[str, float]:
        """
        Absolute score change (finetuned − baseline) for each metric.
        Negative values indicate degradation.
        """
        result = {}
        for k, v in self.finetuned.scores.items():
            baseline_v = self.baseline.scores.get(k, 0.0)
            result[k] = v - baseline_v
        return result

    def summary(self) -> str:
        """Formatted multi-line summary string."""
        lines = [
            "=" * 60,
            " BENCHMARK REPORT",
            "=" * 60,
            f"{'Metric':<18} {'Baseline':>10} {'Fine-tuned':>12} {'Δ':>8}",
            "-" * 60,
        ]
        for metric in self.baseline.scores:
            base_v = self.baseline.scores[metric]
            ft_v = self.finetuned.scores.get(metric, float("nan"))
            delta = ft_v - base_v
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
            lines.append(
                f"{metric:<18} {base_v:>10.4f} {ft_v:>12.4f} {arrow}{abs(delta):>6.4f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================


class BenchmarkRunner:
    """
    Evaluates a model on a dataset with configurable metrics.

    Usage::

        runner = BenchmarkRunner(eval_config, tokenizer)
        baseline = runner.evaluate(base_model, dataset, label="base")
        finetuned = runner.evaluate(ft_model, dataset, label="fine-tuned")
        report = BenchmarkReport(baseline, finetuned)
        print(report.summary())
    """

    def __init__(
        self,
        config: EvaluationConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        label: str = "model",
        text_column: str = "text",
    ) -> EvaluationResult:
        """
        Run all configured metrics on *dataset* using *model*.

        Args:
            model: The model to evaluate (base or fine-tuned).
            dataset: Dataset with a text column.
            label: Human-readable label for the result.
            text_column: Column containing reference text.

        Returns:
            EvaluationResult with per-metric scores.
        """
        self.logger.info(f"Evaluating '{label}' on {len(dataset)} samples...")

        # Limit samples
        samples = dataset
        if self.config.num_samples and len(dataset) > self.config.num_samples:
            samples = dataset.select(range(self.config.num_samples))

        references: List[str] = samples[text_column]
        predictions: List[str] = self._generate(model, references)

        scores: Dict[str, float] = {}
        start = time.time()

        for metric_enum in self.config.metrics:
            try:
                metric = MetricRegistry.get(
                    metric_enum,
                    model=model,
                    tokenizer=self.tokenizer,
                    batch_size=self.config.batch_size,
                    max_length=self.config.generation_max_length,
                )
                score = metric.compute(predictions, references)
                scores[metric_enum.value] = score
                self.logger.info(f"  {metric_enum.value}: {score:.4f}")
            except Exception as exc:
                self.logger.warning(f"  {metric_enum.value}: FAILED — {exc}")
                scores[metric_enum.value] = float("nan")

        return EvaluationResult(
            model_label=label,
            scores=scores,
            num_samples=len(samples),
            elapsed_seconds=time.time() - start,
        )

    def run_comparison(
        self,
        base_model: PreTrainedModel,
        finetuned_model: PreTrainedModel,
        dataset: Dataset,
        text_column: str = "text",
    ) -> BenchmarkReport:
        """
        Evaluate both models and return a comparison report.
        """
        baseline = self.evaluate(base_model, dataset, label="base", text_column=text_column)
        finetuned = self.evaluate(finetuned_model, dataset, label="fine-tuned", text_column=text_column)
        report = BenchmarkReport(baseline=baseline, finetuned=finetuned)
        self.logger.info("\n" + report.summary())
        return report

    # ------------------------------------------------------------------
    # Internal generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        model: PreTrainedModel,
        prompts: List[str],
    ) -> List[str]:
        """Generate text for each prompt using the model."""
        import torch

        model.eval()
        results = []
        device = next(model.parameters()).device
        cfg = self.config

        for i in range(0, len(prompts), self.config.batch_size):
            batch = prompts[i : i + self.config.batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=cfg.generation_max_length,
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=cfg.generation_max_length,
                    temperature=cfg.generation_temperature,
                    top_p=cfg.generation_top_p,
                    do_sample=cfg.generation_do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.batch_decode(
                output_ids[:, input_len:], skip_special_tokens=True
            )
            results.extend(decoded)

        return results