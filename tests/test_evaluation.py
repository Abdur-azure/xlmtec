"""Unit tests for evaluation metrics and BenchmarkRunner."""

from typing import List
from unittest.mock import MagicMock, patch
import math

import pytest

from finetune_cli.core.types import EvaluationMetric, EvaluationConfig
from finetune_cli.evaluation.metrics import RougeMetric, BleuMetric, MetricRegistry
from finetune_cli.evaluation.benchmarker import BenchmarkRunner, BenchmarkReport, EvaluationResult


# ============================================================================
# ROUGE
# ============================================================================


class TestRougeMetric:
    """RougeMetric returns sensible scores."""

    def test_perfect_match_is_one(self):
        metric = RougeMetric(EvaluationMetric.ROUGE_L)
        score = metric.compute(["hello world"], ["hello world"])
        assert score == pytest.approx(1.0)

    def test_zero_overlap_is_zero(self):
        metric = RougeMetric(EvaluationMetric.ROUGE_L)
        score = metric.compute(["abc def"], ["xyz uvw"])
        assert score == pytest.approx(0.0)

    def test_partial_overlap_between_zero_and_one(self):
        metric = RougeMetric(EvaluationMetric.ROUGE_1)
        score = metric.compute(["the cat sat"], ["the dog sat"])
        assert 0.0 < score < 1.0

    def test_multiple_samples_averaged(self):
        metric = RougeMetric(EvaluationMetric.ROUGE_L)
        preds = ["hello world", "foo bar"]
        refs = ["hello world", "foo bar"]
        score = metric.compute(preds, refs)
        assert score == pytest.approx(1.0)

    def test_name_property(self):
        metric = RougeMetric(EvaluationMetric.ROUGE_2)
        assert metric.name == EvaluationMetric.ROUGE_2


# ============================================================================
# BLEU
# ============================================================================


class TestBleuMetric:
    """BleuMetric returns sensible corpus scores."""

    def test_perfect_match_high_score(self):
        metric = BleuMetric()
        preds = ["the quick brown fox"] * 5
        refs = ["the quick brown fox"] * 5
        score = metric.compute(preds, refs)
        assert score > 0.9

    def test_empty_overlap_near_zero(self):
        metric = BleuMetric()
        preds = ["aaa bbb ccc"]
        refs = ["xxx yyy zzz"]
        score = metric.compute(preds, refs)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_name_property(self):
        assert BleuMetric().name == EvaluationMetric.BLEU


# ============================================================================
# METRIC REGISTRY
# ============================================================================


class TestMetricRegistry:
    def test_returns_rouge_for_rouge_variant(self):
        metric = MetricRegistry.get(EvaluationMetric.ROUGE_L)
        assert isinstance(metric, RougeMetric)

    def test_returns_bleu_for_bleu(self):
        metric = MetricRegistry.get(EvaluationMetric.BLEU)
        assert isinstance(metric, BleuMetric)

    def test_perplexity_requires_model(self):
        with pytest.raises(ValueError, match="model and tokenizer"):
            MetricRegistry.get(EvaluationMetric.PERPLEXITY)

    def test_unsupported_metric_raises(self):
        with pytest.raises(NotImplementedError):
            MetricRegistry.get(EvaluationMetric.ACCURACY)


# ============================================================================
# BENCHMARK REPORT
# ============================================================================


class TestBenchmarkReport:
    """BenchmarkReport computes improvements correctly."""

    def _make_report(self, baseline_scores, finetuned_scores) -> BenchmarkReport:
        return BenchmarkReport(
            baseline=EvaluationResult("base", baseline_scores, 100, 1.0),
            finetuned=EvaluationResult("ft", finetuned_scores, 100, 1.0),
        )

    def test_improvements_positive_when_finetuned_higher(self):
        report = self._make_report({"rougeL": 0.3}, {"rougeL": 0.5})
        assert report.improvements["rougeL"] == pytest.approx(0.2)

    def test_improvements_negative_when_finetuned_lower(self):
        report = self._make_report({"rougeL": 0.5}, {"rougeL": 0.3})
        assert report.improvements["rougeL"] == pytest.approx(-0.2)

    def test_summary_contains_metric_name(self):
        report = self._make_report({"rougeL": 0.3}, {"rougeL": 0.4})
        assert "rougeL" in report.summary()

    def test_summary_contains_both_scores(self):
        report = self._make_report({"bleu": 0.25}, {"bleu": 0.35})
        summary = report.summary()
        assert "0.2500" in summary
        assert "0.3500" in summary


# ============================================================================
# BENCHMARK RUNNER (mocked model)
# ============================================================================


class TestBenchmarkRunner:
    """BenchmarkRunner calls evaluate and assembles a report."""

    def _make_config(self, metrics=None):
        return EvaluationConfig(
            metrics=metrics or [EvaluationMetric.ROUGE_L],
            batch_size=2,
            num_samples=4,
            generation_max_length=20,
            generation_do_sample=False,
        )

    @patch.object(BenchmarkRunner, "_generate")
    def test_evaluate_returns_result(self, mock_generate, tiny_dataset, mock_tokenizer):
        mock_generate.return_value = ["hello world"] * len(tiny_dataset)
        runner = BenchmarkRunner(self._make_config(), mock_tokenizer)
        result = runner.evaluate(MagicMock(), tiny_dataset, text_column="text")
        assert isinstance(result, EvaluationResult)
        assert EvaluationMetric.ROUGE_L.value in result.scores

    @patch.object(BenchmarkRunner, "_generate")
    def test_run_comparison_returns_report(self, mock_generate, tiny_dataset, mock_tokenizer):
        mock_generate.return_value = ["hello world"] * len(tiny_dataset)
        runner = BenchmarkRunner(self._make_config(), mock_tokenizer)
        report = runner.run_comparison(MagicMock(), MagicMock(), tiny_dataset, text_column="text")
        assert isinstance(report, BenchmarkReport)
        assert report.baseline.model_label == "base"
        assert report.finetuned.model_label == "fine-tuned"