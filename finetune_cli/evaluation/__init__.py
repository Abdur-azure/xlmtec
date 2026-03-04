"""
Evaluation package for model assessment and benchmarking.

Provides comprehensive evaluation capabilities:
- Multiple metrics (ROUGE, BLEU, Perplexity, F1, Exact Match)
- Model comparison and benchmarking
- Report generation (Markdown, JSON, HTML)
- Quick evaluation utilities

High-level interface:
- evaluate_model: Comprehensive evaluation
- quick_evaluate: Fast evaluation without config
- benchmark_models: Compare base vs fine-tuned
"""
from .metrics import RougeMetric, BleuMetric, PerplexityMetric, MetricRegistry
from .benchmarker import BenchmarkRunner, BenchmarkReport, EvaluationResult

__all__ = [
    "RougeMetric", "BleuMetric", "PerplexityMetric", "MetricRegistry",
    "BenchmarkRunner", "BenchmarkReport", "EvaluationResult",
]