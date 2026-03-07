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
from .benchmarker import BenchmarkReport, BenchmarkRunner, EvaluationResult
from .metrics import BleuMetric, MetricRegistry, PerplexityMetric, RougeMetric

__all__ = [
    "RougeMetric", "BleuMetric", "PerplexityMetric", "MetricRegistry",
    "BenchmarkRunner", "BenchmarkReport", "EvaluationResult",
]
