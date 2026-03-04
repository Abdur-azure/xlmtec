# evaluation/ — Context

Metrics, evaluation runner, and before/after benchmark reports.

## Files

| File | Purpose |
|------|---------|
| `metrics.py` | `RougeMetric`, `BleuMetric`, `PerplexityMetric`, `MetricRegistry`. Each metric has a `compute(predictions, references) -> float` method. |
| `benchmarker.py` | `BenchmarkRunner`, `BenchmarkReport`, `EvaluationResult`. |
| `base.py` | Abstract `Metric`, `Evaluator`, `Benchmarker` classes. |

## Key types

- `EvaluationResult` — frozen dataclass: `{model_label, scores: Dict[str, float], num_samples, elapsed_seconds}`
- `BenchmarkReport` — frozen dataclass: `{baseline: EvaluationResult, finetuned: EvaluationResult}`. Has `.improvements` property and `.summary()` method.

## Usage pattern

```python
runner = BenchmarkRunner(eval_config, tokenizer)
result = runner.evaluate(model, dataset, label="fine-tuned")
report = runner.run_comparison(base_model, ft_model, dataset)
print(report.summary())  # formatted table with ▲/▼ indicators
```

## Adding a new metric

1. Add value to `EvaluationMetric` enum in `core/types.py`
2. Implement class in `metrics.py` extending `Metric`
3. Register in `MetricRegistry`
4. Export from `evaluation/__init__.py`