# xlmtec/inference — Context

Batch inference pipeline powering `xlmtec predict`.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Docstring only |
| `data_loader.py` | `DataLoader` — reads JSONL/CSV, auto-detects text column |
| `writer.py` | `PredictionWriter` — writes predictions to JSONL or CSV |
| `predictor.py` | `BatchPredictor` — batched text generation using HF pipeline |
| `CONTEXT.md` | This file |

## Commands

```bash
xlmtec predict output/run1 --data data/test.jsonl --dry-run
xlmtec predict output/run1 --data data/test.jsonl --output predictions.jsonl
xlmtec predict output/run1 --data data/test.csv --output predictions.csv --format csv
xlmtec predict output/run1 --data data/test.jsonl --batch-size 16 --max-new-tokens 256
```

## Auto-detected Text Columns (priority order)

`text` → `input` → `prompt` → `sentence` → `content` → `question` → `context` → `document` → `instruction`

## Key Types

| Type | Fields |
|------|--------|
| `PredictConfig` | `model_dir`, `data_path`, `output_path`, `output_format`, `text_column`, `batch_size`, `max_new_tokens`, `temperature`, `device` |
| `PredictionResult` | `total_records`, `output_path`, `elapsed_seconds` |

## Rules

- `__init__.py` must stay docstring-only — same circular import risk as `export/`.
- `transformers` / `torch` are imported **inside `BatchPredictor.predict()`** only — keeps dry-run fast with no GPU stack.
- `DataLoader` raises `ValueError` if no text column can be auto-detected — always provide `--text-column` for non-standard schemas.
- `PredictionWriter` creates parent directories automatically.
- Tests use `generate_inference_data.py` — no real model downloads.

## Extension pattern

To support a new input format (e.g. Parquet):
1. Add detection in `DataLoader._load()`
2. Add a test in `tests/test_inference.py`
3. Update `--data` help text in `cli/commands/predict.py`