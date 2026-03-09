# xlmtec/export — Context

Model export backends powering `xlmtec export`.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Docstring only — no re-exports (avoids circular import) |
| `formats.py` | `ExportFormat` enum + `FormatMeta` per-format metadata |
| `exporter.py` | `ModelExporter` — dispatches to format backends |
| `backends/__init__.py` | Empty |
| `backends/safetensors.py` | Export via `safetensors` library |
| `backends/onnx.py` | Export via `optimum[exporters]` |
| `backends/gguf.py` | Export via llama.cpp `convert_hf_to_gguf.py` script |
| `CONTEXT.md` | This file |

## Commands

```bash
xlmtec export output/run1 --format safetensors --output exported/
xlmtec export output/run1 --format onnx --quantize fp16
xlmtec export output/run1 --format gguf --quantize q4_0
xlmtec export output/run1 --format safetensors --dry-run
```

## Supported Formats

| Format | Extra | Quantise options |
|--------|-------|-----------------|
| `safetensors` | none (always available) | — |
| `onnx` | `pip install xlmtec[onnx]` | `fp32`, `fp16`, `int8` |
| `gguf` | `pip install xlmtec[gguf]` + llama.cpp | `q4_0`, `q4_1`, `q8_0`, `f16` |

## CRITICAL Rules

- **`export/__init__.py` must stay docstring-only.** Any re-export from submodules causes a circular import at pytest collection time.
- **Optional dep imports must come AFTER the `dry_run` early return.** If `optimum` is not installed, dry-run must still work.
- Import pattern in backends:
  ```python
  if dry_run:
      return result          # ← early exit BEFORE any optional import
  from optimum.exporters... import main_export   # ← only reached on real export
  ```

## Extension pattern

To add a new export format:
1. Add a value to `ExportFormat` in `formats.py`
2. Add `FormatMeta` entry to `FORMAT_META`
3. Create `backends/yourformat.py` with a `export_yourformat()` function
4. Add a dispatch branch in `ModelExporter.export()`
5. Add tests to `tests/test_export.py`
6. Add optional extras to `pyproject.toml` if needed