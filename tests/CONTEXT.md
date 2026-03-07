# tests/ — Context

Unit tests (mocked, no GPU) + integration tests (real GPT-2, CPU ok).

## Files

| File | Covers |
|------|--------|
| `conftest.py` | Shared fixtures: mock_model (no torch), mock_tokenizer, tiny_dataset, tmp_output_dir |
| `test_config.py` | ConfigBuilder, PipelineConfig validation, YAML round-trip |
| `test_trainers.py` | LoRATrainer, TrainerFactory dispatch, MissingConfigError cases |
| `test_qlora_trainer.py` | QLoRATrainer factory dispatch, 4-bit warning, _setup_peft, train() |
| `test_full_trainer.py` | FullFineTuner, VRAM warning logic |
| `test_instruction_trainer.py` | format_instruction_dataset, InstructionTrainer skip-reformat logic |
| `test_dpo_trainer.py` | validate_dpo_dataset, DPOTrainer init/beta, trl ImportError guard |
| `test_recommend.py` | recommend CLI command, YAML output validity, qlora recommendation |
| `test_evaluation.py` | RougeMetric, BleuMetric, MetricRegistry, BenchmarkReport, BenchmarkRunner |
| `test_cli_train.py` | CLI train command wiring for all 5 methods, config file precedence |
| `test_merge.py` | lmtool merge: happy path, missing dir/config, dtype validation |
| `test_evaluate.py` | lmtool evaluate: output display, unknown metric, num-samples flag |
| `test_benchmark.py` | lmtool benchmark: summary output, run_comparison called |
| `test_upload.py` | lmtool upload: token, private, HF_TOKEN env, merge-adapter flow |
| `test_data.py` | detect_columns, quick_load, prepare_dataset, error types (no HF downloads) |
| `test_integration.py` | End-to-end: real GPT-2, 1 step, asserts adapter saved |

## Rules

- **`conftest.py` must not import torch at module level** — all unit tests must be
  collectable without torch installed. Use `MagicMock` with `param.numel.return_value = N`.
- **All unit tests mock HF Trainer and PEFT** — no real model loads, no GPU needed
- **patch() target = where the name is used, not where it's exported**
  e.g. `lmtool.data.pipeline.DataPipeline` not `lmtool.data.DataPipeline`
- **Integration tests are guarded** with `pytest.importorskip` at module level
- `conftest.py` at repo root handles `sys.path` — never rely on `PYTHONPATH`
- New trainers always get their own test file, not tacked onto `test_trainers.py`

## Running tests

```bash
# Fast (unit only)
pytest tests/ -v --ignore=tests/test_integration.py

# Verify collectable without torch
pytest tests/ --co -q --ignore=tests/test_integration.py

# Full (needs torch + transformers)
pytest tests/test_integration.py -v -s

# All
pytest tests/ -v
```