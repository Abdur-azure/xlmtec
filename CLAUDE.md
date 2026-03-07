# CLAUDE.md

Context file for AI-assisted development. Read this at the start of every session.

---

## Project

**xlmtec** — production-grade LLM fine-tuning framework with a modular CLI.
Version: 3.16.0 | License: MIT | Python: 3.10+

---

## Repo layout

```
xlmtec/              # Main package
  core/                    # Types, config, exceptions (no deps on other subpackages)
  data/                    # Dataset loading, tokenization, pipeline
  models/                  # Model + tokenizer loading, target module detection
  trainers/                # All trainers + pruners + factory
  evaluation/              # ROUGE, BLEU, Perplexity metrics + BenchmarkRunner
  cli/                     # Typer CLI — all subcommands live in xlmtec/cli/main.py
  tui/                     # Textual interactive terminal UI
  utils/                   # Logging only

tests/                     # Unit + integration tests (all absolute imports)
examples/
  configs/                 # Runnable YAML configs (one per training method)
  generate_sample_data.py  # Creates data/*.jsonl — stdlib only, no ML deps

docs/                      # MkDocs source
tasks/                     # todo.md + lessons.md — read every session
```

---

## Architecture rules — enforce in every change

1. **Dependency direction is one-way**: `cli/tui → trainers/evaluation/data → models → core`. Never import upward or sideways between subpackages.
2. **`core/` has zero internal deps** — imports only stdlib and pydantic.
3. **All config objects are frozen dataclasses** (`@dataclass(frozen=True)`). Never mutate after construction.
4. **All errors extend `FineTuneError`** from `xlmtec/core/exceptions.py`. Never raise raw builtins from module code.
5. **Always use `get_logger(__name__)`** from `xlmtec/utils/logging.py`. Never `logging.getLogger` directly.
6. **`TrainingResult` must carry `output_dir: Path`** — downstream CLI and evaluation depend on it.
7. **`xlmtec/trainers/__init__.py` must export every new trainer** — CLI and tests import from there.
8. **`xlmtec/data/__init__.py` must mirror CLI imports exactly** — trace the full import chain when adding exports.
9. **All test files use absolute imports** — `from xlmtec.x import y`, never `from ..x import y`.
10. **No real torch tensors in unit tests** — use `MagicMock` with `param.numel.return_value = N`.
11. **Heavy deps (`torch`, `transformers`) are optional** — guard all imports with try/except or lazy-import inside functions. Core CLI must work without the `[ml]` extra installed.

---

## Install options

```bash
pip install -e ".[dev]"        # lightweight: runs unit tests, linting — no torch
pip install -e ".[ml,tui,dev]" # full: trains, TUI, all tests
pip install -e ".[full]"       # everything including docs
```

---

## Key types (read before touching core/)

| Type | Location | Purpose |
|------|----------|---------|
| `TrainingMethod` | `xlmtec/core/types.py` | Enum of all training methods |
| `PipelineConfig` | `xlmtec/core/config.py` | Top-level Pydantic config |
| `ConfigBuilder` | `xlmtec/core/config.py` | Fluent builder for PipelineConfig |
| `TrainingConfig` | `xlmtec/core/types.py` | Frozen dataclass for training hyper-params |
| `LoRAConfig` | `xlmtec/core/types.py` | Frozen dataclass for LoRA params |
| `DistillationConfig` | `xlmtec/core/types.py` | Frozen dataclass for response distillation |
| `FeatureDistillationConfig` | `xlmtec/core/types.py` | Frozen dataclass for feature distillation |
| `PruningConfig` | `xlmtec/core/types.py` | Frozen dataclass for structured pruning |
| `WandaConfig` | `xlmtec/core/types.py` | Frozen dataclass for WANDA pruning |
| `TrainingResult` | `xlmtec/trainers/base.py` | Frozen dataclass returned by all trainers |
| `PruningResult` | `xlmtec/trainers/structured_pruner.py` | Returned by StructuredPruner |
| `WandaResult` | `xlmtec/trainers/wanda_pruner.py` | Returned by WandaPruner |
| `EvaluationResult` | `xlmtec/evaluation/benchmarker.py` | Per-model scores |
| `BenchmarkReport` | `xlmtec/evaluation/benchmarker.py` | Before/after comparison |

---

## Implemented trainers

| Method | Class | File | Config needed |
|--------|-------|------|---------------|
| `lora` | `LoRATrainer` | `xlmtec/trainers/lora_trainer.py` | `lora_config` |
| `qlora` | `QLoRATrainer` | `xlmtec/trainers/qlora_trainer.py` | `lora_config` + `model_config` |
| `full_finetuning` | `FullFineTuner` | `xlmtec/trainers/full_trainer.py` | none |
| `instruction_tuning` | `InstructionTrainer` | `xlmtec/trainers/instruction_trainer.py` | `lora_config` |
| `dpo` | `DPOTrainer` | `xlmtec/trainers/dpo_trainer.py` | `lora_config` + `trl` dep |
| `vanilla_distillation` | `ResponseDistillationTrainer` | `xlmtec/trainers/response_distillation_trainer.py` | `distillation_config` |
| `feature_distillation` | `FeatureDistillationTrainer` | `xlmtec/trainers/feature_distillation_trainer.py` | `feature_distillation_config` |

## Implemented pruners

| Command | Class | File |
|---------|-------|------|
| `xlmtec prune` | `StructuredPruner` | `xlmtec/trainers/structured_pruner.py` |
| `xlmtec wanda` | `WandaPruner` | `xlmtec/trainers/wanda_pruner.py` |

---

## Adding a new trainer — checklist

1. Create `xlmtec/trainers/<n>_trainer.py`, extend `BaseTrainer`
2. Implement `_setup_peft(model) -> model`
3. Add the new `TrainingMethod` enum value to `xlmtec/core/types.py` if missing
4. Wire the new method in `xlmtec/trainers/factory.py` → `TrainerFactory.create()`
5. Export from `xlmtec/trainers/__init__.py`
6. Add to `_LORA_METHODS` in `xlmtec/cli/main.py` if it needs a LoRA config
7. Write `tests/test_<n>_trainer.py` — mock HF Trainer, no GPU required, absolute imports
8. Add a CLI test to `tests/test_cli_train.py` asserting `exit_code == 0`
9. Add example config to `examples/configs/<n>.yaml` pointing at local data
10. Update `docs/api.md` and `docs/configuration.md`
11. Update `CHANGELOG.md` and `audit_repo.py`

---

## Test commands

```bash
# Unit tests only — no torch needed
pytest tests/ -v --ignore=tests/test_integration.py

# Verify all tests are collectable without torch
pytest tests/ --co -q --ignore=tests/test_integration.py

# Integration tests (CPU ok, downloads GPT-2 once ~500MB)
pytest tests/test_integration.py -v -s

# Full suite
pytest tests/ -v
```

---

## Sprint history

| Sprint | Name | Status |
|--------|------|--------|
| 1 | Stable Foundation | ✅ Complete |
| 2 | Expand | ✅ Complete |
| 3 | First Run | ✅ Complete |
| 4 | Hardened | ✅ Complete |
| 5 | Merge & Release | ✅ Complete |
| 6 | Documented | ✅ Complete |
| 7 | CI Tight | ✅ Complete |
| 8 | DPO | ✅ Complete |
| 9 | Housekeeping | ✅ Complete |
| 10 | DPO Runnable | ✅ Complete |
| 11 | Version Sync | ✅ Complete |
| 12 | Usage Guide Current | ✅ Complete |
| 13 | Test Coverage Complete | ✅ Complete |
| 14 | Sprint 13 Close-out | ✅ Complete |
| 15 | QLoRA Tests + Sync | ✅ Complete |
| 16 | Data Pipeline Tests | ✅ Complete |
| 18 | conftest Hardening | ✅ Complete |
| 19 | Test Import Audit | ✅ Complete |
| 20 | Import Audit Complete | ✅ Complete |
| 21 | Meta Sync | ✅ Complete |
| 23 | Response Distillation | ✅ Complete |
| 24 | Feature Distillation | ✅ Complete |
| 25 | TUI Foundation | ✅ Complete |
| 26 | TUI Train & Recommend | ✅ Complete |
| 27 | TUI Evaluate, Benchmark, Merge | ✅ Complete |
| 28 | TUI Upload + Polish | ✅ Complete |
| 29 | Structured Pruning | ✅ Complete |
| 30 | WANDA Pruning | ✅ Complete |
| 31 | Integration Hardening | ✅ Complete |
| 32 | Stabilise | ✅ Complete |

Current task state: `tasks/todo.md`
Accumulated lessons: `tasks/lessons.md`

---

## Workflow expectations

- Read `tasks/todo.md` and `tasks/lessons.md` before starting any session
- Read `CONTEXT.md` for whichever subpackage you're touching
- Run AST parse verification before handing off any new file
- Never mark a task complete without proving it works
- Update `tasks/lessons.md` after every correction
- Run `python audit_repo.py` before closing any sprint