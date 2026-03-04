# CLAUDE.md

Context file for AI-assisted development. Read this at the start of every session.

---

## Project

**finetune-cli** — production-grade LLM fine-tuning framework with a modular CLI.
Version: 3.8.0 | License: MIT | Python: 3.10+

---

## Repo layout

```
finetune_cli/          # Main package
  core/                # Types, config, exceptions (no deps on other subpackages)
  data/                # Dataset loading, tokenization, pipeline
  models/              # Model + tokenizer loading, target module detection
  trainers/            # LoRA, QLoRA, Full, Instruction, DPO trainers + factory
  evaluation/          # ROUGE, BLEU, Perplexity metrics + BenchmarkRunner
  cli/                 # Typer CLI — train, evaluate, benchmark, upload, recommend, merge
  utils/               # Logging only

tests/                 # Unit + integration tests (all absolute imports)
examples/
  configs/             # Five runnable YAML configs (lora, qlora, instruction, full, dpo)
  generate_sample_data.py  # Creates data/sample.jsonl + data/instructions.jsonl + dpo_sample.jsonl

docs/                  # MkDocs source (Material theme)
tasks/                 # todo.md + lessons.md — read these every session
```

---

## Architecture rules — enforce these in every change

1. **Dependency direction is one-way**: `cli → trainers/evaluation/data → models → core`. Never import upward or sideways between subpackages.
2. **`core/` has zero internal deps** — it imports only stdlib and pydantic.
3. **All config objects are frozen dataclasses** (`@dataclass(frozen=True)`). Never mutate them.
4. **All errors extend `FineTuneError`** from `core/exceptions.py`. Never raise raw builtins from module code.
5. **Always use `get_logger(__name__)`** from `utils/logging.py`. Never `logging.getLogger` directly.
6. **TrainingResult must carry `output_dir: Path`** — downstream CLI and evaluation depend on it.
7. **`trainers/__init__.py` must export every new trainer** — CLI and tests import from there.
8. **`data/__init__.py` must mirror CLI imports exactly** — trace the full import chain when adding exports.
9. **All test files use absolute imports** — `from finetune_cli.x import y`, never `from ..x import y`.
10. **No real torch tensors in unit tests** — use `MagicMock` with `param.numel.return_value = N`.

---

## Key types (read before touching core/)

| Type | Location | Purpose |
|------|----------|---------|
| `TrainingMethod` | `core/types.py` | Enum of all training methods (21 values; 5 implemented) |
| `PipelineConfig` | `core/config.py` | Top-level Pydantic config |
| `ConfigBuilder` | `core/config.py` | Fluent builder for PipelineConfig |
| `TrainingConfig` | `core/types.py` | Frozen dataclass for training hyper-params |
| `LoRAConfig` | `core/types.py` | Frozen dataclass for LoRA params |
| `TrainingResult` | `trainers/base.py` | Frozen dataclass returned by all trainers |
| `EvaluationResult` | `evaluation/benchmarker.py` | Frozen dataclass, per-model scores |
| `BenchmarkReport` | `evaluation/benchmarker.py` | Before/after comparison |

---

## Implemented trainers

| Method | Class | Needs lora_config |
|--------|-------|-------------------|
| `lora` | `LoRATrainer` | Yes |
| `qlora` | `QLoRATrainer` | Yes (+ model_config with load_in_4bit) |
| `full_finetuning` | `FullFineTuner` | No |
| `instruction_tuning` | `InstructionTrainer` | Yes |
| `dpo` | `DPOTrainer` | Yes (requires trl>=0.7.0) |
| `vanilla_distillation` | `ResponseDistillationTrainer` | No (needs distillation_config) |
| `feature_distillation` | `FeatureDistillationTrainer` | No (needs feature_distillation_config) |

---

## Adding a new trainer — checklist

1. Create `finetune_cli/trainers/<n>_trainer.py`, extend `BaseTrainer`
2. Implement `_setup_peft(model) -> model`
3. Add the new `TrainingMethod` enum value to `core/types.py` if missing
4. Wire the new method in `trainers/factory.py` `TrainerFactory.create()`
5. Export from `trainers/__init__.py`
6. Add to `_LORA_METHODS` in `cli/main.py` if it needs a LoRA config
7. Write `tests/test_<n>_trainer.py` — mock HF Trainer, no GPU required, absolute imports
8. Add a CLI test to `tests/test_cli_train.py` asserting `exit_code == 0`
9. Add example config to `examples/configs/<n>.yaml` pointing at local data
10. Update `docs/api.md` and `docs/configuration.md`
11. Update `CHANGELOG.md` and `audit_repo.py`

---

## Test commands

```bash
# Unit tests (fast, no GPU)
pytest tests/ -v --ignore=tests/test_integration.py

# Verify collectable without torch
pytest tests/ --co -q --ignore=tests/test_integration.py

# Integration tests (requires torch + transformers, CPU ok)
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
| 25 | TUI Foundation | ✅ Complete |

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