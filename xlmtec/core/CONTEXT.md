# xlmtec/core — Context

Foundation layer. Zero internal deps — stdlib, pydantic, typing only.
Every other subpackage imports from here. Never import upward.

## Files

| File | Purpose |
|------|---------|
| `types.py` | Enums + frozen dataclasses. Single source of truth for all data shapes. |
| `config.py` | Pydantic validation models + `PipelineConfig`. Validation happens here — fail fast. |
| `config_builder.py` | `ConfigBuilder` fluent API. The only correct way to build config from code. |
| `exceptions.py` | Full exception hierarchy rooted at `FineTuneError`. |
| `CONTEXT.md` | This file. |

## Rules

1. **No torch at module level** — torch is an optional ML dep. Import inside functions if needed.
2. **No imports from other subpackages** — no trainers, data, models, cli.
3. **All dataclasses must be frozen** — callers must never mutate after construction.
4. **Only add a TrainingMethod value when the trainer exists** — no phantom enum values.
5. **Validation belongs in Pydantic validators** — not in trainers or CLI.
6. `ConfigBuilder.build()` → only way to build `PipelineConfig` from code.
7. `PipelineConfig.from_yaml()` / `from_json()` → file-based loading only.

## TrainingMethod ↔ Trainer mapping

| Enum value | Trainer class | File |
|------------|--------------|------|
| `lora` | `LoRATrainer` | `trainers/lora_trainer.py` |
| `qlora` | `QLoRATrainer` | `trainers/qlora_trainer.py` |
| `full_finetuning` | `FullFineTuner` | `trainers/full_trainer.py` |
| `instruction_tuning` | `InstructionTrainer` | `trainers/instruction_trainer.py` |
| `dpo` | `DPOTrainer` | `trainers/dpo_trainer.py` |
| `vanilla_distillation` | `ResponseDistillationTrainer` | `trainers/response_distillation_trainer.py` |
| `feature_distillation` | `FeatureDistillationTrainer` | `trainers/feature_distillation_trainer.py` |
| `structured_pruning` | `StructuredPruner` | `trainers/structured_pruner.py` |

## Extension pattern

Adding a new training method:
1. Add value to `TrainingMethod` in `types.py`
2. Add config dataclass to `types.py` if needed
3. Add Pydantic model to `config.py` and wire into `PipelineConfig`
4. Implement the trainer in `trainers/`
5. Register in `trainers/factory.py`