# core/ — Context

The foundation layer. Every other subpackage depends on this. It has **zero internal deps** — imports only stdlib, pydantic, and typing.

## Files

| File | Purpose |
|------|---------|
| `types.py` | All enums and frozen dataclasses. The single source of truth for data shapes across the repo. |
| `config.py` | `PipelineConfig` (Pydantic) + `ConfigBuilder` (fluent API). Validation happens here — fail fast, not at training time. |
| `exceptions.py` | Full exception hierarchy rooted at `FineTuneError`. Every error raised anywhere in the codebase must extend this. |

## Rules

- **Never add imports from other subpackages** (`trainers`, `data`, etc.) — this creates circular deps.
- **All config dataclasses must be frozen** — callers must not mutate them after construction.
- `ConfigBuilder.build()` is the only valid way to construct a `PipelineConfig` from code. `from_yaml` / `from_json` are for file loading.
- Validation belongs in Pydantic model validators, not in trainers or CLI.

## Key enums (add new values here before implementing trainers)

- `TrainingMethod` — lora, qlora, full_finetuning, instruction_tuning, dpo (planned)
- `DatasetSource` — local_file, huggingface_hub
- `EvaluationMetric` — rouge1, rouge2, rougeL, bleu, perplexity

## Extension pattern

To add a new training method:
1. Add the value to `TrainingMethod` in `types.py`
2. Add any method-specific config dataclass to `types.py`
3. Add a Pydantic config model to `config.py` and wire it into `PipelineConfig`
4. Then implement the trainer in `trainers/`