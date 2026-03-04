# trainers/ — Context

All training logic lives here. Each trainer is a concrete subclass of `BaseTrainer`.

## Files

| File | Purpose |
|------|---------|
| `base.py` | `BaseTrainer` ABC + `TrainingResult` frozen dataclass. The `train()` method is final — subclasses only override `_setup_peft()`. |
| `lora_trainer.py` | `LoRATrainer` — attaches LoRA adapters, freezes base weights. |
| `qlora_trainer.py` | `QLoRATrainer` — extends LoRATrainer with 4-bit quantization via BitsAndBytes. |
| `full_trainer.py` | `FullFineTuner` — unfreezes all params, no PEFT. Issues VRAM warning for models >1B params. |
| `instruction_trainer.py` | `InstructionTrainer` — extends LoRATrainer. Reformats alpaca-style `{instruction, input, response}` datasets before training. Skips reformatting if `input_ids` or `text` column already present. |
| `dpo_trainer.py` | `DPOTrainer` — wraps TRL DPOTrainer with LoRA. Validates prompt/chosen/rejected columns. Requires trl>=0.7.0. |
| `factory.py` | `TrainerFactory` — single entry point. Validates required configs, selects and instantiates the right trainer. |

## Adding a new trainer

1. Extend `BaseTrainer` (or `LoRATrainer` if PEFT-based)
2. Implement `_setup_peft(model) -> model` — this is the only required override
3. Add dispatch in `TrainerFactory.create()`
4. Export from `__init__.py`
5. Write mocked unit tests — no GPU, no HF downloads

## Rules

- `TrainingResult` is **frozen** — never add mutable fields
- `output_dir` must always be populated — downstream CLI depends on it
- VRAM / memory warnings belong in `__init__`, not `_setup_peft`
- Never call `hf_trainer.train()` directly outside `base.py` — use `TrainerFactory.train()`

## Mid-sprint fix — skip condition in `_maybe_format`

`InstructionTrainer._maybe_format()` skips reformatting in two cases:
1. Dataset already has a `text` column → pre-formatted, skip
2. Dataset already has `input_ids` → pre-tokenized, skip entirely

This was added to fix integration test failures where a tokenized dataset
was passed directly but InstructionTrainer tried to reformat it and crashed
with "instruction/response columns not found".

**Rule:** Always check for `input_ids` before checking for `text` in any
trainer that does dataset transformation.