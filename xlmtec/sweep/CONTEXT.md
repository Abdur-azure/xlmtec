# xlmtec/sweep — Context

Hyperparameter sweep system powering `xlmtec sweep`.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Docstring only |
| `config.py` | `ParamSpec` + `SweepConfig` — parsed from the `sweep:` YAML section |
| `runner.py` | `SweepRunner` — runs Optuna trials, applies params, calls `TrainerFactory.train()` |
| `CONTEXT.md` | This file |

## Commands

```bash
# Validate config and print plan
xlmtec sweep examples/configs/sweep_lora_gpt2.yaml --dry-run

# Run 20 trials
xlmtec sweep examples/configs/sweep_lora_gpt2.yaml --trials 20

# Use trial count from config file
xlmtec sweep my_sweep.yaml
```

## Sweep YAML Format

A sweep YAML is a normal `PipelineConfig` YAML with an extra `sweep:` section:

```yaml
# ── base PipelineConfig ───────────────────────────────────────────────────
model:
  name: gpt2
dataset:
  source: local_file
  path: ./data/sample.jsonl
tokenization:
  max_length: 128
training:
  method: lora
  output_dir: ./sweep_output
  num_epochs: 1
lora:
  r: 8
  lora_alpha: 16

# ── sweep section ─────────────────────────────────────────────────────────
sweep:
  n_trials: 20
  metric: train_loss        # field on TrainingResult
  direction: minimize       # minimize | maximize
  output_dir: ./sweep_results
  sampler: tpe              # tpe | random (grid not yet supported)
  params:
    training.learning_rate:
      type: float
      low: 1.0e-5
      high: 1.0e-3
      log: true
    training.batch_size:
      type: categorical
      choices: [2, 4, 8]
    lora.r:
      type: int
      low: 4
      high: 32
```

## Param Path Syntax

Keys in `params:` use dot notation to target nested config fields:

| Path | Resolves to |
|------|-------------|
| `training.learning_rate` | `PipelineConfig.training.learning_rate` |
| `training.batch_size` | `PipelineConfig.training.batch_size` |
| `lora.r` | `PipelineConfig.lora.r` |
| `lora.lora_alpha` | `PipelineConfig.lora.lora_alpha` |

## Supported Metric Values

`metric` must match a field on `TrainingResult`:

| Metric | Notes |
|--------|-------|
| `train_loss` | Final training loss — always available |
| `eval_loss` | Requires `evaluation_strategy != "no"` in training config |
| `steps_completed` | Useful for debugging sweep stop conditions |

## Dependency

```bash
pip install xlmtec[sweep]   # installs optuna>=3.0.0
```

## Rules

- `sweep/__init__.py` must stay docstring-only — same circular import risk as `export/`.
- Optuna is imported **lazily inside `SweepRunner.run()`** only — keeps `--help` and `--dry-run` fast.
- Each trial saves to `{sweep.output_dir}/trial_{N}/` — never shares an output directory.
- Failed trials raise `optuna.TrialPruned` — the study continues; `SweepResult.n_failed` tracks them.
- `grid` sampler is not yet supported (requires explicit search space construction).

## Extension Pattern

To add a new sampler:
1. Add the sampler name to `_VALID_SAMPLERS` in `config.py`
2. Add a branch in `_build_optuna_sampler()` in `runner.py`
3. Add a test in `tests/test_sweep.py`