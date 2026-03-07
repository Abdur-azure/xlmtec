# cli/ — Context

Typer-based CLI. All user-facing subcommands live in `main.py`.

## Commands

| Command | What it does |
|---------|-------------|
| `train` | Loads config (file or flags), loads model, prepares dataset, runs TrainerFactory |
| `evaluate` | Scores a saved checkpoint with BenchmarkRunner |
| `benchmark` | Before/after comparison: base model vs fine-tuned |
| `upload` | Pushes adapter (or merged model) to HuggingFace Hub |
| `recommend` | Inspects model param count + VRAM, outputs a ready-to-use YAML config |
| `merge` | Merges LoRA adapter into base model, saves standalone model (no PEFT required) |

## Rules

- `cli/__init__.py` must be **empty or contain only `"""CLI package."""`** — any `setup()` call here crashes pytest collection.
- The `recommend` command writes valid `PipelineConfig` YAML — validate with `PipelineConfig.from_yaml()` in tests.
- All commands must handle `FineTuneError` and print a clean error message before `raise typer.Exit(code=1)`.
- Never import heavy deps (torch, transformers) at module level — import inside command functions to keep `--help` fast.

## Extension pattern

To add a new subcommand:
1. Add `@app.command()` decorated function in `main.py`
2. Keep imports inside the function body
3. Add the command to `docs/usage.md`