# xlmtec/dashboard — Context

Training run reader and comparator powering `xlmtec dashboard` commands.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `reader.py` | `RunReader` — parses `trainer_state.json` + `config.yaml` from a run dir |
| `comparator.py` | `RunComparator` — side-by-side metric comparison, winner selection, config diff |
| `CONTEXT.md` | This file |

## Commands

```bash
xlmtec dashboard compare output/run1 output/run2 output/run3
xlmtec dashboard compare output/run1 output/run2 --export results.json
xlmtec dashboard show output/run1
```

## Key Types

| Type | Fields |
|------|--------|
| `RunSummary` | `name`, `total_steps`, `total_epochs`, `best_metric`, `best_eval_loss`, `final_train_loss`, `config` |
| `ComparisonResult` | `runs: list[RunSummary]`, `winner: RunSummary \| None`, `config_diff: dict` |

## Winner Selection Priority

1. `best_metric` (higher is better)
2. `best_eval_loss` (lower is better)
3. `final_train_loss` (lower is better)
4. Fallback: most steps completed

## Rules

- `RunReader` is filesystem-only — never loads a model.
- Missing fields in `trainer_state.json` are silently set to `None` — never raise on partial data.
- `config_diff` shows only keys that differ across runs (keys present in some but not all runs are included).
- Tests use `generate_dummy_runs.py` helper — no real training required.

## Extension pattern

To add a new metric column to the comparison table:
1. Add the field to `RunSummary`
2. Extract it in `RunReader.read()`
3. Add a `_row()` call in `cli/commands/dashboard.py`
4. Add a test in `tests/test_dashboard.py`