# Contributing to lmtool

This project uses sprint-based development with AI assistance. Read this before opening a PR.

---

## Getting started

```bash
git clone https://github.com/Abdur-azure/lmtool.git
cd lmtool
pip install -e .
python audit_repo.py        # verify all required files are present
pytest tests/ -v --ignore=tests/test_integration.py
```

---

## How to add a new trainer

Follow this checklist exactly — it matches `CLAUDE.md`:

1. Create `lmtool/trainers/<name>_trainer.py`, extend `BaseTrainer`
2. Implement `_setup_peft(model) -> model` — this is the only required override
3. Add the new `TrainingMethod` enum value to `core/types.py` if missing
4. Wire it in `trainers/factory.py` → `TrainerFactory.create()`
5. Export from `trainers/__init__.py`
6. Add to `_LORA_METHODS` in `cli/main.py` if it needs a LoRA config, or leave it out if not
7. Write `tests/test_<name>_trainer.py` — mock HF Trainer, no GPU required
8. Add a CLI test to `tests/test_cli_train.py` asserting `exit_code == 0`
9. Add an example config to `examples/configs/<name>.yaml`
10. Update `docs/api.md` and `docs/configuration.md`
11. Update `CHANGELOG.md` and `audit_repo.py`

---

## How to add a new CLI command

1. Add `@app.command()` decorated function in `cli/main.py`
2. Keep all heavy imports (`torch`, `transformers`) **inside** the function — keeps `--help` fast
3. Handle `FineTuneError` and exit with `raise typer.Exit(code=1)` on failure
4. Write `tests/test_<command>.py` with mocked stack — no real model loads
5. Add the command to `docs/usage.md`
6. Update `CHANGELOG.md` and `audit_repo.py`

---

## Sprint conventions

- Sprints are named (e.g. "Hardened", "First Run") — add the name to `tasks/todo.md`
- Every sprint ends with an acceptance gate: a specific `pytest` command that must be green
- After any correction: update `tasks/lessons.md` with the pattern that caused it
- After every sprint: update `CHANGELOG.md` and `audit_repo.py`

---

## Rules from lessons.md (must-read)

These have been learned the hard way:

- **Config objects are frozen** — never mutate after construction
- **All errors extend `FineTuneError`** — no raw Python builtins from module code
- **Use `get_logger(__name__)`** from `utils/logging.py` — never `logging.getLogger` directly
- **`data/__init__.py` must mirror CLI imports exactly** — trace the full chain when adding exports
- **Never allocate real tensors in tests** — use `MagicMock` with `numel.return_value = N`
- **YAML paths use forward slashes** — `Path.as_posix()` in tests, never raw Windows strings
- **`cli/__init__.py` must stay empty** — any code there crashes pytest collection on Windows
- **Guard `lora_config` by method** — only attach it for `{lora, qlora, instruction_tuning}`

---

## Sprint-end checklist

Run this every time a sprint closes — no exceptions. These are the items that have drifted every sprint without it.

```
[ ] pyproject.toml          version = "X.Y.Z" bumped
[ ] tasks/CONTEXT.md        new sprint row added to table
[ ] CLAUDE.md               new sprint row added to history table
[ ] docs/index.md           version + test count updated
[ ] CHANGELOG.md            sprint entry added
[ ] audit_repo.py           any new files added to REQUIRED_FILES
[ ] tasks/todo.md           sprint archived, acceptance gate recorded
```

This checklist is enforced by `lessons.md` pattern: *"tasks/CONTEXT.md sprint table drifts — update it every sprint"* and *"docs/index.md is a second source of truth"*. If you skip any item, the next session starts with stale context.

---

## Before submitting a PR

```bash
python audit_repo.py                          # all required files present
pytest tests/ -v                              # all tests green
mkdocs build --strict                         # docs build clean
```

Update `CHANGELOG.md` with what changed.

---

## Repo orientation

Read `CLAUDE.md` at the root for architecture rules, key types, and sprint history.
Read the `CONTEXT.md` in whichever subpackage you're touching before making changes.