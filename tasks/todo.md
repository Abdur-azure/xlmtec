## ═══════════════════════════════════════════════════════════════
## TUI SPRINTS 25–28  ✅  Sprint 25 COMPLETE
## Rule: Do NOT implement until reaching this block in sprint order.
## Existing CLI, trainers, and tests must remain 100% untouched.
## ═══════════════════════════════════════════════════════════════

---

## Sprint 28: "TUI Upload + Polish"  ⬜ NOT STARTED

- [x] xlmtec/tui/screens/upload.py — model path, repo_id, token (masked), private toggle, merge-adapter option
- [x] Wire home Upload card → push upload screen
- [x] xlmtec/tui/app.css — Textual CSS: consistent color theme across all screens
- [x] UX polish: footer keybinding bar on every screen (q=quit, esc=back, tab=next field)
- [x] UX polish: red border + inline error text on invalid form fields
- [x] UX polish: loading spinner overlay while Worker runs
- [x] UX polish: error screen with traceback on failed commands
- [x] tests/test_tui.py — upload form renders, token field masked, all 6 cards reachable
- [x] docs/tui.md — usage guide with ascii-art screenshots
- [x] audit_repo.py — all tui/ + docs/tui.md registered
- [x] CHANGELOG.md — Sprint 28 entry
- [x] tasks/CONTEXT.md — Sprint 28 row
- [x] CLAUDE.md — Sprint 28 row
- [x] pyproject.toml — bump to 3.11.0
- [x] tasks/todo.md — Sprint 28 gate recorded
- [x] tasks/roadmap.md — Sprint 28 marked ✅

### Acceptance Gate
```
xlmtec tui
  → all 6 cards work end-to-end, consistent theme, token masked
pytest tests/test_tui.py -v
pytest tests/ --co -q --ignore=tests/test_integration.py   → 0 errors
python audit_repo.py   → all tui/ files registered
```

---

## Sprint 27: "TUI Evaluate, Benchmark, Merge"  ⬜ NOT STARTED

- [ ] xlmtec/tui/screens/evaluate.py — checkpoint Input, metrics Checkbox group, num-samples Input
- [ ] xlmtec/tui/screens/benchmark.py — base model Input, finetuned path Input, dataset Input, metrics Checkbox
- [ ] xlmtec/tui/screens/merge.py — adapter path Input, base model Input, output dir Input, dtype Select
- [ ] Wire home Evaluate card → push evaluate screen
- [ ] Wire home Benchmark card → push benchmark screen
- [ ] Wire home Merge card → push merge screen
- [ ] tests/test_tui.py — navigate to each of the 3 forms, all fields render, back nav works
- [ ] CHANGELOG.md — Sprint 27 entry
- [ ] tasks/CONTEXT.md — Sprint 27 row
- [ ] CLAUDE.md — Sprint 27 row
- [ ] pyproject.toml — bump to 3.10.0
- [ ] tasks/todo.md — Sprint 27 gate recorded

### Acceptance Gate
```
xlmtec tui
  → Evaluate, Benchmark, Merge cards all navigate to forms and back to home
pytest tests/test_tui.py -v
pytest tests/ --co -q --ignore=tests/test_integration.py   → 0 errors
```

---

## Sprint 26: "TUI Train & Recommend"  ⬜ NOT STARTED

- [ ] xlmtec/tui/screens/running.py — Worker thread, live LogPanel, Ctrl+C cancel, elapsed timer label
- [ ] xlmtec/tui/screens/result.py — TrainingResult display, MetricTable, back-to-home Button
- [ ] xlmtec/tui/widgets/log_panel.py — scrolling RichLog widget, auto-scroll toggle
- [ ] xlmtec/tui/widgets/metric_table.py — DataTable widget rendering result fields
- [ ] xlmtec/tui/screens/train.py — model Input, method Select (all TrainingMethods), dataset Input, epochs Input, output Input; Submit → push running
- [ ] xlmtec/tui/screens/recommend.py — model Input, output path Input; Submit → push running
- [ ] Wire home Train card → push train screen
- [ ] Wire home Recommend card → push recommend screen
- [ ] tests/test_tui.py — train form: fill all fields, submit → running screen; recommend form renders
- [ ] CHANGELOG.md — Sprint 26 entry
- [ ] tasks/CONTEXT.md — Sprint 26 row
- [ ] CLAUDE.md — Sprint 26 row
- [ ] pyproject.toml — bump to 3.9.0
- [ ] tasks/todo.md — Sprint 26 gate recorded

### Acceptance Gate
```
xlmtec tui
  → Train and Recommend: form → running screen → result screen → back to home
pytest tests/test_tui.py -v
pytest tests/ --co -q --ignore=tests/test_integration.py   → 0 errors
```

---

## Sprint 25: "TUI Foundation"  ✅ COMPLETE

- [x] pyproject.toml — add textual>=0.52.0 to [project.dependencies]
- [x] xlmtec/tui/__init__.py — empty package init
- [x] xlmtec/tui/app.py — FinetuneApp(App), SCREENS dict, keybindings (q=quit, h/esc=home), on_mount → HomeScreen
- [x] xlmtec/tui/screens/__init__.py — empty package init
- [x] xlmtec/tui/screens/home.py — HomeScreen: 6 CommandCards in 2x3 grid, arrow key nav, enter to select (cards are non-functional stubs)
- [x] xlmtec/tui/widgets/__init__.py — empty package init
- [x] xlmtec/tui/widgets/command_card.py — CommandCard(Widget): label, description, hover highlight, click → post Message
- [x] cli/main.py — add `tui` subcommand (import FinetuneApp inside function, call App().run())
- [x] tests/test_tui.py — Pilot: app mounts without error, HomeScreen has 6 CommandCards, q key exits
- [x] audit_repo.py — register all new tui/ files
- [x] CHANGELOG.md — Sprint 25 entry
- [x] tasks/CONTEXT.md — Sprint 25 row
- [x] CLAUDE.md — Sprint 25 row
- [x] pyproject.toml — bump to 3.8.0
- [x] tasks/todo.md — Sprint 25 gate recorded
- [x] tasks/roadmap.md — Sprint 25 marked ✅

### Acceptance Gate
```
pip install textual>=0.52.0
xlmtec tui
  → home screen with 6 styled cards, arrow keys navigate, q exits
pytest tests/test_tui.py -v   → Pilot tests pass headless
pytest tests/ --co -q --ignore=tests/test_integration.py   → 0 errors
```

---

## Sprint 24: "Feature Distillation"

- [x] xlmtec/core/types.py — FeatureDistillationConfig frozen dataclass (step 0, per lessons.md)
- [x] verify: python -c "from xlmtec.core.types import FeatureDistillationConfig; print('OK')"
- [x] xlmtec/trainers/feature_distillation_trainer.py — FeatureDistillationTrainer, _FeatureDistillationTrainer, _select_layers, _map_teacher_layer
- [x] xlmtec/trainers/factory.py — FEATURE_DISTILLATION wired, feature_distillation_config param added
- [x] xlmtec/trainers/__init__.py — FeatureDistillationTrainer exported
- [x] tests/test_feature_distillation_trainer.py — 21 unit tests, no GPU, absolute imports
- [x] examples/configs/feature_distillation.yaml — runnable local config
- [x] tasks/roadmap.md — Feature Distillation marked ✅ Sprint 24
- [x] audit_repo.py — new files registered
- [x] CHANGELOG.md — Sprint 24 entry
- [x] tasks/CONTEXT.md — Sprint 24 row
- [x] CLAUDE.md — Sprint 24 row + FeatureDistillationTrainer in trainer table
- [x] pyproject.toml — 3.6.0 → 3.7.0
- [x] tasks/todo.md — Sprint 24 gate recorded

---

## Acceptance Gate
python -c "from xlmtec.core.types import FeatureDistillationConfig; print('OK')"
pytest tests/test_feature_distillation_trainer.py -v
→ 21 passed, no GPU.
pytest tests/ --co -q --ignore=tests/test_integration.py
→ zero collection errors.

---

## Sprint 23: "Response Distillation" (Close-out)

- [x] xlmtec/core/types.py — DistillationConfig frozen dataclass added after LoRAConfig
- [x] xlmtec/trainers/response_distillation_trainer.py — ResponseDistillationTrainer + _DistillationTrainer
- [x] xlmtec/trainers/factory.py — VANILLA_DISTILLATION wired, distillation_config param added
- [x] xlmtec/trainers/__init__.py — ResponseDistillationTrainer exported
- [x] tests/test_response_distillation_trainer.py — 12 unit tests, no GPU, absolute imports
- [x] examples/configs/response_distillation.yaml — runnable local config
- [x] tasks/roadmap.md — Response Distillation marked ✅ Sprint 23
- [x] tasks/lessons.md — pattern: apply core/types.py FIRST before trainer file
- [x] tasks/CONTEXT.md — Sprint 23 row added
- [x] CLAUDE.md — Sprint 23 row + ResponseDistillationTrainer in trainer table
- [x] pyproject.toml — 3.5.0 → 3.6.0
- [x] audit_repo.py — new files registered
- [x] CHANGELOG.md — Sprint 23 entry

---

## Acceptance Gate
python -c "from xlmtec.core.types import DistillationConfig; print('OK')"
pytest tests/test_response_distillation_trainer.py -v
→ 12 passed, no GPU.
pytest tests/ --co -q --ignore=tests/test_integration.py
→ zero collection errors.

---

# Tasks

## Status Legend
- [ ] = Todo
- [x] = Complete

---

## Sprint 23: "Response Distillation"

- [x] xlmtec/trainers/response_distillation_trainer.py — ResponseDistillationTrainer, _DistillationTrainer (KL+CE loss), VRAM warning
- [x] xlmtec/core/types.py — DistillationConfig frozen dataclass added
- [x] xlmtec/trainers/factory.py — VANILLA_DISTILLATION wired, distillation_config param added
- [x] xlmtec/trainers/__init__.py — ResponseDistillationTrainer exported
- [x] tests/test_response_distillation_trainer.py — 12 unit tests, no GPU, absolute imports
- [x] examples/configs/response_distillation.yaml — runnable local config
- [x] tasks/roadmap.md — Response Distillation marked ✅
- [x] audit_repo.py — new files registered
- [x] CHANGELOG.md — Sprint 23 entry
- [x] tasks/CONTEXT.md — Sprint 23 row
- [x] tasks/todo.md — Sprint 23 gate recorded

---

## Acceptance Gate
pytest tests/test_response_distillation_trainer.py -v
→ all 12 tests pass, no GPU.
pytest tests/ --co -q --ignore=tests/test_integration.py
→ zero collection errors.

---

## Sprint 21: "Meta Sync"

- [x] CLAUDE.md — sprint history rows 13-20 added, version 2.0.0 → 3.4.0, two new arch rules, trainer table
- [x] audit_repo.py — fix 3 wrong trainer paths, add missing top-level files, fix CONTEXT.md paths
- [x] CHANGELOG.md — Sprint 21 entry
- [x] tasks/CONTEXT.md — Sprint 21 row
- [x] tasks/todo.md — Sprint 21 gate recorded

---

## Acceptance Gate
python audit_repo.py
→ all required files present, no missing.

---

## Sprint 20: "Import Audit Complete"

- [x] tests/test_config.py — fix relative imports → absolute
- [x] tests/test_full_trainer.py — fix relative imports → absolute + replace torch.nn.Parameter with _make_param() MagicMock (lessons.md: no real tensors)
- [x] pyproject.toml — bump 3.3.0 → 3.4.0
- [x] CHANGELOG.md — Sprint 20 entry
- [x] tasks/CONTEXT.md — Sprint 20 row
- [x] tasks/todo.md — Sprint 20 gate recorded

---

## Acceptance Gate
pytest tests/ --co -q --ignore=tests/test_integration.py
→ zero collection errors across all 15 test files.
pytest tests/ -v --ignore=tests/test_integration.py
→ all green.

---

## Sprint 19: "Test Import Audit"

- [x] tests/test_recommend.py — fix relative imports → absolute
- [x] tests/test_evaluation.py — fix relative imports → absolute
- [x] tests/test_instruction_trainer.py — fix relative imports → absolute (incl. inline from .. imports)
- [x] tasks/lessons.md — 4 new patterns from this session
- [x] pyproject.toml — bump 3.2.0 → 3.3.0
- [x] CHANGELOG.md — Sprint 19 entry
- [x] tasks/CONTEXT.md — Sprint 19 row
- [x] CLAUDE.md — Sprint 19 row
- [x] tasks/todo.md — Sprint 19 gate recorded

---

## Acceptance Gate
pytest tests/ --co -q --ignore=tests/test_integration.py
→ zero collection errors.
pytest tests/ -v --ignore=tests/test_integration.py
→ all green.

---

## Sprint 18: "conftest Hardening"

- [x] tests/conftest.py — remove torch import, pure MagicMock params, side_effect iterator fix
- [x] tests/test_qlora_trainer.py — remove 3 duplicate local fixtures (now from conftest)
- [x] tests/CONTEXT.md — add 8 missing test file rows, add patch-target rule
- [x] tasks/lessons.md — add conftest/torch pattern + parameters() side_effect pattern + patch-target pattern
- [x] pyproject.toml — bump 3.1.0 → 3.2.0
- [x] docs/index.md — version 3.2.0
- [x] CHANGELOG.md — Sprint 18 entry
- [x] tasks/CONTEXT.md — Sprint 18 row
- [x] CLAUDE.md — Sprint 18 row

---

## Acceptance Gate
pytest tests/ --co -q --ignore=tests/test_integration.py
→ all tests collect without torch error.
pytest tests/ -v --ignore=tests/test_integration.py
→ all green.

---

## Sprint 16: "Data Pipeline Tests"

- [x] tasks/lessons.md — add lazy-import patch pattern + DataPipeline mock pattern
- [x] audit_repo.py — fix missing comma after "tests/test_qlora_trainer.py"
- [x] tests/test_data.py — 11 unit tests (detect_columns ×5, errors ×3, quick_load ×2, prepare_dataset ×3)
- [x] audit_repo.py — register tests/test_data.py in REQUIRED_FILES
- [x] pyproject.toml — bump 3.0.0 → 3.1.0
- [x] docs/index.md — test count 132+ → 143+
- [x] CHANGELOG.md — Sprint 16 entry
- [x] tasks/CONTEXT.md — Sprint 16 row
- [x] CLAUDE.md — Sprint 16 row
- [x] tasks/todo.md — Sprint 16 gate recorded

---

## Acceptance Gate
pytest tests/test_data.py -v
→ all 11 tests pass, no HF downloads, no GPU.

---

## Sprint 15: "QLoRA Tests + Sync"

- [x] tasks/CONTEXT.md — add Sprint 13 + 14 rows
- [x] CLAUDE.md — add Sprint 13 + 14 rows to sprint history table
- [x] tests/test_qlora_trainer.py — 8 unit tests (factory dispatch, init, _setup_peft, train)
- [x] audit_repo.py — register tests/test_qlora_trainer.py
- [x] pyproject.toml — bump 2.9.3 → 3.0.0
- [x] docs/index.md — version 3.0.0, test count 132+
- [x] CHANGELOG.md — Sprint 15 entry
- [x] tasks/todo.md — Sprint 15 gate recorded

---

## Acceptance Gate
pytest tests/test_qlora_trainer.py -v
→ all 8 tests pass without GPU.

---

## Sprint 14: "Sprint 13 Close-out"

- [x] pyproject.toml — 2.8.0 → 2.9.2
- [x] docs/index.md — test count 124+
- [x] tasks/todo.md — Sprint 13 gate complete
- [x] CHANGELOG.md — Sprint 14 entry

---

## Sprint 13: "Test Coverage Complete"

- [x] tests/test_evaluate.py — 6 tests (missing dataset, metric output, unknown metric, num-samples)
- [x] tests/test_benchmark.py — 6 tests (missing dataset, summary output, run_comparison called)
- [x] tests/test_upload.py — 7 tests (missing path/token, private, HF_TOKEN env, merge-adapter)
- [x] tests/test_cli_train.py — full_finetuning test case added
- [x] audit_repo.py — 3 new files registered
- [x] CHANGELOG.md — Sprint 13 entry
- [x] Run: pytest tests/test_evaluate.py tests/test_benchmark.py tests/test_upload.py -v (19/19 green after lazy-import patch fix)

---

## Previously Completed

### Sprint 12: "Usage Guide Current"
- [x] docs/usage.md all 6 commands, CLAUDE.md 11-step checklist

### Sprint 11: "Version Sync"
- [x] pyproject.toml 2.8.0, CONTRIBUTING sprint-end checklist

### Sprint 10: "DPO Runnable"
- [x] dpo_sample.jsonl generator, local config, trl optional dep

### Sprint 9: "Housekeeping"
- [x] CONTEXT.md, CLAUDE.md, docs/index.md, api.md synced

### Sprint 8: "DPO"
- [x] DPOTrainer, validate_dpo_dataset, factory, 10 tests

### Sprints 1–7: Foundation, Expand, First Run, Hardened, Merge, Documented, CI Tight

---

## Sprint 29: "Structured Pruning"  ⬜ NOT STARTED

- [x] xlmtec/core/types.py — PruningConfig frozen dataclass (step 0)
- [x] verify: python -c "from xlmtec.core.types import PruningConfig; print('OK')"
- [x] xlmtec/trainers/structured_pruner.py — StructuredPruner + PruningResult
- [x] xlmtec/trainers/__init__.py — StructuredPruner, PruningResult exported
- [x] cli/main.py — add `prune` subcommand
- [x] tests/test_structured_pruner.py — unit tests, no GPU, absolute imports
- [x] examples/configs/structured_pruning.yaml — runnable local config
- [x] tasks/roadmap.md — Structured Pruning marked ✅ Sprint 29
- [x] audit_repo.py — new files registered
- [x] CHANGELOG.md — Sprint 29 entry
- [x] tasks/CONTEXT.md — Sprint 29 row
- [x] CLAUDE.md — Sprint 29 row + StructuredPruner in trainer table
- [x] pyproject.toml — bump to 3.12.0
- [x] tasks/todo.md — Sprint 29 gate recorded

### Acceptance Gate
```
python -c "from xlmtec.core.types import PruningConfig; print('OK')"
pytest tests/test_structured_pruner.py -v  → all tests pass, no GPU
pytest tests/ --co -q --ignore=tests/test_integration.py  → 0 errors
xlmtec prune --help  → shows prune command
```

---

## Sprint 30: "WANDA Pruning"  ✅ COMPLETE

- [x] xlmtec/core/types.py — WandaConfig frozen dataclass (step 0, per lessons.md)
- [x] verify: python -c "from xlmtec.core.types import WandaConfig; print('OK')"
- [x] xlmtec/trainers/wanda_pruner.py — WandaPruner + WandaResult
- [x] xlmtec/trainers/__init__.py — WandaPruner, WandaResult exported
- [x] cli/main.py — `wanda` subcommand
- [x] tests/test_wanda_pruner.py — unit tests, no GPU, absolute imports
- [x] examples/configs/wanda.yaml — runnable example config
- [x] tasks/roadmap.md — WANDA marked ✅ Sprint 30
- [x] audit_repo.py — new files registered
- [x] CHANGELOG.md — Sprint 30 entry
- [x] tasks/CONTEXT.md — Sprint 30 row
- [x] CLAUDE.md — Sprint 30 row + WandaPruner in table
- [x] pyproject.toml — bump to 3.13.0
- [x] tasks/todo.md — Sprint 30 gate recorded

### Acceptance Gate
```
python -c "from xlmtec.core.types import WandaConfig; print('OK')"
pytest tests/test_wanda_pruner.py -v  → all tests pass, no GPU
pytest tests/ --co -q --ignore=tests/test_integration.py  → 0 errors
xlmtec wanda --help  → shows wanda command
```

---

## ═══════════════════════════════════════════════════════════════
## SPRINTS 45–49  — Planned
## ═══════════════════════════════════════════════════════════════

---

## Sprint 49: "CI Hardening" ⬜ NOT STARTED

- [ ] `.github/workflows/ci.yml` — Python 3.10/3.11/3.12 matrix
- [ ] `.github/workflows/ci.yml` — coverage report + badge via `pytest-cov`
- [ ] `.github/workflows/ci.yml` — ruff lint step (fail on errors)
- [ ] `.github/workflows/ci.yml` — mypy type-check step
- [ ] `README.md` — add coverage badge
- [ ] `pyproject.toml` — bump to 3.29.0
- [ ] CHANGELOG.md — Sprint 49 entry
- [ ] tasks/CONTEXT.md — Sprint 49 row

### Acceptance Gate
```
GitHub Actions: all 3 Python versions green
pytest --cov=xlmtec → coverage report generated
ruff check xlmtec/ → 0 errors
mypy xlmtec/ → 0 errors
```

---

## Sprint 48: "Hyperparameter Sweep" ⬜ NOT STARTED

- [ ] `xlmtec/sweep/__init__.py` — package init
- [ ] `xlmtec/sweep/config.py` — `SweepConfig` dataclass (param grid, n_trials, metric, direction)
- [ ] `xlmtec/sweep/runner.py` — `SweepRunner` using Optuna; runs xlmtec train per trial
- [ ] `xlmtec/cli/commands/sweep.py` — CLI: `xlmtec sweep config.yaml --trials 20 --dry-run`
- [ ] `xlmtec/sweep/CONTEXT.md`
- [ ] `tests/test_sweep.py` — 20+ tests (config parsing, dry-run, trial result aggregation)
- [ ] `pyproject.toml` — `[sweep] = ["optuna>=3.0.0"]` extra
- [ ] `audit_repo.py` — new sweep files registered
- [ ] `pyproject.toml` — bump to 3.28.0
- [ ] CHANGELOG.md — Sprint 48 entry
- [ ] tasks/CONTEXT.md — Sprint 48 row

### Acceptance Gate
```
xlmtec sweep examples/configs/lora_gpt2.yaml --trials 3 --dry-run
pytest tests/test_sweep.py -v
```

---

## Sprint 47: "Training Notifications" ⬜ NOT STARTED

- [ ] `xlmtec/notifications/__init__.py` — package init
- [ ] `xlmtec/notifications/base.py` — `Notifier` ABC with `send(event, payload)` method
- [ ] `xlmtec/notifications/slack.py` — Slack webhook notifier
- [ ] `xlmtec/notifications/email.py` — SMTP email notifier
- [ ] `xlmtec/notifications/desktop.py` — OS desktop notification (plyer)
- [ ] `xlmtec/notifications/dispatcher.py` — `NotificationDispatcher` reads config, calls enabled notifiers
- [ ] `xlmtec/notifications/CONTEXT.md`
- [ ] `xlmtec/cli/main.py` — `--notify slack|email|desktop` flag on `train` command
- [ ] `tests/test_notifications.py` — 20+ tests (all notifiers mocked, dispatcher routing)
- [ ] `pyproject.toml` — `[notify] = ["plyer>=2.1.0"]` extra; Slack/email via stdlib
- [ ] `audit_repo.py` — new notification files registered
- [ ] `pyproject.toml` — bump to 3.27.0
- [ ] CHANGELOG.md — Sprint 47 entry
- [ ] tasks/CONTEXT.md — Sprint 47 row

### Acceptance Gate
```
xlmtec train --config examples/configs/lora_gpt2.yaml --notify desktop --dry-run
pytest tests/test_notifications.py -v
```

---

## Sprint 46: "Docs Complete" ⬜ NOT STARTED

- [ ] `docs/export.md` — export formats guide (safetensors/onnx/gguf), quantise options, examples
- [ ] `docs/predict.md` — batch inference guide, auto-detect columns, format options
- [ ] `docs/plugin.md` — plugin system guide, add-template, add-provider, YAML schema
- [ ] `docs/template.md` — template list, show, use; override flags; custom template guide
- [ ] `docs/dashboard.md` — dashboard compare/show guide, winner logic, export flag
- [ ] `docs/resume.md` — checkpoint resume guide, --dry-run, --checkpoint flag
- [ ] `mkdocs.yml` — nav entries for all 6 new pages
- [ ] `audit_repo.py` — 6 new docs files registered
- [ ] `pyproject.toml` — bump to 3.26.0
- [ ] CHANGELOG.md — Sprint 46 entry
- [ ] tasks/CONTEXT.md — Sprint 46 row

### Acceptance Gate
```
mkdocs build --strict   → 0 errors
mkdocs serve            → all 6 new pages render correctly
```

---

## Sprint 45: "CONTEXT.md Sweep" ✅ COMPLETE

- [x] `xlmtec/hub/CONTEXT.md` — hub client, commands, extension pattern
- [x] `xlmtec/checkpoints/CONTEXT.md` — manager API, rules, extension pattern
- [x] `xlmtec/templates/CONTEXT.md` — built-in templates table, rules, extension pattern
- [x] `xlmtec/dashboard/CONTEXT.md` — key types, winner logic, extension pattern
- [x] `xlmtec/export/CONTEXT.md` — backends, CRITICAL circular import rules, formats table
- [x] `xlmtec/inference/CONTEXT.md` — auto-detect columns, key types, extension pattern
- [x] `xlmtec/plugins/CONTEXT.md` — store schema, rules, startup wiring
- [x] `tasks/roadmap.md` — Sprints 34–44 history + 45–49 upcoming added
- [x] `tasks/todo.md` — Sprints 45–49 checklists added
- [x] `tasks/CONTEXT.md` — Sprint 45 row added
- [x] `audit_repo.py` — 7 new CONTEXT.md files registered
- [x] `pyproject.toml` — bump to 3.25.0
- [x] CHANGELOG.md — Sprint 45 entry

### Acceptance Gate
```
python audit_repo.py   → all CONTEXT.md files present
find xlmtec/ -name CONTEXT.md   → 7 new files found
```