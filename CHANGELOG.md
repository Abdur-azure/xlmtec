# Changelog

## [3.29.0] — Sprint 49: "CI Hardening" — 2026-03-11

### Added
- `.github/workflows/ci.yml` — Python 3.10/3.11/3.12 test matrix; each version
  installs `.[dev]` and runs `pytest tests/ --ignore=test_integration.py --cov=xlmtec`.
- `.github/workflows/ci.yml` — dedicated `lint` job: `ruff check xlmtec/` +
  `ruff format --check xlmtec/` + `mypy xlmtec/`. Package installed before ruff
  runs to avoid false-positive import errors (per lessons.md).
- `.github/workflows/ci.yml` — `coverage` job posts coverage % as a PR comment
  via `orgoro/coverage`; thresholds: 70% overall, 80% new code.
- `.github/workflows/ci.yml` — `integration` job (push-only, 600 s timeout)
  installs `.[ml,tui,dev]` and runs `tests/test_integration.py`.
- `pyproject.toml` — `pytest-cov>=4.0.0` added to `[dev]` extra.
- `pyproject.toml` — `[tool.coverage.run]` and `[tool.coverage.report]` sections
  added; `fail_under = 60`.
- `README.md` — CI status badge and coverage badge added.

### Changed
- `pyproject.toml` — version `3.28.0` → `3.29.0`.
- `audit_repo.py` — four telemetry files registered (Sprint 49-A close-out).

---

## [3.28.1] — Sprint 49-A: "App Insights / Telemetry" — 2026-03-11

### Added
- `xlmtec/utils/telemetry.py` — `Session` (JSONL log file per invocation),
  `_AppLogger` singleton (`AppLogger`), `@track` decorator. Zero new dependencies
  (stdlib only). Logs live at `~/.xlmtec/logs/session_<ts>_<cmd>.jsonl`. Sensitive
  keys (api/token/secret/password) auto-redacted; paths reduced to basename.
- `xlmtec/utils/crash_report.py` — `CrashReporter.write()` generates a
  human-readable `.txt` crash file with version, OS, GPU, traceback, and last 10
  session events. `latest()` and `list_recent()` for discovery. Rotates to max 20
  crash files.
- `xlmtec/cli/commands/report.py` — `xlmtec report` command. Shows latest crash
  inline; `--last N` lists recent crashes as a table; `--sessions` lists session
  files; `--open` opens crash file in `$EDITOR`.
- `tests/test_telemetry.py` — 38 tests: `Session`, `_AppLogger`, `@track`,
  `CrashReporter`, and `run_report`. All use `tmp_path` — nothing written to
  `~/.xlmtec` during tests.

### Changed
- `xlmtec/core/exceptions.py` — `FineTuneError.__init__` now calls
  `AppLogger.log_error(self)` via lazy import. All subclasses inherit this
  automatically.
- `xlmtec/cli/main.py` — `@app.callback()` calls `AppLogger.start(cmd=...)`;
  `app.command("report")(report)` registered.

---

## [3.27.0] — Sprint 47: "Training Notifications" — 2026-03-09

### Added
- `xlmtec/notifications/__init__.py` — package init
- `xlmtec/notifications/base.py` — `Notifier` ABC, `NotifyEvent` enum, `NotifyPayload` dataclass
- `xlmtec/notifications/slack.py` — Slack webhook notifier (stdlib urllib, zero extra deps)
- `xlmtec/notifications/email.py` — SMTP email notifier (stdlib smtplib, zero extra deps)
- `xlmtec/notifications/desktop.py` — OS desktop notification via `plyer` (console fallback if not installed)
- `xlmtec/notifications/dispatcher.py` — `NotificationDispatcher.from_channels(["slack","desktop"])` factory
- `xlmtec/notifications/CONTEXT.md` — channel table, rules, extension pattern
- `tests/test_notifications.py` — 22 tests: payload titles, Slack send/failure/env, Email SMTP/starttls/env, Desktop fallback, dispatcher routing/unknown-channel/safe_send
- `pyproject.toml` — `[notify] = ["plyer>=2.1.0"]` extra added

### Changed
- `audit_repo.py` — 8 new notification files registered
- `pyproject.toml` — version 3.26.0 → 3.27.0

### CLI wiring (add to `main.py`)
```python
# In the train command signature:
notify: str = typer.Option(
    "", "--notify", "-n",
    help="Notification channels, comma-separated: slack,email,desktop"
)
# At end of successful training:
if notify:
    from xlmtec.notifications.dispatcher import NotificationDispatcher
    from xlmtec.notifications.base import NotifyEvent
    d = NotificationDispatcher.from_channels([c.strip() for c in notify.split(",")])
    d.notify(NotifyEvent.TRAINING_COMPLETE, run_name=str(output_dir), message="Training finished.")
```

---

## [3.26.0] — Sprint 46: "Docs Complete" — 2026-03-09

### Added
- `docs/resume.md` — checkpoint resume guide: dry-run, --checkpoint, --epochs, troubleshooting
- `docs/template.md` — template list/show/use guide, built-in templates table, override flags, custom template YAML schema
- `docs/dashboard.md` — compare/show guide, winner selection priority, config diff, export flag
- `docs/export.md` — safetensors/onnx/gguf guide, quantise options, extras install, GGUF workflow, merge-first advice
- `docs/predict.md` — batch inference guide, auto-detect columns, JSONL/CSV formats, dry-run, generate test data
- `docs/plugin.md` — custom template + provider walkthrough, storage schema, reserved names, list/remove
- `mkdocs.yml` — Commands nav section added with all 8 command pages

### Changed
- `audit_repo.py` — 6 new docs files + mkdocs.yml registered
- `pyproject.toml` — version 3.25.0 → 3.26.0

---

## [3.25.0] — Sprint 45: "CONTEXT.md Sweep" — 2026-03-09

### Added
- `xlmtec/hub/CONTEXT.md` — hub client, commands, rules, extension pattern
- `xlmtec/checkpoints/CONTEXT.md` — CheckpointManager API, rules, extension pattern
- `xlmtec/templates/CONTEXT.md` — built-in templates table, reserved names, extension pattern
- `xlmtec/dashboard/CONTEXT.md` — key types, winner selection logic, extension pattern
- `xlmtec/export/CONTEXT.md` — backends, CRITICAL circular import rules, formats + quantise table
- `xlmtec/inference/CONTEXT.md` — auto-detect column priority, key types, extension pattern
- `xlmtec/plugins/CONTEXT.md` — store schema, startup wiring, reserved names rules

### Changed
- `tasks/roadmap.md` — Sprints 34–44 history table + Sprints 45–49 upcoming table added
- `tasks/todo.md` — Sprint 45–49 checklists added
- `audit_repo.py` — 7 new CONTEXT.md paths registered
- `pyproject.toml` — version 3.24.0 → 3.25.0

---

## [3.24.0] — Sprint 44: "CHANGELOG + Audit" — 2026-03-09

### Fixed (catch-up)
- `CHANGELOG.md` — Sprints 34–43 entries added
- `audit_repo.py` — all new modules from Sprints 34–43 registered
- `tasks/CONTEXT.md` — sprint rows 34–43 added
- `pyproject.toml` — version 3.23.0 → 3.24.0

---

## [3.23.0] — Sprint 43: "Plugin System" — 2026-03-09

### Added
- `xlmtec/plugins/__init__.py` — package init
- `xlmtec/plugins/store.py` — `PluginStore`, `register_template`, `register_provider`, `remove_plugin`; persists to `~/.xlmtec/plugins.json`
- `xlmtec/plugins/loader.py` — `PluginLoader` injects custom templates/providers into live registries at startup
- `xlmtec/cli/commands/plugin.py` — CLI: `add-template`, `add-provider`, `list`, `remove`
- `tests/test_plugins.py` — 21 tests: store I/O, register/remove, loader injection, error handling

---

## [3.22.0] — Sprint 42: "Batch Inference" — 2026-03-09

### Added
- `xlmtec/inference/__init__.py` — package init
- `xlmtec/inference/data_loader.py` — `DataLoader`: reads JSONL/CSV, auto-detects text column
- `xlmtec/inference/writer.py` — `PredictionWriter`: writes predictions to JSONL or CSV
- `xlmtec/inference/predictor.py` — `BatchPredictor`: batched text generation over a dataset
- `xlmtec/cli/commands/predict.py` — CLI: `xlmtec predict` with `--dry-run`, `--batch-size`, `--max-new-tokens`
- `generate_inference_data.py` — zero-dep helper to create dummy model dir + test data
- `tests/test_inference.py` — 24 tests: DataLoader (JSONL/CSV, auto-detect, errors), PredictionWriter (JSONL/CSV, dirs), CLI logic

---

## [3.21.0] — Sprint 41: "Export Formats" — 2026-03-09

### Added
- `xlmtec/export/__init__.py` — package init
- `xlmtec/export/formats.py` — `ExportFormat` enum + `FormatMeta` per-format metadata
- `xlmtec/export/exporter.py` — `ModelExporter` dispatches to format backends
- `xlmtec/export/backends/__init__.py` — backends package
- `xlmtec/export/backends/safetensors.py` — export via `safetensors`
- `xlmtec/export/backends/onnx.py` — export via `optimum` (lazy import after dry-run check)
- `xlmtec/export/backends/gguf.py` — export via llama.cpp convert script
- `xlmtec/cli/commands/export.py` — CLI: `xlmtec export --format safetensors|onnx|gguf --dry-run`
- `tests/test_export.py` — 30 tests: format enum, per-backend dry-run, exporter dispatch, CLI logic
- `pyproject.toml` — `[onnx]` and `[gguf]` extras added; `[full]` bundle updated

---

## [3.20.0] — Sprint 40: "Evaluation Dashboard" — 2026-03-09

### Added
- `xlmtec/dashboard/__init__.py` — package init
- `xlmtec/dashboard/reader.py` — `RunReader`: parses `trainer_state.json`, config, eval results
- `xlmtec/dashboard/comparator.py` — `RunComparator`: side-by-side metric comparison, winner selection, config diff
- `xlmtec/cli/commands/dashboard.py` — CLI: `xlmtec dashboard compare` (rich table + winner panel + config diff) and `xlmtec dashboard show`
- `generate_dummy_runs.py` — zero-dep helper creates 3 dummy runs for testing
- `tests/test_dashboard.py` — 22 tests: RunReader, RunComparator, diff_configs, ComparisonResult

---

## [3.19.0] — Sprint 39: "Config Templates" — 2026-03-09

### Added
- `xlmtec/templates/__init__.py` — package init
- `xlmtec/templates/registry.py` — 7 built-in templates: `sentiment`, `classification`, `qa`, `summarisation`, `code`, `chat`, `dpo`; `get_template()`, `list_templates()`
- `xlmtec/cli/commands/template.py` — CLI: `xlmtec template list|show|use` with `--model`, `--data`, `--epochs` overrides
- `tests/test_templates.py` — 26 tests: list, get, as_dict overrides, YAML validity, per-template spot checks

### Also (Sprint 38: "Checkpoint Resume")
- `xlmtec/checkpoints/__init__.py` — package init
- `xlmtec/checkpoints/manager.py` — `CheckpointManager`: list, latest, get, summary
- `xlmtec/cli/commands/resume.py` — CLI: `xlmtec resume <output-dir> --dry-run --checkpoint --epochs`
- `tests/test_checkpoints.py` — 22 tests: list/latest/get/summary, resume_training dry-run logic

---

## [3.18.0] — Sprint 37: "Docs Overhaul" — 2026-03-09

### Changed
- `README.md` — full rewrite: xlmtec branding, PyPI badges, all commands table, AI providers, quickstart
- `docs/installation.md` — Python 3.10+, extras-based install, GPU setup, HF token
- `docs/usage.md` — all commands: hub, ai-suggest, config validate, --dry-run, template, resume, predict, export, dashboard
- `docs/ai_integrations.md` — NEW: ai-suggest guide, all 3 providers, output format, how it works
- `docs/hub.md` — NEW: hub search/info/trending guide, workflow example

---

## [3.17.0] — Sprint 34–36: "AI Integrations + CLI UX + Model Hub" — 2026-03-08

### Added (Sprint 34 — AI Integrations)
- `xlmtec/integrations/__init__.py` — `get_provider(name, api_key)` factory
- `xlmtec/integrations/base.py` — `AIIntegration` ABC, `SuggestResult` dataclass
- `xlmtec/integrations/claude.py` — Anthropic Claude provider
- `xlmtec/integrations/gemini.py` — Google Gemini provider (`google-genai>=0.8.0`)
- `xlmtec/integrations/codex.py` — OpenAI GPT provider
- `xlmtec/integrations/prompt_builder.py` — shared system + user prompt templates
- `xlmtec/integrations/response_parser.py` — parses raw JSON → `SuggestResult`
- `xlmtec/cli/commands/ai_suggest.py` — CLI: `xlmtec ai-suggest "<task>" --provider`
- `tests/test_integrations.py` — 24 tests, zero real API calls
- `pyproject.toml` — `[claude]`, `[gemini]`, `[codex]`, `[ai]` extras

### Added (Sprint 35 — CLI UX Polish)
- `xlmtec/cli/ux.py` — `get_version`, `print_error`, `print_success`, `task_progress`, `make_training_progress`
- `xlmtec/cli/commands/dry_run.py` — `execute_dry_run(path)` pure function
- `xlmtec/cli/commands/config_validate.py` — `validate_config(path, strict)` + CLI wrapper
- `tests/test_ux.py` — 22 tests
- `xlmtec --version` / `-V` flag wired

### Added (Sprint 36 — Model Hub)
- `xlmtec/hub/__init__.py` — re-exports `HubClient`, `ModelSummary`, `ModelDetail`
- `xlmtec/hub/client.py` — `HubClient` wrapping `HfApi`
- `xlmtec/hub/formatter.py` — rich tables for search/trending/info
- `xlmtec/cli/commands/hub.py` — CLI: `xlmtec hub search|info|trending`
- `tests/test_hub.py` — 21 tests

---

## [3.16.0] — Sprint 33: "Post-Rename Cleanup" — 2026-03-07

### Changed
- Project renamed from `xlmtec` / `xlmtec` to `xlmtec`
- `pyproject.toml` — `name`, `scripts` entrypoint, URLs, `packages.find` all updated
- All Python source files — `from xlmtec.` imports → `from xlmtec.`
- All patch targets in tests — `xlmtec.` → `xlmtec.`
- CLI command — `xlmtec` → `xlmtec`
- `CLAUDE.md`, `CHANGELOG.md`, `README.md`, `docs/` — project name updated
- Version 3.15.0 → 3.16.0

---


All notable changes to this project are documented here.

---

## [3.15.0] — Sprint 32: "Stabilise" — 2026-03-07

### Fixed
- `pyproject.toml` — split mandatory dependencies: `torch`, `transformers`, `peft`,
  `accelerate`, `bitsandbytes`, `datasets`, `textual` moved to optional extras.
  New extras: `[ml]` (GPU stack), `[tui]` (Textual), `[dpo]` (trl), `[dev]`
  (test/lint tools), `[full]` (everything). Core package now installs in seconds
  with no GPU libraries. Version `3.14.0` → `3.15.0`.
- `.github/workflows/ci.yml` — fixed 5 issues: (1) install now uses
  `pip install -e ".[ml,tui,dev]"` instead of legacy `requirements.txt`;
  (2) cache key hashes `pyproject.toml` not `requirements.txt`; (3) integration
  test timeout raised `120s` → `600s` (cold runner GPT-2 download); (4) `textual`
  and `pytest-asyncio` now installed via extras (TUI tests were silently failing);
  (5) lint job installs package first so ruff can resolve all imports.
- `audit_repo.py` — fixed `finetune_xlmtec/...` doubled-prefix typo on TUI
  widget entries; added `docs/` entries; cleaned up and sorted all sections.
- `CLAUDE.md` — fixed all `cli/main.py` bare path refs →
  `xlmtec/cli/main.py`; added architecture rule 11 (heavy deps optional);
  added `[ml,tui,dev]` install note; pruner table added; Sprint 30–32 history rows.
- `docs/installation.md` — corrected Python requirement `3.8+` → `3.10+`;
  replaced `pip install -r requirements.txt` with `pip install -e ".[full]"`;
  added extras table; removed all `requirements.txt` references.

### Removed
- `requirements.txt` — deleted. `pyproject.toml` is the single source of truth.
  `docs/installation.md` updated accordingly.
---


## [3.14.0] — Sprint 31: "Integration Hardening" — 2026-03-05

### Added
- `tests/test_integration.py` — 15 new integration tests across 5 new test classes:
  - `TestResponseDistillationIntegration` (2 tests): gpt2→gpt2 distillation on CPU,
    asserts student saved and TrainingResult fields typed correctly.
  - `TestFeatureDistillationIntegration` (2 tests): feature distillation with auto
    layer selection and explicit `feature_layers=[0,5,11]`.
  - `TestStructuredPrunerIntegration` (3 tests): heads pruning, PruningResult field
    types, FFN method on real GPT-2.
  - `TestWandaPrunerIntegration` (4 tests): magnitude-only, WandaResult fields,
    calibration data (random calib_ids), global mode.
  - `TestCLISmoke` (4 tests): CliRunner-level smoke for train/prune/wanda/distillation
    CLI commands — asserts exit codes with mocked model stack.
  Total integration tests: 21 (was 6, +15).

### Fixed
- `.github/workflows/ci.yml` — install now uses `pip install -e ".[dev]"` from
  `pyproject.toml` instead of legacy `requirements.txt`. Cache key updated to hash
  `pyproject.toml`. Integration test timeout raised to 300s (was 120s). Removed
  stray `requirements.txt` reference.

### Changed
- `pyproject.toml` — version 3.13.0 → 3.14.0

---


## [3.13.0] — Sprint 30: "WANDA Pruning" — 2026-03-05

### Added
- `xlmtec/core/types.py` — `WandaConfig` frozen dataclass
  (output_dir, sparsity=0.5, n_calibration_samples=128,
  calibration_seq_len=128, layer_types=None, use_row_wise=True).
- `xlmtec/trainers/wanda_pruner.py` — `WandaPruner` + `WandaResult`.
  WANDA unstructured pruning: registers forward hooks to collect input
  activation norms over a calibration dataset, scores each weight by
  |W_ij| * ||X_j||_2, zeros the bottom `sparsity` fraction per layer.
  Falls back to magnitude-only scoring when no calibration data is
  provided. Supports row-wise and global sparsity modes. Auto-targets
  nn.Linear and Conv1D layers. Returns `WandaResult` frozen dataclass.
- `xlmtec/trainers/__init__.py` — `WandaPruner`, `WandaResult` exported.
- `cli/main.py` — `wanda` subcommand. Args: model_path, --output, --sparsity
  (0.5), --n-samples (128), --seq-len (128), --dataset (optional),
  --row-wise/--global.
- `tests/test_wanda_pruner.py` — 18 unit tests: WandaConfig validation (4),
  helper functions `_wanda_score` and `_apply_wanda_mask` (5), pruner
  without calibration (7), pruner with real calibration data (2). Uses
  real CPU torch tensors — no GPU required.
- `tests/test_wanda_cli.py` — 6 CLI-level tests: missing path, invalid
  sparsity, happy path exits 0, sparsity forwarded to config, output
  contains sparsity, missing dataset exits 1.
- `examples/configs/wanda.yaml` — runnable example with sparsity guidance
  and WANDA vs Structured Pruning comparison table.
- `tasks/roadmap.md` — WANDA marked ✅ Sprint 30.

### Changed
- `pyproject.toml` — version 3.12.0 → 3.13.0
- `audit_repo.py` — new files registered.

---

## [3.12.0] — Sprint 29: "Structured Pruning" — 2026-03-04

### Added
- `xlmtec/core/types.py` — `PruningConfig` frozen dataclass
  (output_dir, sparsity=0.3, method="heads", importance_metric="magnitude",
  min_heads_per_layer=1). Added BEFORE any trainer code per lessons.md pattern.
- `xlmtec/trainers/structured_pruner.py` — `StructuredPruner` class.
  Soft structured attention-head pruning: scores each head by mean absolute
  weight magnitude, zeroes the bottom `sparsity` fraction per layer, respects
  `min_heads_per_layer` safety floor. Supports `method="heads"` (attention
  head rows) and `method="ffn"` (FFN gate/fc1 neuron rows). Auto-detects
  transformer layer structure across GPT-2, LLaMA, OPT, BERT-style models.
  Returns `PruningResult` frozen dataclass. No training loop — model is
  modified in-place and saved via `save_pretrained`.
- `xlmtec/trainers/__init__.py` — `StructuredPruner`, `PruningResult`
  exported.
- `cli/main.py` — `prune` subcommand. Args: model_path, --output, --sparsity
  (default 0.3), --method (heads|ffn), --min-heads (default 1).
- `tests/test_structured_pruner.py` — 16 unit tests: PruningConfig validation
  (4), helper functions `_head_importance_scores`, `_zero_head_rows`,
  `_count_params` (4), attention-head pruning flow (7), FFN pruning (1).
  No GPU, no real torch tensors in fixtures.
- `tests/test_prune.py` — 7 CLI-level tests: missing path, invalid sparsity,
  invalid method, happy path exits 0, sparsity forwarded to config, ffn method
  accepted, output contains sparsity achieved.
- `examples/configs/structured_pruning.yaml` — runnable example with
  sparsity guidance table.
- `tasks/roadmap.md` — Structured Pruning marked ✅ Sprint 29.

### Changed
- `pyproject.toml` — version 3.11.0 → 3.12.0
- `audit_repo.py` — new files registered.

---

## [3.11.0] — Sprint 28: "TUI Upload + Polish" — 2026-03-04

### Added
- `xlmtec/tui/screens/upload.py` — `UploadScreen`: model path Input,
  repo_id Input, token Input (`password=True` — masked), commit message Input,
  private `Switch`, merge-adapter `Switch` (reveals base model Input when toggled),
  inline validation. Builds `xlmtec upload` command → `RunningScreen`.
- `xlmtec/tui/app.css` — full Textual CSS theme. Styles all widget types
  (Input, Select, Checkbox, Switch, Button, DataTable, RichLog, CommandCard,
  running/result/form shared classes). Replaces inline CSS in `app.py`.
- `xlmtec/tui/screens/home.py` — Upload card now wired to `UploadScreen`.
  All 6 cards fully functional.
- `xlmtec/tui/app.py` — inline `CSS` block replaced with `CSS_PATH`
  pointing to `app.css`.
- `docs/tui.md` — full usage guide: install, home screen ASCII art, global
  keybindings table, per-command field reference, running/result screen ASCII
  art, env vars, headless testing instructions.
- `tests/test_tui.py` — 9 new Pilot tests: upload card navigation, form fields,
  token masking, private/merge switches, back nav, empty validation, all-6-cards
  integration test. Total: 55 tests.

### Changed
- `pyproject.toml` — version 3.10.0 → 3.11.0
- `audit_repo.py` — upload.py, app.css, docs/tui.md registered.

---

## [3.10.0] — Sprint 27: "TUI Evaluate, Benchmark, Merge" — 2026-03-04

### Added
- `xlmtec/tui/screens/evaluate.py` — `EvaluateScreen`: model path Input,
  dataset Input, 5-metric Checkbox group (ROUGE-1/2/L, BLEU, Perplexity — default
  ROUGE pre-selected), max-samples Input, optional report path. Builds
  `xlmtec evaluate` command → `RunningScreen`.
- `xlmtec/tui/screens/benchmark.py` — `BenchmarkScreen`: base model Input,
  fine-tuned path Input, dataset Input, same metric checkboxes, max-samples, report
  path. Builds `xlmtec evaluate benchmark` command → `RunningScreen`.
- `xlmtec/tui/screens/merge.py` — `MergeScreen`: adapter dir Input, base model
  Input, output dir Input, dtype `Select` (float32/float16/bfloat16). Builds
  `xlmtec merge` command → `RunningScreen`.
- `xlmtec/tui/screens/home.py` — Evaluate, Benchmark, Merge cards now push
  real screens. Only Upload remains as a stub (Sprint 28).
- `tests/test_tui.py` — 16 new Pilot tests across 3 new screen classes (card nav,
  form fields render, back nav, empty-submit validation). Total: 46 tests.

### Changed
- `pyproject.toml` — version 3.9.0 → 3.10.0
- `audit_repo.py` — 3 new screen files registered.
- `tests/test_tui.py` — stub card tests updated to use `#card-upload` (last
  remaining stub until Sprint 28).

---

## [3.9.0] — Sprint 26: "TUI Train & Recommend" — 2026-03-04

### Added
- `xlmtec/tui/widgets/log_panel.py` — `LogPanel`: scrolling `RichLog` widget,
  `write_line()` + `clear()` helpers, auto-scroll reactive.
- `xlmtec/tui/widgets/metric_table.py` — `MetricTable`: `DataTable` widget
  with `populate(dict)` helper for displaying result key/value pairs.
- `xlmtec/tui/screens/running.py` — `RunningScreen`: runs CLI command in
  background thread via `@work(thread=True)`. Streams stdout/stderr to `LogPanel`.
  Elapsed timer ticks every second. Cancel via button or q/Ctrl+C → home.
  On finish → switches to `ResultScreen`.
- `xlmtec/tui/screens/result.py` — `ResultScreen`: success/failure banner,
  `MetricTable` of results, Home + Quit buttons.
- `xlmtec/tui/screens/train.py` — `TrainScreen`: full form (model Input,
  method Select with all 7 methods, dataset Input, epochs Input, lr Input, output
  Input). Inline validation. Submit → `RunningScreen` with `xlmtec train` cmd.
- `xlmtec/tui/screens/recommend.py` — `RecommendScreen`: model Input,
  optional output path Input. Submit → `RunningScreen` with `xlmtec recommend`.
- `xlmtec/tui/screens/home.py` — Train + Recommend cards now push real screens.
  Evaluate/Benchmark/Merge/Upload show "coming in Sprint 27" toast stub.
- `tests/test_tui.py` — 14 new Pilot tests: train card navigation, form fields
  render, back button returns home, empty submit shows validation, recommend card
  navigation, result screen success/failure, result home button. Total: 30 tests.

### Changed
- `pyproject.toml` — version 3.8.0 → 3.9.0
- `audit_repo.py` — 6 new tui files registered.

---

## [3.8.0] — Sprint 25: "TUI Foundation" — 2025-03-03

### Added
- `xlmtec/tui/__init__.py` — TUI package.
- `xlmtec/tui/app.py` — `FinetuneApp(App)`: root Textual app, screen stack
  management, global keybindings (q=quit, h/esc=go_home, ctrl+c=quit),
  `on_mount` pushes HomeScreen, global CSS theme.
- `xlmtec/tui/screens/__init__.py` — screens package.
- `xlmtec/tui/screens/home.py` — `HomeScreen`: 6 `CommandCard` widgets in a
  3×2 `Grid`, header with clock, subtitle bar, footer with keybinding hints.
  `on_command_card_selected` stub (full routing Sprint 26+).
- `xlmtec/tui/widgets/__init__.py` — widgets package.
- `xlmtec/tui/widgets/command_card.py` — `CommandCard(Widget)`: focusable
  styled card with label, description, icon, hover/focus CSS states.
  Posts `CommandCard.Selected` message on click or Enter.
- `xlmtec/cli/main.py` — `tui` subcommand added (lazy import, graceful
  ImportError message if textual not installed).
- `tests/test_tui.py` — 10 Textual Pilot headless tests: app mounts, HomeScreen
  is initial screen, 6 cards render, card IDs correct, cards focusable,
  title/subtitle labels present, q exits, tab moves focus, escape stays on home,
  click posts Selected message.
- `tasks/roadmap.md` — TUI section added (Sprints 25–28 plan).
- `tasks/todo.md` — TUI sprint checklists 25–28 added as not-started blocks.
- `tasks/CONTEXT.md` — planned sprint rows 25–28 added + TUI Rule section.

### Changed (sprint-end checklist)
- `pyproject.toml` — textual>=0.52.0 added to dependencies; version 3.7.0 → 3.8.0.
- `audit_repo.py` — all new tui/ files registered.

---

## [3.8.0] — Sprint 25: "TUI Foundation" — 2026-03-04

### Added
- `xlmtec/tui/__init__.py` — TUI package init.
- `xlmtec/tui/app.py` — `FinetuneApp(App)` root Textual app. Global bindings:
  `q`=quit, `h`/`escape`=home, `ctrl+c`=quit. `action_go_home()` pops all screens
  above home. `run()` entry point called by CLI.
- `xlmtec/tui/screens/__init__.py` — screens package init.
- `xlmtec/tui/screens/home.py` — `HomeScreen`: 6 `CommandCard` widgets in a
  3×2 CSS grid. Arrow-key nav (wraps), Tab/Shift+Tab, Enter to select. Sprint 25
  shows a `notify()` stub — real screen push wired in Sprint 26+.
- `xlmtec/tui/widgets/__init__.py` — widgets package init.
- `xlmtec/tui/widgets/command_card.py` — `CommandCard(Widget)`: focusable card
  with icon, bold label, description. Hover + focus border highlight via CSS.
  Posts `CommandCard.Selected` on click or Enter. `can_focus = True`.
- `cli/main.py` — `tui` subcommand added (lazy import of `FinetuneApp`, graceful
  error if textual not installed).
- `tests/test_tui.py` — 16 async Pilot tests (headless): app mounts, HomeScreen is
  initial screen, title set, 6 cards rendered, card IDs correct, cards focusable,
  title/subtitle labels present, q exits, tab moves focus, escape stays home,
  right/down arrow navigate, click/enter don't crash, card_id matches attribute.
- `pyproject.toml` — `textual>=0.52.0` added to dependencies;
  `pytest-asyncio>=0.21.0` added to dev; `asyncio_mode = "auto"` added to pytest.

### Changed
- `pyproject.toml` — version 3.7.0 → 3.8.0

---

## [3.7.0] — Sprint 24: "Feature Distillation" — 2025-03-02

### Added
- `xlmtec/core/types.py` — `FeatureDistillationConfig` frozen dataclass
  (teacher_model_name, temperature=2.0, alpha=0.3, beta=0.2,
  feature_layers=None, feature_loss_weight=1.0).
- `xlmtec/trainers/feature_distillation_trainer.py` — `FeatureDistillationTrainer`
  + `_FeatureDistillationTrainer` (HF Trainer subclass). Three-component loss:
  alpha*CE + beta*KL_output*T² + gamma*MSE_hidden. Auto layer selection when
  feature_layers=None. Dimension mismatch fallback via L2 normalisation.
  `ResourceWarning` for models >1B params.
- `_select_layers()` helper — evenly-spaced auto selection or explicit validation.
- `_map_teacher_layer()` helper — proportional student→teacher layer index mapping.
- `xlmtec/trainers/factory.py` — `FEATURE_DISTILLATION` wired to
  `FeatureDistillationTrainer`; `feature_distillation_config` param added.
- `xlmtec/trainers/__init__.py` — `FeatureDistillationTrainer` exported.
- `tests/test_feature_distillation_trainer.py` — 21 unit tests: `_select_layers`
  (5), `_map_teacher_layer` (3), factory dispatch (2), init/VRAM (3),
  `_setup_peft` (2), `train()` flow (6).
- `examples/configs/feature_distillation.yaml` — runnable local config.
- `tasks/roadmap.md` — Feature Distillation marked ✅ Sprint 24.
- `audit_repo.py` — new trainer, test file, example config registered.

### Fixed (sprint-end checklist)
- `pyproject.toml` — version 3.6.0 → 3.7.0

---

## [3.6.0] — Sprint 23: "Response Distillation" — 2025-03-01

### Added
- `xlmtec/trainers/response_distillation_trainer.py` — `ResponseDistillationTrainer`
  extending `BaseTrainer`. Loads teacher (frozen, eval mode), runs KL divergence +
  CE blended loss via internal `_DistillationTrainer` HF Trainer subclass.
  Issues `ResourceWarning` for student models >1B params (dual-model VRAM concern).
- `xlmtec/core/types.py` — `DistillationConfig` frozen dataclass
  (teacher_model_name, temperature=2.0, alpha=0.5).
- `xlmtec/trainers/factory.py` — `VANILLA_DISTILLATION` wired to
  `ResponseDistillationTrainer`; `distillation_config` param added to `create()` and `train()`.
- `xlmtec/trainers/__init__.py` — `ResponseDistillationTrainer` exported.
- `tests/test_response_distillation_trainer.py` — 12 unit tests: factory dispatch,
  MissingConfigError, config stored, VRAM warning/no-warning, _setup_peft, train() flow,
  eval_loss extracted, model saved, teacher frozen, teacher load failure → TrainingError.
- `examples/configs/response_distillation.yaml` — runnable local config (gpt2 → gpt2-medium).
- `tasks/roadmap.md` — Response Distillation marked ✅ Sprint 23.
- `audit_repo.py` — new trainer, test file, example config registered.

### Fixed (sprint-end checklist)
- `pyproject.toml` — version 3.5.0 → 3.6.0

---

## [3.5.0] — Sprint 21: "Meta Sync" — 2025-03-01

### Fixed
- `CLAUDE.md` — sprint history frozen at Sprint 12 → updated through Sprint 21; version 2.0.0 → 3.4.0; added two new architecture rules (absolute imports, no real tensors); trainer table added; test commands updated
- `audit_repo.py` — three trainer paths wrong (`trainers/` → `xlmtec/trainers/`); added missing top-level files (`README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `audit_repo.py`, `.gitignore`); added `xlmtec/data/pipeline.py`; CONTEXT.md paths corrected to `xlmtec/` prefix
- `pyproject.toml` — version 3.4.0 → 3.5.0

---

## [3.4.0] — Sprint 20: "Import Audit Complete" — 2025-03-01

### Fixed
- `tests/test_config.py` — `from ..core.config`, `from ..core.types`, `from ..core.exceptions` → absolute imports
- `tests/test_full_trainer.py` — relative imports → absolute; replaced `torch.nn.Parameter(torch.randn(...))` with pure `_make_param()` MagicMock helper (no real tensors, no torch required per lessons.md)
- `pyproject.toml` — version 3.3.0 → 3.4.0

---

## [3.3.0] — Sprint 19: "Test Import Audit" — 2025-03-01

### Fixed
- `tests/test_recommend.py` — `from ..cli.main import app` → absolute import
- `tests/test_evaluation.py` — all `from ..core` / `from ..evaluation` → absolute imports
- `tests/test_instruction_trainer.py` — top-level and inline relative imports all → absolute
- `tasks/lessons.md` — 4 new patterns: YAML path posix, stale unsupported-method tests,
  fixture-called-directly, enum iteration scope
- `pyproject.toml` — version 3.2.0 → 3.3.0

---

## [3.2.0] — Sprint 18: "conftest Hardening" — 2025-02-28

### Fixed
- `tests/conftest.py` — removed `import torch` and `torch.randn` from `mock_model`.
  Replaced with pure `MagicMock` param (`numel.return_value=1_000_000`,
  `requires_grad=True`). Used `side_effect` instead of `return_value` so
  `parameters()` returns a fresh iterator on every call (was silently exhausted).
  conftest now importable with zero ML deps.
- `tests/test_qlora_trainer.py` — removed 3 duplicate local fixtures
  (`mock_model`, `mock_tokenizer`, `tmp_output_dir`) now inherited from conftest.
- `tests/CONTEXT.md` — 8 missing test file rows added; patch-target rule and
  conftest torch rule added to Rules section.
- `tasks/lessons.md` — 3 new patterns: conftest no-torch, parameters() side_effect,
  patch-target-at-usage-site.
- `pyproject.toml` — version 3.1.0 → 3.2.0

---

## [3.1.0] — Sprint 16: "Data Pipeline Tests" — 2025-02-28

### Added
- `tests/test_data.py` — 11 unit tests: `detect_columns` (text/prompt/response/content/numeric columns),
  error cases (`DatasetNotFoundError`, `EmptyDatasetError`, `NoTextColumnsError`),
  `quick_load` happy path (returns Dataset, max_samples forwarded),
  `prepare_dataset` happy path (no split, with split returns DatasetDict).
  No HF downloads, no GPU. `DataPipeline.run()` patched for wiring tests.
- `audit_repo.py` — `tests/test_data.py` registered; missing comma after
  `test_qlora_trainer.py` fixed (was causing silent SyntaxError)

### Fixed (sprint-end checklist)
- `tasks/lessons.md` — lazy-import patch pattern added (overdue from Sprint 15 fix);
  DataPipeline mock pattern added
- `tasks/CONTEXT.md` — Sprint 16 row added
- `CLAUDE.md` — sprint history updated through Sprint 16
- `pyproject.toml` — version 3.0.0 → 3.1.0
- `docs/index.md` — version 3.1.0, test count 143+
- `tasks/todo.md` — Sprint 16 acceptance gate recorded

---

## [3.0.0] — Sprint 15: "QLoRA Tests + Sync" — 2025-02-28

### Added
- `tests/test_qlora_trainer.py` — 8 unit tests: factory dispatch (creates QLoRATrainer, raises
  MissingConfigError for missing lora_config/model_config), init (stores config, warns when
  load_in_4bit=False, no warning when True), _setup_peft (kbit prep called before LoRA,
  gradient_checkpointing forwarded), full train() mock (returns TrainingResult)
- `audit_repo.py` — `tests/test_qlora_trainer.py` added to REQUIRED_FILES

### Fixed (sprint-end checklist drift)
- `tasks/CONTEXT.md` — Sprint 13 "Test Coverage Complete" and Sprint 14 "Sprint 13 Close-out" rows added
- `CLAUDE.md` — sprint history table updated through Sprint 15
- `pyproject.toml` — version 2.9.3 → 3.0.0
- `docs/index.md` — version 3.0.0, test count 132+
- `tasks/todo.md` — Sprint 15 acceptance gate recorded

---

## [2.9.3] — Sprint 14: "Sprint 13 Close-out" — 2025-02-27

### Fixed (sprint-end checklist)
- `pyproject.toml` — version 2.8.0 → 2.9.2
- `docs/index.md` — test count updated to 124+
- `tasks/todo.md` — Sprint 13 acceptance gate marked complete
- `tasks/lessons.md` — lazy-import patch pattern added (patch at source, not at cli.main)

---

## [2.9.2] — Sprint 13: "Test Coverage Complete" — 2025-02-27

### Added
- `tests/test_evaluate.py` — 6 unit tests for `xlmtec evaluate`
- `tests/test_benchmark.py` — 6 unit tests for `xlmtec benchmark`
- `tests/test_upload.py` — 7 unit tests for `xlmtec upload`
- `tests/test_cli_train.py` — `test_full_finetuning_via_flags` added
- `audit_repo.py` — three new test files added to REQUIRED_FILES

---

## [2.9.1] — Sprint 12: "Usage Guide Current" — 2025-02-27

### Fixed
- `docs/usage.md` — fully rewritten: all 6 commands, all 5 training methods
- `CLAUDE.md` — trainer checklist expanded to 11 steps

---

## [2.9.0] — Sprint 11: "Version Sync" — 2025-02-27

### Fixed
- `pyproject.toml` — version bumped to 2.8.0
- `tasks/CONTEXT.md` + `CONTRIBUTING.md` — sprint-end checklist added

---

## [2.8.0] — Sprint 10: "DPO Runnable" — 2025-02-27

### Added
- `examples/generate_sample_data.py` — `generate_dpo_samples(200)`
- `examples/configs/dpo.yaml` — switched to local_file; fully offline-runnable
- `pyproject.toml` — optional dep `[dpo] = ["trl>=0.7.0"]`
- `docs/configuration.md` — DPO section

---

## [2.7.0] — Sprint 9: "Housekeeping" — 2025-02-27

### Fixed
- `tasks/CONTEXT.md`, `CLAUDE.md`, `docs/index.md`, `trainers/CONTEXT.md`, `docs/api.md` synced

---

## [2.6.0] — Sprint 8: "DPO" — 2025-02-27

### Added
- `trainers/dpo_trainer.py`, `validate_dpo_dataset()`, factory wired
- `tests/test_dpo_trainer.py` — 10 tests
- `tests/test_cli_train.py` — `test_dpo_via_flags`

---

## [2.5.0] — Sprint 7: "CI Tight" — 2025-02-27

### Fixed
- `ci.yml` — pytest-timeout installed, paths corrected
- `tasks/CONTEXT.md` — Sprints 4-6 rows added
- `docs/index.md` — version, test count, component table updated

---

## [2.4.0] — Sprint 6: "Documented" — 2025-02-27

### Added
- `README.md`, `CONTRIBUTING.md`, `docs/api.md` fully rewritten for v2

---

## [2.3.0] — Sprint 5: "Merge & Release" — 2025-02-27

### Added
- `xlmtec merge` subcommand
- `tests/test_merge.py` — 8 tests

---

## Earlier Sprints (1–4)

Foundation, Expand, First Run, Hardened — see git log.