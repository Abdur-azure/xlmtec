# Changelog

All notable changes to this project are documented here.

---

## [3.13.0] — Sprint 30: "WANDA Pruning" — 2026-03-05

### Added
- `finetune_cli/core/types.py` — `WandaConfig` frozen dataclass
  (output_dir, sparsity=0.5, n_calibration_samples=128,
  calibration_seq_len=128, layer_types=None, use_row_wise=True).
- `finetune_cli/trainers/wanda_pruner.py` — `WandaPruner` + `WandaResult`.
  WANDA unstructured pruning: registers forward hooks to collect input
  activation norms over a calibration dataset, scores each weight by
  |W_ij| * ||X_j||_2, zeros the bottom `sparsity` fraction per layer.
  Falls back to magnitude-only scoring when no calibration data is
  provided. Supports row-wise and global sparsity modes. Auto-targets
  nn.Linear and Conv1D layers. Returns `WandaResult` frozen dataclass.
- `finetune_cli/trainers/__init__.py` — `WandaPruner`, `WandaResult` exported.
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
- `finetune_cli/core/types.py` — `PruningConfig` frozen dataclass
  (output_dir, sparsity=0.3, method="heads", importance_metric="magnitude",
  min_heads_per_layer=1). Added BEFORE any trainer code per lessons.md pattern.
- `finetune_cli/trainers/structured_pruner.py` — `StructuredPruner` class.
  Soft structured attention-head pruning: scores each head by mean absolute
  weight magnitude, zeroes the bottom `sparsity` fraction per layer, respects
  `min_heads_per_layer` safety floor. Supports `method="heads"` (attention
  head rows) and `method="ffn"` (FFN gate/fc1 neuron rows). Auto-detects
  transformer layer structure across GPT-2, LLaMA, OPT, BERT-style models.
  Returns `PruningResult` frozen dataclass. No training loop — model is
  modified in-place and saved via `save_pretrained`.
- `finetune_cli/trainers/__init__.py` — `StructuredPruner`, `PruningResult`
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
- `finetune_cli/tui/screens/upload.py` — `UploadScreen`: model path Input,
  repo_id Input, token Input (`password=True` — masked), commit message Input,
  private `Switch`, merge-adapter `Switch` (reveals base model Input when toggled),
  inline validation. Builds `finetune-cli upload` command → `RunningScreen`.
- `finetune_cli/tui/app.css` — full Textual CSS theme. Styles all widget types
  (Input, Select, Checkbox, Switch, Button, DataTable, RichLog, CommandCard,
  running/result/form shared classes). Replaces inline CSS in `app.py`.
- `finetune_cli/tui/screens/home.py` — Upload card now wired to `UploadScreen`.
  All 6 cards fully functional.
- `finetune_cli/tui/app.py` — inline `CSS` block replaced with `CSS_PATH`
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
- `finetune_cli/tui/screens/evaluate.py` — `EvaluateScreen`: model path Input,
  dataset Input, 5-metric Checkbox group (ROUGE-1/2/L, BLEU, Perplexity — default
  ROUGE pre-selected), max-samples Input, optional report path. Builds
  `finetune-cli evaluate` command → `RunningScreen`.
- `finetune_cli/tui/screens/benchmark.py` — `BenchmarkScreen`: base model Input,
  fine-tuned path Input, dataset Input, same metric checkboxes, max-samples, report
  path. Builds `finetune-cli evaluate benchmark` command → `RunningScreen`.
- `finetune_cli/tui/screens/merge.py` — `MergeScreen`: adapter dir Input, base model
  Input, output dir Input, dtype `Select` (float32/float16/bfloat16). Builds
  `finetune-cli merge` command → `RunningScreen`.
- `finetune_cli/tui/screens/home.py` — Evaluate, Benchmark, Merge cards now push
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
- `finetune_cli/tui/widgets/log_panel.py` — `LogPanel`: scrolling `RichLog` widget,
  `write_line()` + `clear()` helpers, auto-scroll reactive.
- `finetune_cli/tui/widgets/metric_table.py` — `MetricTable`: `DataTable` widget
  with `populate(dict)` helper for displaying result key/value pairs.
- `finetune_cli/tui/screens/running.py` — `RunningScreen`: runs CLI command in
  background thread via `@work(thread=True)`. Streams stdout/stderr to `LogPanel`.
  Elapsed timer ticks every second. Cancel via button or q/Ctrl+C → home.
  On finish → switches to `ResultScreen`.
- `finetune_cli/tui/screens/result.py` — `ResultScreen`: success/failure banner,
  `MetricTable` of results, Home + Quit buttons.
- `finetune_cli/tui/screens/train.py` — `TrainScreen`: full form (model Input,
  method Select with all 7 methods, dataset Input, epochs Input, lr Input, output
  Input). Inline validation. Submit → `RunningScreen` with `finetune-cli train` cmd.
- `finetune_cli/tui/screens/recommend.py` — `RecommendScreen`: model Input,
  optional output path Input. Submit → `RunningScreen` with `finetune-cli recommend`.
- `finetune_cli/tui/screens/home.py` — Train + Recommend cards now push real screens.
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
- `finetune_cli/tui/__init__.py` — TUI package.
- `finetune_cli/tui/app.py` — `FinetuneApp(App)`: root Textual app, screen stack
  management, global keybindings (q=quit, h/esc=go_home, ctrl+c=quit),
  `on_mount` pushes HomeScreen, global CSS theme.
- `finetune_cli/tui/screens/__init__.py` — screens package.
- `finetune_cli/tui/screens/home.py` — `HomeScreen`: 6 `CommandCard` widgets in a
  3×2 `Grid`, header with clock, subtitle bar, footer with keybinding hints.
  `on_command_card_selected` stub (full routing Sprint 26+).
- `finetune_cli/tui/widgets/__init__.py` — widgets package.
- `finetune_cli/tui/widgets/command_card.py` — `CommandCard(Widget)`: focusable
  styled card with label, description, icon, hover/focus CSS states.
  Posts `CommandCard.Selected` message on click or Enter.
- `finetune_cli/cli/main.py` — `tui` subcommand added (lazy import, graceful
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
- `finetune_cli/tui/__init__.py` — TUI package init.
- `finetune_cli/tui/app.py` — `FinetuneApp(App)` root Textual app. Global bindings:
  `q`=quit, `h`/`escape`=home, `ctrl+c`=quit. `action_go_home()` pops all screens
  above home. `run()` entry point called by CLI.
- `finetune_cli/tui/screens/__init__.py` — screens package init.
- `finetune_cli/tui/screens/home.py` — `HomeScreen`: 6 `CommandCard` widgets in a
  3×2 CSS grid. Arrow-key nav (wraps), Tab/Shift+Tab, Enter to select. Sprint 25
  shows a `notify()` stub — real screen push wired in Sprint 26+.
- `finetune_cli/tui/widgets/__init__.py` — widgets package init.
- `finetune_cli/tui/widgets/command_card.py` — `CommandCard(Widget)`: focusable card
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
- `finetune_cli/core/types.py` — `FeatureDistillationConfig` frozen dataclass
  (teacher_model_name, temperature=2.0, alpha=0.3, beta=0.2,
  feature_layers=None, feature_loss_weight=1.0).
- `finetune_cli/trainers/feature_distillation_trainer.py` — `FeatureDistillationTrainer`
  + `_FeatureDistillationTrainer` (HF Trainer subclass). Three-component loss:
  alpha*CE + beta*KL_output*T² + gamma*MSE_hidden. Auto layer selection when
  feature_layers=None. Dimension mismatch fallback via L2 normalisation.
  `ResourceWarning` for models >1B params.
- `_select_layers()` helper — evenly-spaced auto selection or explicit validation.
- `_map_teacher_layer()` helper — proportional student→teacher layer index mapping.
- `finetune_cli/trainers/factory.py` — `FEATURE_DISTILLATION` wired to
  `FeatureDistillationTrainer`; `feature_distillation_config` param added.
- `finetune_cli/trainers/__init__.py` — `FeatureDistillationTrainer` exported.
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
- `finetune_cli/trainers/response_distillation_trainer.py` — `ResponseDistillationTrainer`
  extending `BaseTrainer`. Loads teacher (frozen, eval mode), runs KL divergence +
  CE blended loss via internal `_DistillationTrainer` HF Trainer subclass.
  Issues `ResourceWarning` for student models >1B params (dual-model VRAM concern).
- `finetune_cli/core/types.py` — `DistillationConfig` frozen dataclass
  (teacher_model_name, temperature=2.0, alpha=0.5).
- `finetune_cli/trainers/factory.py` — `VANILLA_DISTILLATION` wired to
  `ResponseDistillationTrainer`; `distillation_config` param added to `create()` and `train()`.
- `finetune_cli/trainers/__init__.py` — `ResponseDistillationTrainer` exported.
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
- `audit_repo.py` — three trainer paths wrong (`trainers/` → `finetune_cli/trainers/`); added missing top-level files (`README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `audit_repo.py`, `.gitignore`); added `finetune_cli/data/pipeline.py`; CONTEXT.md paths corrected to `finetune_cli/` prefix
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
- `tests/test_evaluate.py` — 6 unit tests for `finetune-cli evaluate`
- `tests/test_benchmark.py` — 6 unit tests for `finetune-cli benchmark`
- `tests/test_upload.py` — 7 unit tests for `finetune-cli upload`
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
- `finetune-cli merge` subcommand
- `tests/test_merge.py` — 8 tests

---

## Earlier Sprints (1–4)

Foundation, Expand, First Run, Hardened — see git log.