# Changelog

All notable changes to this project are documented here.

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