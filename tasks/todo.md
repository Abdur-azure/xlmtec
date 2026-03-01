# Tasks

## Status Legend
- [ ] = Todo
- [x] = Complete

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