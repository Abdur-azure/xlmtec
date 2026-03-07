# tasks/ — Context

Session state. Read both files at the start of every working session.

## Files

| File | Purpose |
|------|---------|
| `todo.md` | Sprint-structured task list. Mark items `[x]` as you go. Includes sprint name, acceptance gate, and history of completed sprints. |
| `lessons.md` | Accumulated patterns from corrections. Read before implementing anything — most bugs we've hit before are documented here. |
| `roadmap.md` | Validated feature roadmap — 9 methods across fine-tuning, distillation, pruning. Update status column when a method ships. |

## Sprints so far

| Sprint | Name | Outcome |
|--------|------|---------|
| 1 | Stable Foundation | All tests green, CI, pyproject.toml, docs |
| 2 | Expand | FullFineTuner, InstructionTrainer, recommend command |
| 3 | First Run | Runnable examples, sample data generator, integration tests |
| 4 | Hardened | CLI lora guard, test_cli_train.py, CHANGELOG, audit_repo |
| 5 | Merge & Release | xlmtec merge command, test_merge.py (8 tests) |
| 6 | Documented | README, CONTRIBUTING, api.md, docs/index.md all updated |
| 7 | CI Tight | ci.yml paths, pytest-timeout, ruff lint section, absolute imports |
| 8 | DPO | DPOTrainer, validate_dpo_dataset, factory wired, 10 tests |
| 9 | Housekeeping | CONTEXT.md, CLAUDE.md, docs/index.md, api.md synced |
| 10 | DPO Runnable | dpo_sample.jsonl generator, local config, trl optional dep |
| 11 | Version Sync | pyproject.toml 2.8.0, CONTRIBUTING sprint-end checklist |
| 12 | Usage Guide Current | docs/usage.md all 6 commands, CLAUDE.md 11-step checklist |
| 13 | Test Coverage Complete | test_evaluate.py, test_benchmark.py, test_upload.py (19 tests green) |
| 14 | Sprint 13 Close-out | pyproject.toml 2.9.2, docs/index.md 124+ tests, CHANGELOG |
| 15 | QLoRA Tests + Sync | test_qlora_trainer.py (8 tests), CONTEXT.md/CLAUDE.md Sprint 13-14 rows |
| 16 | Data Pipeline Tests | test_data.py (11 tests), lessons.md updated, audit_repo.py comma fix |
| 18 | conftest Hardening | conftest.py no-torch, test_qlora deduped, tests/CONTEXT.md 8 rows |
| 19 | Test Import Audit | test_recommend, test_evaluation, test_instruction_trainer absolute imports |
| 20 | Import Audit Complete | test_config, test_full_trainer absolute imports + no-real-tensor fix |
| 21 | Meta Sync | CLAUDE.md sprints 13-20, audit_repo.py paths fixed |
| 23 | Response Distillation | DistillationConfig in core/types.py, ResponseDistillationTrainer, 12 tests, example config |
| 24 | Feature Distillation | FeatureDistillationConfig, FeatureDistillationTrainer, 21 tests, layer helpers |
| 25 | TUI Foundation | ✅ FinetuneApp, HomeScreen, 6 CommandCards, `xlmtec tui`, 16 Pilot tests |
| 26 | TUI Train & Recommend | ✅ Complete — TrainScreen, RecommendScreen, RunningScreen, ResultScreen, LogPanel, MetricTable, 30 tests |
| 27 | TUI Evaluate, Benchmark, Merge | ✅ Complete — EvaluateScreen, BenchmarkScreen, MergeScreen, 46 tests |
| 28 | TUI Upload + Polish | ✅ Complete — UploadScreen, app.css, docs/tui.md, 55 tests |
| 29 | Structured Pruning | ✅ Complete — PruningConfig, StructuredPruner, prune CLI command, 23 tests |
| 30 | WANDA Pruning | ✅ Complete — WandaConfig, WandaPruner, wanda CLI command, 24 tests |
| 31 | Integration Hardening | ✅ Complete — 15 new integration tests (distill×2, prune×2, CLI smoke×4), ci.yml fixed |
| 32 | Stabilise |
| 33 | Post-Rename Cleanup | Renamed xlmtec → xlmtec; pyproject.toml, imports, docs, CLI entrypoint | ✅ Complete — pyproject.toml extras split, ci.yml 5 fixes, audit_repo typo, CLAUDE.md paths, installation.md |

## TUI Rule

Sprints 25–28 are **additive only**. The TUI is purely additive. All Typer commands remain unchanged.