"""
audit_repo.py — run from the repo root to find missing files.

Usage:
    python audit_repo.py

Prints every file that should exist but doesn't.
"""

from pathlib import Path

REQUIRED_FILES = [
    # ── Root ─────────────────────────────────────────────────────────────
    "conftest.py",
    "pyproject.toml",
    "audit_repo.py",
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    ".github/workflows/ci.yml",
    ".gitignore",

    # ── Main package ─────────────────────────────────────────────────────
    "xlmtec/__init__.py",

    # ── Core ─────────────────────────────────────────────────────────────
    "xlmtec/core/__init__.py",
    "xlmtec/core/exceptions.py",
    "xlmtec/core/types.py",
    "xlmtec/core/config.py",
    "xlmtec/core/CONTEXT.md",

    # ── Utils ─────────────────────────────────────────────────────────────
    "xlmtec/utils/__init__.py",
    "xlmtec/utils/logging.py",

    # ── Models ────────────────────────────────────────────────────────────
    "xlmtec/models/__init__.py",
    "xlmtec/models/loader.py",

    # ── Trainers ──────────────────────────────────────────────────────────
    "xlmtec/trainers/__init__.py",
    "xlmtec/trainers/base.py",
    "xlmtec/trainers/lora_trainer.py",
    "xlmtec/trainers/qlora_trainer.py",
    "xlmtec/trainers/full_trainer.py",
    "xlmtec/trainers/instruction_trainer.py",
    "xlmtec/trainers/dpo_trainer.py",
    "xlmtec/trainers/response_distillation_trainer.py",
    "xlmtec/trainers/feature_distillation_trainer.py",
    "xlmtec/trainers/structured_pruner.py",
    "xlmtec/trainers/wanda_pruner.py",
    "xlmtec/trainers/factory.py",
    "xlmtec/trainers/CONTEXT.md",

    # ── TUI ───────────────────────────────────────────────────────────────
    "xlmtec/tui/__init__.py",
    "xlmtec/tui/app.py",
    "xlmtec/tui/app.css",
    "xlmtec/tui/screens/__init__.py",
    "xlmtec/tui/screens/home.py",
    "xlmtec/tui/screens/train.py",
    "xlmtec/tui/screens/recommend.py",
    "xlmtec/tui/screens/evaluate.py",
    "xlmtec/tui/screens/benchmark.py",
    "xlmtec/tui/screens/merge.py",
    "xlmtec/tui/screens/upload.py",
    "xlmtec/tui/screens/running.py",
    "xlmtec/tui/screens/result.py",
    "xlmtec/tui/widgets/__init__.py",
    "xlmtec/tui/widgets/command_card.py",
    "xlmtec/tui/widgets/log_panel.py",
    "xlmtec/tui/widgets/metric_table.py",

    # ── Data ──────────────────────────────────────────────────────────────
    "xlmtec/data/__init__.py",
    "xlmtec/data/pipeline.py",
    "xlmtec/data/CONTEXT.md",

    # ── Evaluation ────────────────────────────────────────────────────────
    "xlmtec/evaluation/__init__.py",
    "xlmtec/evaluation/metrics.py",
    "xlmtec/evaluation/benchmarker.py",
    "xlmtec/evaluation/CONTEXT.md",

    # ── CLI ───────────────────────────────────────────────────────────────
    "xlmtec/cli/__init__.py",
    "xlmtec/cli/main.py",
    "xlmtec/cli/CONTEXT.md",

    # ── Tests ─────────────────────────────────────────────────────────────
    "tests/conftest.py",
    "tests/test_config.py",
    "tests/test_trainers.py",
    "tests/test_full_trainer.py",
    "tests/test_instruction_trainer.py",
    "tests/test_qlora_trainer.py",
    "tests/test_dpo_trainer.py",
    "tests/test_response_distillation_trainer.py",
    "tests/test_feature_distillation_trainer.py",
    "tests/test_structured_pruner.py",
    "tests/test_wanda_pruner.py",
    "tests/test_wanda_cli.py",
    "tests/test_prune.py",
    "tests/test_tui.py",
    "tests/test_evaluation.py",
    "tests/test_recommend.py",
    "tests/test_cli_train.py",
    "tests/test_merge.py",
    "tests/test_evaluate.py",
    "tests/test_benchmark.py",
    "tests/test_upload.py",
    "tests/test_data.py",
    "tests/test_integration.py",
    "tests/CONTEXT.md",
    # Sprints 34–43
    "tests/test_integrations.py",
    "tests/test_ux.py",
    "tests/test_hub.py",
    "tests/test_checkpoints.py",
    "tests/test_templates.py",
    "tests/test_dashboard.py",
    "tests/test_export.py",
    "tests/test_inference.py",
    "tests/test_plugins.py",

    # ── Integrations (Sprint 34) ───────────────────────────────────────────
    "xlmtec/integrations/__init__.py",
    "xlmtec/integrations/base.py",
    "xlmtec/integrations/claude.py",
    "xlmtec/integrations/gemini.py",
    "xlmtec/integrations/codex.py",
    "xlmtec/integrations/prompt_builder.py",
    "xlmtec/integrations/response_parser.py",

    # ── CLI UX + commands (Sprints 35–36, 38–43) ──────────────────────────
    "xlmtec/cli/ux.py",
    "xlmtec/cli/commands/ai_suggest.py",
    "xlmtec/cli/commands/dry_run.py",
    "xlmtec/cli/commands/config_validate.py",
    "xlmtec/cli/commands/hub.py",
    "xlmtec/cli/commands/resume.py",
    "xlmtec/cli/commands/template.py",
    "xlmtec/cli/commands/dashboard.py",
    "xlmtec/cli/commands/export.py",
    "xlmtec/cli/commands/predict.py",
    "xlmtec/cli/commands/plugin.py",

    # ── Hub (Sprint 36) ───────────────────────────────────────────────────
    "xlmtec/hub/__init__.py",
    "xlmtec/hub/client.py",
    "xlmtec/hub/formatter.py",

    # ── Checkpoints (Sprint 38) ───────────────────────────────────────────
    "xlmtec/checkpoints/__init__.py",
    "xlmtec/checkpoints/manager.py",

    # ── Templates (Sprint 39) ─────────────────────────────────────────────
    "xlmtec/templates/__init__.py",
    "xlmtec/templates/registry.py",

    # ── Dashboard (Sprint 40) ─────────────────────────────────────────────
    "xlmtec/dashboard/__init__.py",
    "xlmtec/dashboard/reader.py",
    "xlmtec/dashboard/comparator.py",

    # ── Export (Sprint 41) ────────────────────────────────────────────────
    "xlmtec/export/__init__.py",
    "xlmtec/export/formats.py",
    "xlmtec/export/exporter.py",
    "xlmtec/export/backends/__init__.py",
    "xlmtec/export/backends/safetensors.py",
    "xlmtec/export/backends/onnx.py",
    "xlmtec/export/backends/gguf.py",

    # ── Inference (Sprint 42) ─────────────────────────────────────────────
    "xlmtec/inference/__init__.py",
    "xlmtec/inference/data_loader.py",
    "xlmtec/inference/writer.py",
    "xlmtec/inference/predictor.py",

    # ── Plugins (Sprint 43) ───────────────────────────────────────────────
    "xlmtec/plugins/__init__.py",
    "xlmtec/plugins/store.py",
    "xlmtec/plugins/loader.py",

    # ── Notifications (Sprint 47) ──────────────────────────────────────
    "xlmtec/notifications/__init__.py",
    "xlmtec/notifications/base.py",
    "xlmtec/notifications/slack.py",
    "xlmtec/notifications/email.py",
    "xlmtec/notifications/desktop.py",
    "xlmtec/notifications/dispatcher.py",
    "xlmtec/notifications/CONTEXT.md",
    "tests/test_notifications.py",

    # Telemetry / App Insights (Sprint 49-A) ──────────────────────────
    "xlmtec/utils/telemetry.py",
    "xlmtec/utils/crash_report.py",
    "xlmtec/cli/commands/report.py",
    "tests/test_telemetry.py",

    # ── CONTEXT.md files (Sprint 45) ──────────────────────────────────────
    "xlmtec/hub/CONTEXT.md",
    "xlmtec/checkpoints/CONTEXT.md",
    "xlmtec/templates/CONTEXT.md",
    "xlmtec/dashboard/CONTEXT.md",
    "xlmtec/export/CONTEXT.md",
    "xlmtec/inference/CONTEXT.md",
    "xlmtec/plugins/CONTEXT.md",

    # ── Docs (Sprint 37) ──────────────────────────────────────────────────
    "docs/ai_integrations.md",
    "docs/hub.md",
    # Sprint 46 docs
    "docs/resume.md",
    "docs/template.md",
    "docs/dashboard.md",
    "docs/export.md",
    "docs/predict.md",
    "docs/plugin.md",

    # ── Tasks ─────────────────────────────────────────────────────────────
    "tasks/todo.md",
    "tasks/lessons.md",
    "tasks/CONTEXT.md",
    "tasks/roadmap.md",

    # ── Examples ──────────────────────────────────────────────────────────
    "examples/generate_sample_data.py",
    "examples/configs/lora_gpt2.yaml",
    "examples/configs/qlora_llama.yaml",
    "examples/configs/instruction_tuning.yaml",
    "examples/configs/full_finetuning.yaml",
    "examples/configs/dpo.yaml",
    "examples/configs/response_distillation.yaml",
    "examples/configs/feature_distillation.yaml",
    "examples/configs/structured_pruning.yaml",
    "examples/configs/wanda.yaml",

    # ── Docs ──────────────────────────────────────────────────────────────
    "docs/tui.md",
    "docs/index.md",
    "docs/usage.md",
    "docs/api.md",
    "docs/configuration.md",

    # ── Meta ──────────────────────────────────────────────────────────────
    "CLAUDE.md",
]


def main() -> None:
    root = Path(__file__).parent
    missing = [f for f in REQUIRED_FILES if not (root / f).exists()]
    present = [f for f in REQUIRED_FILES if (root / f).exists()]

    print(f"\n{'=' * 55}")
    print(f"  REPO AUDIT — {root.name}")
    print(f"{'=' * 55}")
    print(f"  Present : {len(present)}/{len(REQUIRED_FILES)}")
    print(f"  Missing : {len(missing)}/{len(REQUIRED_FILES)}")
    print(f"{'=' * 55}")

    if missing:
        print("\n  MISSING FILES:\n")
        for f in missing:
            print(f"    ✗  {f}")
        print()
    else:
        print("\n  ✓  All required files present.\n")
        print("  Next steps:")
        print("    pytest tests/ -v --ignore=tests/test_integration.py")
        print("    pytest tests/test_integration.py -v -s")
        print()


if __name__ == "__main__":
    main()