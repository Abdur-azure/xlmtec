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
    "lmtool/__init__.py",

    # ── Core ─────────────────────────────────────────────────────────────
    "lmtool/core/__init__.py",
    "lmtool/core/exceptions.py",
    "lmtool/core/types.py",
    "lmtool/core/config.py",
    "lmtool/core/CONTEXT.md",

    # ── Utils ─────────────────────────────────────────────────────────────
    "lmtool/utils/__init__.py",
    "lmtool/utils/logging.py",

    # ── Models ────────────────────────────────────────────────────────────
    "lmtool/models/__init__.py",
    "lmtool/models/loader.py",

    # ── Trainers ──────────────────────────────────────────────────────────
    "lmtool/trainers/__init__.py",
    "lmtool/trainers/base.py",
    "lmtool/trainers/lora_trainer.py",
    "lmtool/trainers/qlora_trainer.py",
    "lmtool/trainers/full_trainer.py",
    "lmtool/trainers/instruction_trainer.py",
    "lmtool/trainers/dpo_trainer.py",
    "lmtool/trainers/response_distillation_trainer.py",
    "lmtool/trainers/feature_distillation_trainer.py",
    "lmtool/trainers/structured_pruner.py",
    "lmtool/trainers/wanda_pruner.py",
    "lmtool/trainers/factory.py",
    "lmtool/trainers/CONTEXT.md",

    # ── TUI ───────────────────────────────────────────────────────────────
    "lmtool/tui/__init__.py",
    "lmtool/tui/app.py",
    "lmtool/tui/app.css",
    "lmtool/tui/screens/__init__.py",
    "lmtool/tui/screens/home.py",
    "lmtool/tui/screens/train.py",
    "lmtool/tui/screens/recommend.py",
    "lmtool/tui/screens/evaluate.py",
    "lmtool/tui/screens/benchmark.py",
    "lmtool/tui/screens/merge.py",
    "lmtool/tui/screens/upload.py",
    "lmtool/tui/screens/running.py",
    "lmtool/tui/screens/result.py",
    "lmtool/tui/widgets/__init__.py",
    "lmtool/tui/widgets/command_card.py",
    "lmtool/tui/widgets/log_panel.py",
    "lmtool/tui/widgets/metric_table.py",

    # ── Data ──────────────────────────────────────────────────────────────
    "lmtool/data/__init__.py",
    "lmtool/data/pipeline.py",
    "lmtool/data/CONTEXT.md",

    # ── Evaluation ────────────────────────────────────────────────────────
    "lmtool/evaluation/__init__.py",
    "lmtool/evaluation/metrics.py",
    "lmtool/evaluation/benchmarker.py",
    "lmtool/evaluation/CONTEXT.md",

    # ── CLI ───────────────────────────────────────────────────────────────
    "lmtool/cli/__init__.py",
    "lmtool/cli/main.py",
    "lmtool/cli/CONTEXT.md",

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
