"""
audit_repo.py — run this from your repo root to find missing files.

Usage:
    python audit_repo.py

Prints every file that should exist but doesn't.
Copy the missing ones from the outputs/ folder provided.
"""

from pathlib import Path

REQUIRED_FILES = [
    # Root
    "conftest.py",
    "pyproject.toml",
    "audit_repo.py",
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    ".github/workflows/ci.yml",
    ".gitignore",

    # Main package
    "finetune_cli/__init__.py",

    # Core
    "finetune_cli/core/__init__.py",
    "finetune_cli/core/exceptions.py",
    "finetune_cli/core/types.py",
    "finetune_cli/core/config.py",

    # Utils
    "finetune_cli/utils/__init__.py",
    "finetune_cli/utils/logging.py",

    # Models
    "finetune_cli/models/__init__.py",
    "finetune_cli/models/loader.py",

    # Trainers
    "finetune_cli/trainers/__init__.py",
    "finetune_cli/trainers/base.py",
    "finetune_cli/trainers/lora_trainer.py",
    "finetune_cli/trainers/qlora_trainer.py",
    "finetune_cli/trainers/full_trainer.py",
    "finetune_cli/trainers/instruction_trainer.py",
    "finetune_cli/trainers/dpo_trainer.py",
    "finetune_cli/trainers/response_distillation_trainer.py",
    "finetune_cli/trainers/feature_distillation_trainer.py",
    "finetune_cli/trainers/structured_pruner.py",
    "finetune_cli/trainers/wanda_pruner.py",
    "finetune_cli/trainers/factory.py",

    # TUI    
    "finetune_cli/tui/__init__.py",
    "finetune_cli/tui/app.py",
    "finetune_cli/tui/screens/__init__.py",
    "finetune_cli/tui/screens/home.py",
    "finetune_cli/tui/widgets/__init__.py",
    "finetune_cli/tui/widgets/command_card.py",
    "finetune_cli/tui/widgets/log_panel.py",
    "finetune_cli/tui/widgets/metric_table.py",
    "finetune_cli/tui/screens/running.py",
    "finetune_cli/tui/screens/result.py",
    "finetune_cli/tui/screens/train.py",
    "finetune_cli/tui/screens/recommend.py",
    "finetune_cli/tui/screens/evaluate.py",
    "finetune_cli/tui/screens/benchmark.py",
    "finetune_cli/tui/screens/merge.py",
    "finetune_cli/tui/screens/upload.py",
    "finetune_cli/tui/app.css",


    # Data
    "finetune_cli/data/__init__.py",
    "finetune_cli/data/pipeline.py",

    # Evaluation
    "finetune_cli/evaluation/__init__.py",
    "finetune_cli/evaluation/metrics.py",
    "finetune_cli/evaluation/benchmarker.py",

    # CLI
    "finetune_cli/cli/__init__.py",
    "finetune_cli/cli/main.py",

    # Tests
    "tests/__init__.py",
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
    "tests/test_prune.py",
    "docs/tui.md",
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
    "tests/test_wanda_pruner.py",
    "tests/test_wanda_cli.py",

    # Tasks
    "tasks/todo.md",
    "tasks/lessons.md",
    "tasks/CONTEXT.md",
    "tasks/roadmap.md",

    # Examples
    "examples/configs/lora_gpt2.yaml",
    "examples/configs/qlora_llama.yaml",
    "examples/configs/instruction_tuning.yaml",
    "examples/configs/full_finetuning.yaml",
    "examples/configs/dpo.yaml",
    "examples/configs/response_distillation.yaml",
    "examples/configs/feature_distillation.yaml",       
    "examples/configs/structured_pruning.yaml",
    "examples/configs/wanda.yaml",
    "examples/generate_sample_data.py",

    # Context files
    "CLAUDE.md",
    "finetune_cli/core/CONTEXT.md",
    "finetune_cli/trainers/CONTEXT.md",
    "finetune_cli/data/CONTEXT.md",
    "finetune_cli/evaluation/CONTEXT.md",
    "finetune_cli/cli/CONTEXT.md",
    "tests/CONTEXT.md",
]

root = Path(__file__).parent
missing = [f for f in REQUIRED_FILES if not (root / f).exists()]
present = [f for f in REQUIRED_FILES if (root / f).exists()]

print(f"\n{'='*55}")
print(f"  REPO AUDIT — {root.name}")
print(f"{'='*55}")
print(f"  Present : {len(present)}/{len(REQUIRED_FILES)}")
print(f"  Missing : {len(missing)}/{len(REQUIRED_FILES)}")
print(f"{'='*55}")

if missing:
    print("\n  MISSING FILES (copy these from the output zip):\n")
    for f in missing:
        print(f"    ✗  {f}")
else:
    print("\n  All required files present. Run: pytest tests/ -v\n")

print()