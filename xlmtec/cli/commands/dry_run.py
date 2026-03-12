"""
xlmtec.cli.commands.dry_run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dry-run logic for the train command.

Validates config structure and prints a summary of what would happen,
without loading any model or touching the filesystem.
"""

from __future__ import annotations

from pathlib import Path

from xlmtec.cli.ux import console, print_dry_run_table, print_error, print_success

try:
    from xlmtec.core.config import PipelineConfig
except ImportError:
    PipelineConfig = None  # type: ignore[assignment,misc]


def execute_dry_run(config_path: Path) -> int:
    """Validate config and print training plan. Returns exit code (0/1)."""
    try:
        import yaml
    except ImportError as exc:
        print_error("Missing dependency", str(exc))
        return 1

    # ── Load ──────────────────────────────────────────────────────────────
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, yaml.YAMLError) as exc:
        print_error("Config error", str(exc))
        return 1

    if not isinstance(raw, dict):
        print_error("Config error", "Config must be a YAML mapping.")
        return 1

    # ── Structural validation only (skip file-existence checks) ───────────
    if PipelineConfig is None:
        print_error("Missing dependency", "xlmtec.core.config not found.")
        return 1
    try:
        PipelineConfig.model_construct(**raw)
    except Exception as exc:
        print_error("Config error", str(exc))
        return 1

    # ── Print plan ────────────────────────────────────────────────────────
    training = raw.get("training", {}) or {}
    model = raw.get("model", {}) or {}
    dataset = raw.get("dataset", {}) or {}
    lora = raw.get("lora", {}) or {}

    rows = [
        ("Model", model.get("name", "—")),
        ("Method", raw.get("method", "lora")),
        ("Dataset source", dataset.get("source", "—")),
        ("Dataset path", dataset.get("path", "—")),
        ("Epochs", str(training.get("num_epochs", 3))),
        ("Batch size", str(training.get("batch_size", 4))),
        ("Learning rate", str(training.get("learning_rate", 2e-4))),
        ("Output dir", training.get("output_dir", "output/")),
    ]
    if lora:
        rows += [
            ("LoRA r", str(lora.get("r", "—"))),
            ("LoRA alpha", str(lora.get("alpha", "—"))),
        ]

    print_dry_run_table(rows)
    print_success("Dry run complete", "Config is valid. No model was loaded.")
    console.print("[dim]Remove --dry-run to start training.[/dim]\n")
    return 0
