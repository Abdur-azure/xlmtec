"""
xlmtec.cli.commands.resume
~~~~~~~~~~~~~~~~~~~~~~~~~~~
CLI command: xlmtec resume

Continues interrupted training from the latest (or specified) checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from xlmtec.cli.ux import console, print_dry_run_table, print_error, print_success, task_progress

try:
    from xlmtec.checkpoints.manager import CheckpointManager
except ImportError:
    CheckpointManager = None  # type: ignore[assignment,misc]


def resume_training(
    output_dir: Path,
    config_path: Optional[Path],
    checkpoint: Optional[str],
    additional_epochs: Optional[int],
    dry_run: bool,
) -> int:
    """Core resume logic. Returns exit code 0/1.
    Separated from CLI for testability.
    """
    if CheckpointManager is None:
        print_error("Missing dependency", "xlmtec.checkpoints not found.")
        return 1

    # ── Find checkpoint ──────────────────────────────────────────────────
    manager = CheckpointManager(output_dir)

    try:
        if checkpoint:
            ckpt = manager.get(checkpoint)
        else:
            ckpt = manager.latest()
    except (FileNotFoundError, ValueError) as exc:
        print_error("Checkpoint error", str(exc))
        return 1

    # ── Load config ──────────────────────────────────────────────────────
    config_file = config_path or (output_dir / "config.yaml")
    if not config_file.exists():
        print_error(
            "Config not found",
            f"Expected config at {config_file}\nPass --config to specify a different path.",
        )
        return 1

    # ── Build plan ───────────────────────────────────────────────────────
    rows = [
        ("Checkpoint", str(ckpt.path.name)),
        ("Resumed from step", str(ckpt.step)),
        ("Resumed from epoch", f"{ckpt.epoch:.1f}"),
        ("Config", str(config_file)),
        ("Output dir", str(output_dir)),
    ]
    if additional_epochs:
        rows.append(("Additional epochs", str(additional_epochs)))
    if ckpt.best_metric is not None:
        rows.append(("Best metric so far", f"{ckpt.best_metric:.4f}"))

    print_dry_run_table(rows, title="Resume Plan")

    if dry_run:
        print_success("Dry run complete", "Checkpoint found and config valid. Ready to resume.")
        console.print("[dim]Remove --dry-run to start training.[/dim]\n")
        return 0

    # ── Resume training ──────────────────────────────────────────────────
    try:
        import yaml
        from xlmtec.core.config import PipelineConfig
    except ImportError as exc:
        print_error("Missing dependency", str(exc))
        return 1

    try:
        raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        cfg = PipelineConfig(**raw)
    except Exception as exc:
        print_error("Config error", str(exc))
        return 1

    # Patch additional epochs if requested
    if additional_epochs:
        from dataclasses import replace

        cfg = cfg.model_copy(
            update={"training": replace(cfg.training, num_epochs=additional_epochs)}
        )

    try:
        from xlmtec.trainers import TrainerFactory

        with task_progress(f"Resuming from {ckpt.path.name}..."):
            # TrainerFactory.create returns a trainer; train() accepts resume_from_checkpoint
            trainer = TrainerFactory.create(cfg)

        trainer.train(resume_from_checkpoint=str(ckpt.path))
        print_success("Training complete", f"Resumed from {ckpt.path.name} and finished.")
        return 0

    except Exception as exc:
        print_error("Training error", str(exc))
        return 1


def resume(
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory containing checkpoint folders.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Config YAML path (defaults to <output-dir>/config.yaml).",
    ),
    checkpoint: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        help="Specific checkpoint to resume from e.g. 'checkpoint-500'. Defaults to latest.",
    ),
    additional_epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        "-e",
        help="Override number of training epochs.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show resume plan without starting training.",
    ),
) -> None:
    """Resume training from the latest (or specified) checkpoint.

    Examples:\n
        xlmtec resume output/run1\n
        xlmtec resume output/run1 --dry-run\n
        xlmtec resume output/run1 --checkpoint checkpoint-500\n
        xlmtec resume output/run1 --epochs 5
    """
    raise typer.Exit(
        resume_training(
            output_dir=output_dir,
            config_path=config,
            checkpoint=checkpoint,
            additional_epochs=additional_epochs,
            dry_run=dry_run,
        )
    )
