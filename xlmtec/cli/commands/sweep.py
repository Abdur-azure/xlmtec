"""
xlmtec.cli.commands.sweep
~~~~~~~~~~~~~~~~~~~~~~~~~~
CLI command: xlmtec sweep

Runs a hyperparameter sweep over a base PipelineConfig using Optuna.

Registered in main.py as:
    app.command("sweep")(sweep)

Usage:
    xlmtec sweep examples/configs/sweep_lora_gpt2.yaml --trials 20 --dry-run
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from xlmtec.cli.ux import console, print_error, print_success


def run_sweep(
    config_path: Path,
    n_trials: Optional[int],
    dry_run: bool,
) -> int:
    """Core sweep logic. Returns exit code 0/1. Separated for testability."""
    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    if not config_path.exists():
        print_error("Config not found", str(config_path))
        return 1

    # Parse YAML
    try:
        import yaml
    except ImportError:
        print_error("Missing dependency", "PyYAML is required: pip install pyyaml")
        return 1

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        print_error("YAML parse error", str(exc))
        return 1

    if not isinstance(raw, dict):
        print_error("Invalid config", "Config file must be a YAML mapping.")
        return 1

    # Extract and validate sweep section
    sweep_raw = raw.pop("sweep", None)
    if sweep_raw is None:
        print_error(
            "Missing 'sweep' section",
            "Add a 'sweep:' block to your config. See examples/configs/sweep_lora_gpt2.yaml",
        )
        return 1

    try:
        from xlmtec.sweep.config import SweepConfig

        sweep_cfg = SweepConfig.from_dict(sweep_raw)
    except (ValueError, KeyError) as exc:
        print_error("Invalid sweep config", str(exc))
        return 1

    # Override n_trials from flag
    if n_trials is not None:
        if n_trials < 1:
            print_error("Invalid --trials", f"Must be >= 1, got {n_trials}")
            return 1
        sweep_cfg = SweepConfig(
            n_trials=n_trials,
            metric=sweep_cfg.metric,
            direction=sweep_cfg.direction,
            output_dir=sweep_cfg.output_dir,
            params=sweep_cfg.params,
            sampler=sweep_cfg.sampler,
            timeout=sweep_cfg.timeout,
        )

    # Validate base config structure
    try:
        from xlmtec.core.config import PipelineConfig

        PipelineConfig(**raw)
    except Exception as exc:
        print_error("Invalid base config", str(exc))
        return 1

    # Print plan
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="cyan")
    table.add_column("Type")
    table.add_column("Space")

    for name, spec in sweep_cfg.params.items():
        if spec.type == "categorical":
            space = f"choices={spec.choices}"
        elif spec.log:
            space = f"log-uniform [{spec.low}, {spec.high}]"
        else:
            space = f"[{spec.low}, {spec.high}]"
        table.add_row(name, spec.type, space)

    console.print()
    console.print(
        Panel(
            f"[bold]Config:[/bold]    {config_path}\n"
            f"[bold]Trials:[/bold]    {sweep_cfg.n_trials}\n"
            f"[bold]Metric:[/bold]    {sweep_cfg.metric}  ({sweep_cfg.direction})\n"
            f"[bold]Sampler:[/bold]   {sweep_cfg.sampler}\n"
            f"[bold]Output:[/bold]    {sweep_cfg.output_dir}",
            title="[bold cyan]Sweep Plan[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print(table)

    if dry_run:
        print_success(
            "Dry run complete",
            f"Config valid — {len(sweep_cfg.params)} param(s), "
            f"{sweep_cfg.n_trials} trials planned. Remove --dry-run to start.",
        )
        console.print()
        return 0

    # Run sweep
    try:
        from xlmtec.sweep.runner import SweepRunner
    except ImportError as exc:
        print_error("Missing dependency", str(exc))
        return 1

    try:
        runner = SweepRunner(
            base_config_dict=raw,
            sweep_config=sweep_cfg,
        )
        result = runner.run()
    except ImportError as exc:
        print_error("optuna not installed", f"{exc}\nInstall with: pip install xlmtec[sweep]")
        return 1
    except Exception as exc:
        print_error("Sweep failed", str(exc))
        return 1

    # Print results
    console.print()
    rows = [
        ("Best trial", str(result.best_trial)),
        ("Best metric", f"{result.best_metric:.6f}"),
        ("Completed", str(result.n_completed)),
        ("Failed", str(result.n_failed)),
    ]
    for k, v in result.best_params.items():
        rows.append((f"  {k}", str(v)))

    console.print(
        Panel(
            "\n".join(f"[bold]{k}:[/bold]  {v}" for k, v in rows),
            title="[bold green]Sweep Results[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    print_success(
        "Sweep complete",
        f"Best {sweep_cfg.metric} = {result.best_metric:.6f} (trial {result.best_trial})",
    )
    console.print()
    return 0


def sweep(
    config: Path = typer.Argument(
        ...,
        help="Path to a sweep YAML config (base PipelineConfig + 'sweep:' section).",
    ),
    trials: Optional[int] = typer.Option(
        None,
        "--trials",
        "-t",
        help="Number of trials. Overrides the value in the config file.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and show the sweep plan without running any trials.",
    ),
) -> None:
    """Run a hyperparameter sweep using Optuna.

    The config file must include both a standard PipelineConfig and a
    ``sweep:`` section defining the param search space.

    Examples:\n
        xlmtec sweep examples/configs/sweep_lora_gpt2.yaml --dry-run\n
        xlmtec sweep examples/configs/sweep_lora_gpt2.yaml --trials 20\n
        xlmtec sweep my_sweep.yaml --trials 50 --dry-run
    """
    raise typer.Exit(run_sweep(config_path=config, n_trials=trials, dry_run=dry_run))
