"""
xlmtec.cli.commands.dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CLI commands: xlmtec dashboard compare / show
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from xlmtec.cli.ux import print_error, print_success

try:
    from xlmtec.dashboard.comparator import RunComparator
    from xlmtec.dashboard.reader import RunReader
except ImportError:
    RunReader = None  # type: ignore[assignment,misc]
    RunComparator = None  # type: ignore[assignment,misc]

console = Console()
app = typer.Typer(help="Compare and inspect training runs.")


@app.command("compare")
def compare(
    run_dirs: list[Path] = typer.Argument(..., help="Two or more run directories to compare."),
    export: Optional[Path] = typer.Option(
        None,
        "--export",
        "-e",
        help="Export comparison results to a JSON file.",
    ),
) -> None:
    """Compare multiple training runs side by side.

    Examples:\n
        xlmtec dashboard compare output/run1 output/run2\n
        xlmtec dashboard compare output/run1 output/run2 output/run3\n
        xlmtec dashboard compare output/run1 output/run2 --export results.json
    """
    if len(run_dirs) < 2:
        print_error("Too few runs", "Provide at least 2 run directories to compare.")
        raise typer.Exit(1)

    try:
        result = RunComparator().compare(run_dirs)
    except ValueError as exc:
        print_error("Compare failed", str(exc))
        raise typer.Exit(1)

    # ── Metrics table ─────────────────────────────────────────────────────
    table = Table(
        title="[bold cyan]Run Comparison[/bold cyan]",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold", min_width=24)
    for run in result.runs:
        is_winner = result.winner and run.name == result.winner.name
        label = f"[green]{run.name} ★[/green]" if is_winner else run.name
        table.add_column(label, justify="right", min_width=14)

    def _row(label: str, values: list) -> None:
        table.add_row(label, *[_fmt(v) for v in values])

    _row("Total steps", [r.total_steps for r in result.runs])
    _row("Total epochs", [f"{r.total_epochs:.1f}" for r in result.runs])
    _row("Best metric", [r.best_metric for r in result.runs])
    _row("Best eval loss", [r.best_eval_loss for r in result.runs])
    _row("Final train loss", [r.final_train_loss for r in result.runs])
    _row(
        "Runtime (s)",
        [f"{r.train_runtime_seconds:.0f}" if r.train_runtime_seconds else "—" for r in result.runs],
    )
    _row(
        "Samples/sec",
        [
            f"{r.train_samples_per_second:.1f}" if r.train_samples_per_second else "—"
            for r in result.runs
        ],
    )

    console.print()
    console.print(table)

    # ── Winner ────────────────────────────────────────────────────────────
    if result.winner:
        console.print(
            Panel(
                f"[bold green]{result.winner.name}[/bold green] — {result.winner_reason}",
                title="[bold]Winner[/bold]",
                border_style="green",
                padding=(0, 2),
            )
        )

    # ── Config diff (2 runs only) ─────────────────────────────────────────
    if len(result.runs) == 2:
        diffs = RunComparator().diff_configs(result.runs[0], result.runs[1])
        if diffs:
            diff_table = Table(
                title="[bold]Config differences[/bold]",
                box=box.SIMPLE,
                header_style="bold",
            )
            diff_table.add_column("Field", style="bold")
            diff_table.add_column(result.runs[0].name, style="yellow")
            diff_table.add_column(result.runs[1].name, style="cyan")
            for key, (va, vb) in diffs.items():
                diff_table.add_row(key, str(va), str(vb))
            console.print()
            console.print(diff_table)

    # ── Export ────────────────────────────────────────────────────────────
    if export:
        data = {
            "runs": [
                {
                    "name": r.name,
                    "total_steps": r.total_steps,
                    "total_epochs": r.total_epochs,
                    "best_metric": r.best_metric,
                    "best_eval_loss": r.best_eval_loss,
                    "final_train_loss": r.final_train_loss,
                }
                for r in result.runs
            ],
            "winner": result.winner.name if result.winner else None,
            "winner_reason": result.winner_reason,
        }
        export.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print_success("Exported", f"Results saved to {export}")

    console.print()


@app.command("show")
def show(
    run_dir: Path = typer.Argument(..., help="Run directory to inspect."),
) -> None:
    """Show detailed metrics for a single training run.

    Examples:\n
        xlmtec dashboard show output/run1
    """
    try:
        run = RunReader(run_dir).read()
    except (FileNotFoundError, ValueError) as exc:
        print_error("Read failed", str(exc))
        raise typer.Exit(1)

    # ── Summary panel ─────────────────────────────────────────────────────
    lines = [
        f"[bold]Steps:[/bold]          {run.total_steps}",
        f"[bold]Epochs:[/bold]         {run.total_epochs:.1f}",
        f"[bold]Best metric:[/bold]    {_fmt(run.best_metric)} ({run.best_metric_name or '—'})",
        f"[bold]Best eval loss:[/bold] {_fmt(run.best_eval_loss)}",
        f"[bold]Final train loss:[/bold] {_fmt(run.final_train_loss)}",
        f"[bold]Runtime:[/bold]        {run.train_runtime_seconds:.0f}s"
        if run.train_runtime_seconds
        else "[bold]Runtime:[/bold]        —",
        f"[bold]Samples/sec:[/bold]    {run.train_samples_per_second:.1f}"
        if run.train_samples_per_second
        else "[bold]Samples/sec:[/bold]    —",
    ]
    if run.best_model_checkpoint:
        lines.append(f"[bold]Best checkpoint:[/bold] {Path(run.best_model_checkpoint).name}")

    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold cyan]{run.name}[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # ── Training history table ─────────────────────────────────────────────
    if run.history:
        hist_table = Table(
            title="Training History",
            box=box.SIMPLE,
            header_style="bold",
        )
        hist_table.add_column("Step", justify="right")
        hist_table.add_column("Epoch", justify="right")
        hist_table.add_column("Train loss", justify="right")
        hist_table.add_column("Eval loss", justify="right")
        hist_table.add_column("LR", justify="right")

        # Show max 20 rows
        shown = (
            run.history
            if len(run.history) <= 20
            else run.history[:: max(1, len(run.history) // 20)]
        )
        for h in shown:
            hist_table.add_row(
                str(h.step),
                f"{h.epoch:.2f}",
                _fmt(h.train_loss),
                _fmt(h.eval_loss),
                f"{h.learning_rate:.2e}" if h.learning_rate else "—",
            )
        console.print()
        console.print(hist_table)

    console.print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
