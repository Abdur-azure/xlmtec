"""
xlmtec.cli.ux
~~~~~~~~~~~~~~
Shared UX utilities for the xlmtec CLI.

Centralises: error panels, success panels, progress bars, version string.
Import from here — never instantiate Console() or Progress() in command files.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Generator

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def get_version() -> str:
    """Return the installed xlmtec version string."""
    try:
        return version("xlmtec")
    except PackageNotFoundError:
        return "0.0.0-dev"


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def print_error(title: str, message: str) -> None:
    """Print a formatted error panel to stderr."""
    err_console.print(
        Panel(
            f"[bold]{message}[/bold]",
            title=f"[red]✗ {title}[/red]",
            border_style="red",
            padding=(0, 1),
        )
    )


def print_success(title: str, message: str) -> None:
    """Print a formatted success panel."""
    console.print(
        Panel(
            f"[bold]{message}[/bold]",
            title=f"[green]✓ {title}[/green]",
            border_style="green",
            padding=(0, 1),
        )
    )


def print_warning(message: str) -> None:
    """Print a yellow warning line."""
    console.print(f"[yellow]⚠[/yellow]  {message}")


def print_dry_run_table(
    rows: list[tuple[str, str]], title: str = "Dry Run — Training Plan"
) -> None:
    """Print a table summarising what a dry run would do."""
    table = Table(
        title=title,
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    for key, val in rows:
        table.add_row(key, str(val))
    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


@contextmanager
def task_progress(description: str) -> Generator[None, None, None]:
    """Context manager that shows a spinner while work is in progress.

    Usage::

        with task_progress("Loading model..."):
            model = load_model(...)
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description, total=None)
        yield


def make_training_progress() -> Progress:
    """Return a Rich Progress bar suitable for training loops.

    Usage::

        progress = make_training_progress()
        with progress:
            task = progress.add_task("Training", total=num_steps)
            for step in steps:
                ...
                progress.advance(task)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("step {task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
