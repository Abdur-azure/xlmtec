"""
xlmtec.cli.commands.report
~~~~~~~~~~~~~~~~~~~~~~~~~~~
CLI command: xlmtec report

Shows recent session logs and crash reports.

Registered in main.py as:
    app.command("report")(report)

Usage:
    xlmtec report                  # show latest crash report
    xlmtec report --last 5         # list last 5 crashes
    xlmtec report --sessions       # list last 10 session files
    xlmtec report --open           # open latest crash in $EDITOR / notepad
    xlmtec report --log-dir PATH   # use a custom log directory
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from xlmtec.cli.ux import console, print_error

_DEFAULT_LOG_DIR = Path.home() / ".xlmtec" / "logs"


# ---------------------------------------------------------------------------
# Core logic (separated for testability)
# ---------------------------------------------------------------------------


def run_report(
    last: int = 1,
    sessions: bool = False,
    open_file: bool = False,
    log_dir: Path = _DEFAULT_LOG_DIR,
) -> int:
    """Print crash/session report. Returns exit code 0/1."""
    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    if not log_dir.exists():
        console.print(
            f"[yellow]No logs found.[/yellow] "
            f"Log directory does not exist yet: {log_dir}\n"
            f"Logs are created automatically when you run any xlmtec command."
        )
        return 0

    # ------------------------------------------------------------------
    # --sessions mode: list session files
    # ------------------------------------------------------------------
    if sessions:
        files = sorted(log_dir.glob("session_*.jsonl"), reverse=True)[: last or 10]
        if not files:
            console.print("[yellow]No session logs found.[/yellow]")
            return 0

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=3)
        table.add_column("File")
        table.add_column("Size")
        table.add_column("Events")

        for i, f in enumerate(files, 1):
            try:
                lines = sum(1 for _ in f.open(encoding="utf-8"))
                size = f"{f.stat().st_size / 1024:.1f} KB"
            except Exception:
                lines, size = 0, "?"
            table.add_row(str(i), f.name, size, str(lines))

        console.print(
            Panel(
                table,
                title="[bold cyan]Session Logs[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            )
        )
        console.print(f"\n[dim]Log directory:[/dim] {log_dir}")
        return 0

    # ------------------------------------------------------------------
    # Default: crash reports
    # ------------------------------------------------------------------
    from xlmtec.utils.crash_report import CrashReporter

    crashes = CrashReporter.list_recent(n=last, log_dir=log_dir)

    if not crashes:
        console.print(
            "[green]✓ No crash reports found.[/green] "
            "The application has not encountered any unhandled errors."
        )
        console.print(f"\n[dim]Log directory:[/dim] {log_dir}")
        return 0

    # List view when last > 1
    if last > 1 or (last == 1 and not open_file):
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold red")
        table.add_column("#", style="dim", width=3)
        table.add_column("Timestamp")
        table.add_column("Command")
        table.add_column("Path")

        for i, cf in enumerate(crashes, 1):
            table.add_row(str(i), cf.timestamp, cf.cmd, cf.path.name)

        console.print(
            Panel(
                table,
                title="[bold red]Crash Reports[/bold red]",
                border_style="red",
                padding=(0, 1),
            )
        )

    # Print latest crash content inline
    latest = crashes[0]
    if not open_file:
        try:
            content = latest.path.read_text(encoding="utf-8")
            # Show first 60 lines to avoid flooding the terminal
            crash_lines: list[str] = content.splitlines()
            preview = "\n".join(crash_lines[:60])
            if len(crash_lines) > 60:
                preview += f"\n\n... ({len(crash_lines) - 60} more lines)"
            console.print()
            console.print(
                Panel(
                    preview,
                    title=f"[bold red]Latest crash — {latest.path.name}[/bold red]",
                    border_style="red",
                    padding=(1, 2),
                )
            )
        except Exception as exc:
            print_error("Could not read crash file", str(exc))
            return 1

    console.print(f"\n[dim]Full file:[/dim] {latest.path}")
    console.print(
        "[dim]Tip:[/dim] Attach the crash file to a GitHub issue → "
        "https://github.com/Abdur-azure/xlmtec/issues"
    )

    # --open: open in editor
    if open_file:
        _open_in_editor(latest.path)

    return 0


def _open_in_editor(path: Path) -> None:
    """Try to open the file in the system's default editor."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            editor = os.environ.get("EDITOR", "xdg-open")
            subprocess.run([editor, str(path)], check=False)
    except Exception as exc:
        console.print(f"[yellow]Could not open editor:[/yellow] {exc}")
        console.print(f"Open manually: {path}")


# ---------------------------------------------------------------------------
# Typer command
# ---------------------------------------------------------------------------


def report(
    last: int = typer.Option(
        1,
        "--last",
        "-n",
        help="Number of crash reports to show (default: 1 = latest only).",
    ),
    sessions: bool = typer.Option(
        False,
        "--sessions",
        help="List session log files instead of crash reports.",
    ),
    open_file: bool = typer.Option(
        False,
        "--open",
        help="Open the latest crash report in your default editor.",
    ),
    log_dir: Optional[Path] = typer.Option(
        None,
        "--log-dir",
        help="Log directory (default: ~/.xlmtec/logs).",
    ),
) -> None:
    """Show recent session logs and crash reports.

    Useful for diagnosing errors and generating bug reports.

    Examples:\n
        xlmtec report                  Show latest crash report\n
        xlmtec report --last 5         List last 5 crash reports\n
        xlmtec report --sessions       List recent session log files\n
        xlmtec report --open           Open latest crash in your editor
    """
    resolved_dir = log_dir or _DEFAULT_LOG_DIR
    raise typer.Exit(
        run_report(
            last=last,
            sessions=sessions,
            open_file=open_file,
            log_dir=resolved_dir,
        )
    )
