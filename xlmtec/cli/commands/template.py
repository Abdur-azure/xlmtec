"""
xlmtec.cli.commands.template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CLI commands: xlmtec template list / show / use
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from xlmtec.cli.ux import print_error, print_success

try:
    from xlmtec.templates.registry import get_template, list_templates
except ImportError:
    get_template = None  # type: ignore[assignment]
    list_templates = None  # type: ignore[assignment]

console = Console()
app = typer.Typer(help="Built-in starter configs for common tasks.")


@app.command("list")
def list_cmd() -> None:
    """List all available config templates.

    Examples:\n
        xlmtec template list
    """
    templates = list_templates()

    table = Table(
        title="[bold cyan]xlmtec Built-in Templates[/bold cyan]",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
    )
    table.add_column("Name", style="bold", min_width=16)
    table.add_column("Task", style="yellow", min_width=22)
    table.add_column("Method", style="green", min_width=12)
    table.add_column("Base model", style="dim")
    table.add_column("Tags", style="dim")

    for t in templates:
        table.add_row(
            t.name,
            t.task,
            t.method,
            t.base_model,
            ", ".join(t.tags),
        )

    console.print()
    console.print(table)
    console.print(
        "\n[dim]Use [bold]xlmtec template show <name>[/bold] to preview a config.[/dim]\n"
    )


@app.command("show")
def show(
    name: str = typer.Argument(..., help="Template name e.g. sentiment"),
) -> None:
    """Preview a template config without saving it.

    Examples:\n
        xlmtec template show sentiment\n
        xlmtec template show summarisation
    """
    try:
        template = get_template(name)
    except ValueError as exc:
        print_error("Template not found", str(exc))
        raise typer.Exit(1)

    console.print()
    console.print(
        Panel(
            f"[bold]{template.description}[/bold]\n\n"
            f"[bold]Task:[/bold]        {template.task}\n"
            f"[bold]Method:[/bold]      {template.method}\n"
            f"[bold]Base model:[/bold]  {template.base_model}\n"
            f"[bold]Tags:[/bold]        {', '.join(template.tags)}",
            title=f"[bold cyan]Template: {template.name}[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    console.print("\n[bold]Config:[/bold]")
    console.print(Syntax(template.to_yaml(), "yaml", theme="monokai", line_numbers=False))
    console.print(
        f"[dim]Run [bold]xlmtec template use {name} --output config.yaml[/bold] to save this config.[/dim]\n"
    )


@app.command("use")
def use(
    name: str = typer.Argument(..., help="Template name e.g. sentiment"),
    output: Path = typer.Option(
        Path("config.yaml"),
        "--output",
        "-o",
        help="Path to save the generated config.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Override the base model name.",
    ),
    data_path: Optional[str] = typer.Option(
        None,
        "--data",
        "-d",
        help="Override the dataset path.",
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        "-e",
        help="Override number of training epochs.",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Override the training output directory.",
    ),
) -> None:
    """Generate a config file from a template.

    Examples:\n
        xlmtec template use sentiment --output config.yaml\n
        xlmtec template use summarisation --model facebook/bart-base --output config.yaml\n
        xlmtec template use classification --data data/mytrain.jsonl --epochs 10
    """
    try:
        template = get_template(name)
    except ValueError as exc:
        print_error("Template not found", str(exc))
        raise typer.Exit(1)

    # Build overrides from CLI flags
    overrides: dict = {}
    if model:
        overrides["model"] = {"name": model}
    if data_path:
        overrides["dataset"] = {"path": data_path}
    if epochs:
        overrides.setdefault("training", {})["num_epochs"] = epochs
    if output_dir:
        overrides.setdefault("training", {})["output_dir"] = output_dir

    yaml_str = template.to_yaml(overrides if overrides else None)

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml_str, encoding="utf-8")

    print_success("Config saved", f"Template [bold]{name}[/bold] saved to [bold]{output}[/bold]")
    console.print("\n[dim]Next steps:[/dim]")
    console.print(f"  [cyan]xlmtec config validate {output}[/cyan]")
    console.print(f"  [cyan]xlmtec train --config {output} --dry-run[/cyan]")
    console.print(f"  [cyan]xlmtec train --config {output}[/cyan]\n")
