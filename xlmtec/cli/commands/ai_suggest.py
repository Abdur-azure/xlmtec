"""
xlmtec.cli.commands.ai_suggest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CLI command: xlmtec ai-suggest
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()
app = typer.Typer()


@app.command("ai-suggest")
def ai_suggest(
    task: str = typer.Argument(
        ...,
        help='Plain-English description. E.g. "fine-tune GPT-2 for sentiment analysis"',
    ),
    provider: str = typer.Option(
        "claude",
        "--provider",
        "-p",
        help="AI provider to use: claude | gemini | codex",
    ),
    api_key: str = typer.Option(
        None,
        "--api-key",
        envvar=["ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"],
        help="API key (falls back to provider env variable)",
    ),
    save: str = typer.Option(
        None,
        "--save",
        "-s",
        help="Save the generated YAML to this path (e.g. config.yaml)",
    ),
) -> None:
    """Generate a ready-to-run xlmtec config from a plain-English task description.

    Examples:\n
        xlmtec ai-suggest "fine-tune GPT-2 for sentiment analysis"\n
        xlmtec ai-suggest "qlora on llama-3 for code generation" --provider gemini\n
        xlmtec ai-suggest "instruction tune for customer support" --save config.yaml
    """
    try:
        from xlmtec.integrations import get_provider
    except ImportError as exc:
        console.print(f"[red]Import error:[/red] {exc}")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]xlmtec ai-suggest[/bold cyan] — provider: [yellow]{provider}[/yellow]\n")

    try:
        integration = get_provider(provider, api_key=api_key)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    with console.status(f"Asking {provider} for suggestions...", spinner="dots"):
        try:
            result = integration.suggest(task)
        except (ImportError, RuntimeError) as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)

    # ── Display results ────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold green]Method:[/bold green] {result.method}\n\n"
        f"[bold]Why:[/bold] {result.explanation}",
        title="Recommendation",
        border_style="green",
    ))

    console.print("\n[bold]Generated config:[/bold]")
    console.print(Syntax(result.yaml_config, "yaml", theme="monokai", line_numbers=False))

    console.print(Panel(
        f"[bold cyan]{result.command}[/bold cyan]",
        title="Run this",
        border_style="cyan",
    ))

    if save:
        from pathlib import Path
        Path(save).write_text(result.yaml_config, encoding="utf-8")
        console.print(f"\n[green]✓[/green] Config saved to [bold]{save}[/bold]")