"""
xlmtec.cli.commands.plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~
CLI commands: xlmtec plugin add-template / add-provider / list / remove
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table
from rich import box

from xlmtec.cli.ux import console, print_error, print_success

app = typer.Typer(help="Manage custom template and provider plugins.")


@app.command("add-template")
def add_template(
    name: str = typer.Argument(..., help="Plugin name e.g. 'my_task'"),
    source: Path = typer.Argument(..., help="Path to a YAML config template file."),
) -> None:
    """Register a custom config template from a YAML file.

    Examples:\n
        xlmtec plugin add-template my_task templates/my_task.yaml\n
        xlmtec plugin add-template legal_ner configs/legal_ner.yaml
    """
    try:
        from xlmtec.plugins.store import register_template
        plugin = register_template(name, source)
    except (FileNotFoundError, ValueError) as exc:
        print_error("Registration failed", str(exc))
        raise typer.Exit(1)

    print_success(
        "Template registered",
        f"[bold]{name}[/bold] → {plugin.source}\n"
        f"Run [cyan]xlmtec template list[/cyan] to confirm."
    )
    console.print()


@app.command("add-provider")
def add_provider(
    name: str = typer.Argument(..., help="Provider name e.g. 'my_provider'"),
    source: Path = typer.Argument(..., help="Path to a .py file containing the provider class."),
    class_name: str = typer.Option(
        ..., "--class", "-c",
        help="Name of the AIIntegration subclass in the file.",
    ),
) -> None:
    """Register a custom AI provider from a Python file.

    The class must subclass xlmtec.integrations.base.AIIntegration.

    Examples:\n
        xlmtec plugin add-provider my_llm providers/my_llm.py --class MyLLMIntegration
    """
    try:
        from xlmtec.plugins.store import register_provider
        plugin = register_provider(name, source, class_name)
    except (FileNotFoundError, ValueError) as exc:
        print_error("Registration failed", str(exc))
        raise typer.Exit(1)

    print_success(
        "Provider registered",
        f"[bold]{name}[/bold] ({class_name}) → {plugin.source}\n"
        f"Use with [cyan]xlmtec ai-suggest --provider {name}[/cyan]"
    )
    console.print()


@app.command("list")
def list_plugins() -> None:
    """List all registered plugins.

    Examples:\n
        xlmtec plugin list
    """
    from xlmtec.plugins.store import load_store

    store = load_store()
    console.print()

    # Templates
    t_table = Table(
        title="[bold cyan]Custom Templates[/bold cyan]",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
    )
    t_table.add_column("Name", style="bold", min_width=16)
    t_table.add_column("Source", style="dim")
    t_table.add_column("Registered", style="dim", min_width=20)

    if store.templates:
        for p in store.templates.values():
            t_table.add_row(p.name, p.source, p.registered_at[:19].replace("T", " "))
    else:
        t_table.add_row("[dim]no custom templates[/dim]", "", "")

    console.print(t_table)
    console.print()

    # Providers
    # FIX lines 125-126: renamed loop variable p → prov to avoid mypy type conflict.
    # The template loop above leaves p typed as TemplatePlugin; using the same name
    # for ProviderPlugin causes [assignment] and [attr-defined] errors.
    p_table = Table(
        title="[bold cyan]Custom Providers[/bold cyan]",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
    )
    p_table.add_column("Name", style="bold", min_width=16)
    p_table.add_column("Class", style="yellow", min_width=20)
    p_table.add_column("Source", style="dim")
    p_table.add_column("Registered", style="dim", min_width=20)

    if store.providers:
        for prov in store.providers.values():
            p_table.add_row(prov.name, prov.class_name, prov.source, prov.registered_at[:19].replace("T", " "))
    else:
        p_table.add_row("[dim]no custom providers[/dim]", "", "", "")

    console.print(p_table)
    console.print(
        "[dim]Add templates with [bold]xlmtec plugin add-template[/bold], "
        "providers with [bold]xlmtec plugin add-provider[/bold][/dim]\n"
    )


@app.command("remove")
def remove(
    name: str = typer.Argument(..., help="Plugin name to remove."),
) -> None:
    """Remove a custom template or provider plugin.

    Examples:\n
        xlmtec plugin remove my_task\n
        xlmtec plugin remove my_provider
    """
    from xlmtec.plugins.store import remove_plugin

    removed = remove_plugin(name)
    if removed:
        print_success("Plugin removed", f"[bold]{name}[/bold] has been unregistered.")
    else:
        print_error("Not found", f"No plugin named {name!r} is registered.")
        raise typer.Exit(1)
    console.print()