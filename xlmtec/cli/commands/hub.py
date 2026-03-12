"""
xlmtec.cli.commands.hub
~~~~~~~~~~~~~~~~~~~~~~~~
CLI commands: xlmtec hub search / info / trending
"""

from __future__ import annotations

from typing import Optional

import typer

from xlmtec.cli.ux import print_error, task_progress

app = typer.Typer(help="Browse and search HuggingFace models.")

# Module-level so tests can patch it
try:
    from xlmtec.hub.client import HubClient
except ImportError:
    HubClient = None  # type: ignore[assignment,misc]

try:
    from xlmtec.hub.formatter import (
        print_model_info,
        print_search_results,
        print_trending,
    )
except ImportError:
    print_model_info = None  # type: ignore[assignment]
    print_search_results = None  # type: ignore[assignment]
    print_trending = None  # type: ignore[assignment]


@app.command("search")
def search(
    query: str = typer.Argument(..., help='Search query e.g. "bert" or "llama"'),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        "-t",
        help="Filter by task e.g. text-classification, text-generation",
    ),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results (max 100)"),
    sort: str = typer.Option(
        "downloads",
        "--sort",
        "-s",
        help="Sort by: downloads | likes | lastModified",
    ),
) -> None:
    """Search HuggingFace models by name or keyword.

    Examples:\n
        xlmtec hub search bert\n
        xlmtec hub search llama --task text-generation --limit 5\n
        xlmtec hub search gpt --sort likes
    """
    if HubClient is None:
        print_error("Missing dependency", "huggingface-hub is not installed.")
        raise typer.Exit(1)

    try:
        with task_progress(f"Searching for {query!r}..."):
            client = HubClient()
            results = client.search(query, task=task, limit=limit, sort=sort)
        print_search_results(results, query)
    except Exception as exc:
        print_error("Search failed", str(exc))
        raise typer.Exit(1)


@app.command("info")
def info(
    model_id: str = typer.Argument(..., help='Model ID e.g. "google/bert-base-uncased"'),
) -> None:
    """Show detailed info for a specific model.

    Examples:\n
        xlmtec hub info google/bert-base-uncased\n
        xlmtec hub info mistralai/Mistral-7B-v0.1
    """
    if HubClient is None:
        print_error("Missing dependency", "huggingface-hub is not installed.")
        raise typer.Exit(1)

    try:
        with task_progress(f"Fetching info for {model_id!r}..."):
            client = HubClient()
            detail = client.info(model_id)
        print_model_info(detail)
    except ValueError as exc:
        print_error("Not found", str(exc))
        raise typer.Exit(1)
    except Exception as exc:
        print_error("Request failed", str(exc))
        raise typer.Exit(1)


@app.command("trending")
def trending(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results (max 100)"),
) -> None:
    """Show the top trending models on HuggingFace.

    Examples:\n
        xlmtec hub trending\n
        xlmtec hub trending --limit 20
    """
    if HubClient is None:
        print_error("Missing dependency", "huggingface-hub is not installed.")
        raise typer.Exit(1)

    try:
        with task_progress("Fetching trending models..."):
            client = HubClient()
            results = client.trending(limit=limit)
        print_trending(results)
    except Exception as exc:
        print_error("Request failed", str(exc))
        raise typer.Exit(1)
