"""
xlmtec.cli.commands.predict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CLI command: xlmtec predict

Registered in main.py as:
    app.command("predict")(predict)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from xlmtec.cli.ux import console, print_error, print_success

try:
    from xlmtec.inference.data_loader import DataLoader
    from xlmtec.inference.predictor import BatchPredictor, PredictConfig
except ImportError:
    DataLoader = None  # type: ignore[assignment,misc]
    BatchPredictor = None  # type: ignore[assignment,misc]
    PredictConfig = None  # type: ignore[assignment,misc]


def run_predict(
    model_dir: Path,
    data_path: Path,
    output_path: Path,
    output_format: str,
    text_column: Optional[str],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
    dry_run: bool,
) -> int:
    """Core inference logic. Returns exit code 0/1. Separated for testability."""
    from rich.panel import Panel

    if DataLoader is None:
        print_error("Missing dependency", "xlmtec.inference not found.")
        return 1

    _VALID_FORMATS = {"jsonl", "csv"}
    if output_format not in _VALID_FORMATS:
        print_error(
            "Invalid format",
            f"{output_format!r} is not supported. Use: {', '.join(sorted(_VALID_FORMATS))}",
        )
        return 1

    if not model_dir.exists():
        print_error("Model not found", f"Directory does not exist: {model_dir}")
        return 1

    if not data_path.exists():
        print_error("Data not found", f"File does not exist: {data_path}")
        return 1

    try:
        loader = DataLoader(data_path, text_column=text_column)
        records = loader.load()
    except (FileNotFoundError, ValueError) as exc:
        print_error("Data load failed", str(exc))
        return 1

    lines = [
        f"[bold]Model:[/bold]      {model_dir}",
        f"[bold]Input:[/bold]      {data_path}  ({len(records)} records)",
        f"[bold]Output:[/bold]     {output_path}  ({output_format})",
        f"[bold]Batch size:[/bold] {batch_size}",
        f"[bold]Max tokens:[/bold] {max_new_tokens}",
        f"[bold]Device:[/bold]     {device}",
    ]
    if dry_run:
        lines.append("\n[yellow]Dry run — model will not be loaded.[/yellow]")

    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title="[bold cyan]Batch Inference Plan[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    if dry_run:
        print_success(
            "Dry run complete",
            f"Input valid — {len(records)} records ready. Remove --dry-run to run inference.",
        )
        console.print()
        return 0

    try:
        cfg = PredictConfig(
            model_dir=model_dir,
            data_path=data_path,
            output_path=output_path,
            output_format=output_format,
            text_column=text_column,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        result = BatchPredictor().predict(cfg)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print_error("Inference failed", str(exc))
        return 1

    print_success(
        "Inference complete",
        f"{result.total_records} predictions saved to [bold]{result.output_path}[/bold]",
    )
    console.print()
    return 0


def predict(
    model_dir: Path = typer.Argument(..., help="Fine-tuned model directory (e.g. output/run1)."),
    data: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="Input file path (.jsonl or .csv).",
    ),
    output: Path = typer.Option(
        Path("predictions.jsonl"),
        "--output",
        "-o",
        help="Output file path for predictions.",
    ),
    fmt: str = typer.Option(
        "jsonl",
        "--format",
        "-f",
        help="Output format: jsonl or csv.",
    ),
    text_column: Optional[str] = typer.Option(
        None,
        "--text-column",
        "-t",
        help="Column name containing input text. Auto-detected if not set.",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        "-b",
        help="Number of inputs to process at once.",
    ),
    max_new_tokens: int = typer.Option(
        128,
        "--max-new-tokens",
        help="Maximum tokens to generate per input.",
    ),
    temperature: float = typer.Option(
        1.0,
        "--temperature",
        help="Sampling temperature (1.0 = greedy).",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device to run on: auto, cpu, cuda.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate input and show plan without loading the model.",
    ),
) -> None:
    """Run batch predictions on a dataset using a fine-tuned model.

    Examples:\n
        xlmtec predict output/run1 --data inputs.jsonl\n
        xlmtec predict output/run1 --data inputs.csv --format csv\n
        xlmtec predict output/run1 --data inputs.jsonl --dry-run
    """
    raise typer.Exit(
        run_predict(
            model_dir=model_dir,
            data_path=data,
            output_path=output,
            output_format=fmt,
            text_column=text_column,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            dry_run=dry_run,
        )
    )
