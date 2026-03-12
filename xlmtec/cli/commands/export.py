"""
xlmtec.cli.commands.export
~~~~~~~~~~~~~~~~~~~~~~~~~~~
CLI command: xlmtec export

Registered in main.py as:
    app.command("export")(export)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel

from xlmtec.cli.ux import console, print_error, print_success

try:
    from xlmtec.export.exporter import ModelExporter
    from xlmtec.export.formats import FORMAT_META, ExportFormat
except ImportError:
    ExportFormat = None  # type: ignore[assignment,misc]
    ModelExporter = None  # type: ignore[assignment,misc]


def export_model(
    model_dir: Path,
    output_dir: Path,
    fmt: str,
    quantize: Optional[str],
    llama_cpp_dir: Optional[Path],
    dry_run: bool,
) -> int:
    """Core export logic. Returns exit code 0/1. Separated for testability."""
    try:
        export_fmt = ExportFormat.from_str(fmt)
    except ValueError as exc:
        print_error("Invalid format", str(exc))
        return 1

    if not model_dir.exists():
        print_error("Model not found", f"Directory does not exist: {model_dir}")
        return 1

    meta = FORMAT_META[export_fmt]

    if quantize and meta.quantize_options and quantize not in meta.quantize_options:
        print_error(
            "Invalid quantise type",
            f"{quantize!r} is not valid for {fmt}.\nOptions: {', '.join(meta.quantize_options)}",
        )
        return 1

    lines = [
        f"[bold]Source:[/bold]   {model_dir}",
        f"[bold]Output:[/bold]   {output_dir}",
        f"[bold]Format:[/bold]   {meta.name}  —  {meta.description}",
    ]
    if quantize:
        lines.append(f"[bold]Quantise:[/bold] {quantize}")
    if dry_run:
        lines.append("\n[yellow]Dry run — no files will be written.[/yellow]")

    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title="[bold cyan]Export Plan[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    if dry_run:
        print_success("Dry run complete", "Options are valid. Remove --dry-run to export.")
        console.print()
        return 0

    try:
        exporter = ModelExporter()
        result = exporter.export(
            model_dir=model_dir,
            output_dir=output_dir,
            fmt=export_fmt,
            quantize=quantize,
            llama_cpp_dir=llama_cpp_dir,
            dry_run=False,
        )
    except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print_error("Export failed", str(exc))
        return 1

    size_str = f"{result.file_size_mb:.1f} MB" if result.file_size_mb else "unknown size"
    print_success(
        "Export complete",
        f"[bold]{result.output_path.name}[/bold]  ({size_str})\n"
        f"Saved to: {result.output_path.parent}",
    )
    console.print()
    return 0


def export(
    model_dir: Path = typer.Argument(..., help="Trained model directory (e.g. output/run1)."),
    fmt: str = typer.Option(
        "safetensors",
        "--format",
        "-f",
        help="Export format: safetensors, onnx, gguf.",
    ),
    output: Path = typer.Option(
        Path("exported/"),
        "--output",
        "-o",
        help="Output directory for exported files.",
    ),
    quantize: Optional[str] = typer.Option(
        None,
        "--quantize",
        "-q",
        help="Quantisation type (format-specific). ONNX: fp16/int8. GGUF: q4_0/q8_0/etc.",
    ),
    llama_cpp_dir: Optional[Path] = typer.Option(
        None,
        "--llama-cpp-dir",
        help="Path to llama.cpp repo (GGUF only).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate options without writing files.",
    ),
) -> None:
    """Export a fine-tuned model to a deployment format.

    Examples:\n
        xlmtec export output/run1 --format safetensors\n
        xlmtec export output/run1 --format onnx --quantize fp16\n
        xlmtec export output/run1 --format gguf --quantize q4_0\n
        xlmtec export output/run1 --format safetensors --dry-run
    """
    raise typer.Exit(
        export_model(
            model_dir=model_dir,
            output_dir=output,
            fmt=fmt,
            quantize=quantize,
            llama_cpp_dir=llama_cpp_dir,
            dry_run=dry_run,
        )
    )
