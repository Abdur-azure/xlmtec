"""
xlmtec.cli.commands.config_validate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CLI command: xlmtec config validate <path>
"""

from __future__ import annotations

from pathlib import Path

import typer

from xlmtec.cli.ux import print_error, print_success, print_warning

try:
    from xlmtec.core.config import PipelineConfig
except ImportError:
    PipelineConfig = None  # type: ignore[assignment,misc]

app = typer.Typer(help="Config utilities.")


def validate_config(config_path: Path, strict: bool = False) -> int:
    """Core validation logic. Returns exit code 0/1.
    Separated from CLI so it can be tested without typer argument parsing.
    """
    try:
        import yaml
        from pydantic import ValidationError
    except ImportError as exc:
        print_error("Missing dependency", str(exc))
        return 1

    if not config_path.exists():
        print_error("File not found", str(config_path))
        return 1

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        print_error("YAML parse error", str(exc))
        return 1

    if not isinstance(raw, dict):
        print_error("Invalid config", "Config file must be a YAML mapping.")
        return 1

    if PipelineConfig is None:
        print_error("Missing dependency", "xlmtec.core.config not found.")
        return 1

    try:
        PipelineConfig(**raw)
    except ValidationError as exc:
        errors = exc.errors()
        print_error(
            f"Validation failed — {len(errors)} error(s)",
            "\n".join(f"  {' → '.join(str(loc_part) for loc_part in e['loc'])}: {e['msg']}" for e in errors),
        )
        return 1
    except Exception as exc:
        print_error("Unexpected error", str(exc))
        return 1

    warnings = _check_warnings(raw)
    for w in warnings:
        print_warning(w)

    if warnings and strict:
        print_error("Strict mode", f"{len(warnings)} warning(s) treated as errors.")
        return 1

    print_success("Config valid", f"{config_path.name} passed all validation checks.")
    return 0


@app.command("validate")
def validate(
    config_path: Path = typer.Argument(..., help="Path to the YAML config file."),
    strict: bool = typer.Option(False, "--strict", help="Fail on warnings too."),
) -> None:
    """Validate a YAML config file and report all errors.

    Examples:\n
        xlmtec config validate config.yaml\n
        xlmtec config validate config.yaml --strict
    """
    raise typer.Exit(validate_config(config_path, strict=strict))


def _check_warnings(raw: dict) -> list[str]:
    warnings = []
    training = raw.get("training", {}) or {}
    if training.get("num_epochs", 1) > 10:
        warnings.append("num_epochs > 10 — this may take a long time.")
    if training.get("learning_rate", 2e-4) > 1e-2:
        warnings.append("learning_rate > 0.01 — unusually high, may cause instability.")
    if training.get("batch_size", 4) > 64:
        warnings.append("batch_size > 64 — ensure you have enough VRAM.")
    return warnings
