"""
xlmtec.export.backends.onnx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Export a HuggingFace model to ONNX format using optimum.

Requires: pip install xlmtec[onnx]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class OnnxResult:
    output_path: Path
    file_size_mb: float
    quantized: bool
    quantize_type: str


def export_onnx(
    model_dir: Path,
    output_dir: Path,
    quantize: str | None = None,
    dry_run: bool = False,
) -> OnnxResult:
    """Export a model to ONNX format using optimum.

    Args:
        model_dir:  Source model directory.
        output_dir: Destination for .onnx file(s).
        quantize:   Optional quantisation type: 'fp16' or 'int8'.
        dry_run:    Validate only, do not write files.

    Returns:
        OnnxResult with path, size, and quantisation info.

    Raises:
        FileNotFoundError: If model_dir does not exist.
        ImportError: If optimum is not installed.
        ValueError: If quantize type is invalid.
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    VALID_QUANTIZE = {"fp16", "int8"}
    if quantize and quantize not in VALID_QUANTIZE:
        raise ValueError(
            f"Invalid ONNX quantise type {quantize!r}. Choose from: {', '.join(VALID_QUANTIZE)}"
        )

    if dry_run:
        return OnnxResult(
            output_path=output_dir / "model.onnx",
            file_size_mb=0.0,
            quantized=bool(quantize),
            quantize_type=quantize or "",
        )

    try:
        from optimum.exporters.onnx import main_export
    except ImportError as exc:
        raise ImportError(
            "ONNX export requires optimum.\n"
            "Install with: pip install xlmtec[onnx]\n"
            f"Original error: {exc}"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    main_export(
        model_name_or_path=str(model_dir),
        output=output_dir,
        task="text-generation",
        fp16=(quantize == "fp16"),
        int8=(quantize == "int8"),
    )

    onnx_files = list(output_dir.glob("*.onnx"))
    out_file = onnx_files[0] if onnx_files else output_dir / "model.onnx"
    size_mb = out_file.stat().st_size / (1024 * 1024) if out_file.exists() else 0.0

    return OnnxResult(
        output_path=out_file,
        file_size_mb=size_mb,
        quantized=bool(quantize),
        quantize_type=quantize or "",
    )