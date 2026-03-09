"""
xlmtec.export.exporter
~~~~~~~~~~~~~~~~~~~~~~~
ModelExporter — dispatches to format-specific backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xlmtec.export.formats import ExportFormat, FormatMeta, get_format_meta


@dataclass
class ExportResult:
    format: ExportFormat
    output_path: Path
    file_size_mb: float
    dry_run: bool
    extra: dict


class ModelExporter:
    """Dispatch model export to the correct backend."""

    def export(
        self,
        model_dir: Path,
        output_dir: Path,
        fmt: ExportFormat,
        quantize: str | None = None,
        llama_cpp_dir: Path | None = None,
        dry_run: bool = False,
    ) -> ExportResult:
        """Export model to the specified format.

        Args:
            model_dir:      Source model directory (trained output or merged adapter).
            output_dir:     Destination directory for exported files.
            fmt:            Target export format.
            quantize:       Quantisation type (format-specific, optional).
            llama_cpp_dir:  Path to llama.cpp repo (GGUF only).
            dry_run:        Validate only — do not write files.

        Returns:
            ExportResult with output path and metadata.

        Raises:
            FileNotFoundError: If model_dir does not exist.
            ImportError: If required packages are not installed.
            ValueError: If options are invalid for the format.
        """
        meta = get_format_meta(fmt)

        if not dry_run:
            self._check_dependencies(meta)

        if fmt == ExportFormat.SAFETENSORS:
            return self._export_safetensors(model_dir, output_dir, dry_run)
        elif fmt == ExportFormat.ONNX:
            return self._export_onnx(model_dir, output_dir, quantize, dry_run)
        elif fmt == ExportFormat.GGUF:
            return self._export_gguf(model_dir, output_dir, quantize or "q4_0", llama_cpp_dir, dry_run)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    # ------------------------------------------------------------------
    # Format backends
    # ------------------------------------------------------------------

    def _export_safetensors(self, model_dir, output_dir, dry_run):
        from xlmtec.export.backends.safetensors import export_safetensors
        result = export_safetensors(model_dir, output_dir, dry_run=dry_run)
        return ExportResult(
            format=ExportFormat.SAFETENSORS,
            output_path=result.output_path,
            file_size_mb=result.file_size_mb,
            dry_run=dry_run,
            extra={"num_files": result.num_files},
        )

    def _export_onnx(self, model_dir, output_dir, quantize, dry_run):
        from xlmtec.export.backends.onnx import export_onnx
        result = export_onnx(model_dir, output_dir, quantize=quantize, dry_run=dry_run)
        return ExportResult(
            format=ExportFormat.ONNX,
            output_path=result.output_path,
            file_size_mb=result.file_size_mb,
            dry_run=dry_run,
            extra={"quantized": result.quantized, "quantize_type": result.quantize_type},
        )

    def _export_gguf(self, model_dir, output_dir, quantize, llama_cpp_dir, dry_run):
        from xlmtec.export.backends.gguf import export_gguf
        result = export_gguf(
            model_dir, output_dir,
            quantize=quantize,
            llama_cpp_dir=llama_cpp_dir,
            dry_run=dry_run,
        )
        return ExportResult(
            format=ExportFormat.GGUF,
            output_path=result.output_path,
            file_size_mb=result.file_size_mb,
            dry_run=dry_run,
            extra={"quantize_type": result.quantize_type},
        )

    # ------------------------------------------------------------------
    # Dependency check
    # ------------------------------------------------------------------

    def _check_dependencies(self, meta: FormatMeta) -> None:
        """Warn if required packages are missing (non-fatal — backend will raise)."""
        missing = []
        for pkg in meta.required_packages:
            try:
                __import__(pkg.replace("-", "_"))
            except ImportError:
                missing.append(pkg)
        if missing:
            raise ImportError(
                f"Missing packages for {meta.name} export: {', '.join(missing)}\n"
                f"Install with: pip install xlmtec[{meta.pip_extra}]"
            )