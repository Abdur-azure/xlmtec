"""
tests/test_export.py
~~~~~~~~~~~~~~~~~~~~~
Tests for export formats, exporter dispatch, and CLI command.
All model I/O is mocked — no real models or ML deps needed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from xlmtec.export.formats import ExportFormat, get_format_meta

# ---------------------------------------------------------------------------
# ExportFormat
# ---------------------------------------------------------------------------


class TestExportFormat:
    def test_from_str_valid(self):
        assert ExportFormat.from_str("safetensors") == ExportFormat.SAFETENSORS
        assert ExportFormat.from_str("onnx") == ExportFormat.ONNX
        assert ExportFormat.from_str("gguf") == ExportFormat.GGUF

    def test_from_str_case_insensitive(self):
        assert ExportFormat.from_str("ONNX") == ExportFormat.ONNX
        assert ExportFormat.from_str("Safetensors") == ExportFormat.SAFETENSORS

    def test_from_str_invalid(self):
        with pytest.raises(ValueError, match="Unknown export format"):
            ExportFormat.from_str("parquet")

    def test_error_lists_available(self):
        with pytest.raises(ValueError, match="safetensors"):
            ExportFormat.from_str("bad")


# ---------------------------------------------------------------------------
# FORMAT_META
# ---------------------------------------------------------------------------


class TestFormatMeta:
    def test_all_formats_have_meta(self):
        for fmt in ExportFormat:
            meta = get_format_meta(fmt)
            assert meta.name
            assert meta.extension
            assert meta.description

    def test_safetensors_extension(self):
        assert get_format_meta(ExportFormat.SAFETENSORS).extension == ".safetensors"

    def test_onnx_extension(self):
        assert get_format_meta(ExportFormat.ONNX).extension == ".onnx"

    def test_gguf_extension(self):
        assert get_format_meta(ExportFormat.GGUF).extension == ".gguf"

    def test_onnx_has_quantize_options(self):
        meta = get_format_meta(ExportFormat.ONNX)
        assert "fp16" in meta.quantize_options
        assert "int8" in meta.quantize_options

    def test_gguf_has_quantize_options(self):
        meta = get_format_meta(ExportFormat.GGUF)
        assert "q4_0" in meta.quantize_options
        assert "q8_0" in meta.quantize_options


# ---------------------------------------------------------------------------
# Safetensors backend
# ---------------------------------------------------------------------------


class TestSafetensorsBackend:
    def test_dry_run_returns_result(self, tmp_path):
        from xlmtec.export.backends.safetensors import export_safetensors

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        result = export_safetensors(model_dir, tmp_path / "out", dry_run=True)
        assert result.output_path.suffix == ".safetensors"
        assert result.file_size_mb == 0.0

    def test_missing_model_dir_raises(self, tmp_path):
        from xlmtec.export.backends.safetensors import export_safetensors

        with pytest.raises(FileNotFoundError):
            export_safetensors(tmp_path / "missing", tmp_path / "out")

    def test_missing_dep_raises_import_error(self, tmp_path):
        from xlmtec.export.backends.safetensors import export_safetensors

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        with patch.dict("sys.modules", {"transformers": None, "safetensors": None}):
            with pytest.raises((ImportError, TypeError)):
                export_safetensors(model_dir, tmp_path / "out", dry_run=False)


# ---------------------------------------------------------------------------
# ONNX backend
# ---------------------------------------------------------------------------


class TestOnnxBackend:
    def test_dry_run_returns_result(self, tmp_path):
        from xlmtec.export.backends.onnx import export_onnx

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        result = export_onnx(model_dir, tmp_path / "out", dry_run=True)
        assert result.output_path.suffix == ".onnx"

    def test_dry_run_with_quantize(self, tmp_path):
        from xlmtec.export.backends.onnx import export_onnx

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        result = export_onnx(model_dir, tmp_path / "out", quantize="fp16", dry_run=True)
        assert result.quantized is True
        assert result.quantize_type == "fp16"

    def test_invalid_quantize_raises(self, tmp_path):
        from xlmtec.export.backends.onnx import export_onnx

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        with pytest.raises(ValueError, match="Invalid ONNX"):
            export_onnx(model_dir, tmp_path / "out", quantize="q4_0")

    def test_missing_model_dir_raises(self, tmp_path):
        from xlmtec.export.backends.onnx import export_onnx

        with pytest.raises(FileNotFoundError):
            export_onnx(tmp_path / "missing", tmp_path / "out")


# ---------------------------------------------------------------------------
# GGUF backend
# ---------------------------------------------------------------------------


class TestGgufBackend:
    def test_dry_run_returns_result(self, tmp_path):
        from xlmtec.export.backends.gguf import export_gguf

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        result = export_gguf(model_dir, tmp_path / "out", quantize="q4_0", dry_run=True)
        assert ".gguf" in result.output_path.suffix
        assert result.quantize_type == "q4_0"

    def test_invalid_quantize_raises(self, tmp_path):
        from xlmtec.export.backends.gguf import export_gguf

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        with pytest.raises(ValueError, match="Invalid GGUF"):
            export_gguf(model_dir, tmp_path / "out", quantize="int8")

    def test_missing_model_dir_raises(self, tmp_path):
        from xlmtec.export.backends.gguf import export_gguf

        with pytest.raises(FileNotFoundError):
            export_gguf(tmp_path / "missing", tmp_path / "out")


# ---------------------------------------------------------------------------
# ModelExporter dispatch
# ---------------------------------------------------------------------------


class TestModelExporter:
    def _model_dir(self, tmp_path):
        d = tmp_path / "model"
        d.mkdir()
        return d

    def test_dispatch_safetensors_dry_run(self, tmp_path):
        from xlmtec.export.exporter import ModelExporter

        result = ModelExporter().export(
            model_dir=self._model_dir(tmp_path),
            output_dir=tmp_path / "out",
            fmt=ExportFormat.SAFETENSORS,
            dry_run=True,
        )
        assert result.format == ExportFormat.SAFETENSORS
        assert result.dry_run is True

    def test_dispatch_onnx_dry_run(self, tmp_path):
        from xlmtec.export.exporter import ModelExporter

        result = ModelExporter().export(
            model_dir=self._model_dir(tmp_path),
            output_dir=tmp_path / "out",
            fmt=ExportFormat.ONNX,
            dry_run=True,
        )
        assert result.format == ExportFormat.ONNX

    def test_dispatch_gguf_dry_run(self, tmp_path):
        from xlmtec.export.exporter import ModelExporter

        result = ModelExporter().export(
            model_dir=self._model_dir(tmp_path),
            output_dir=tmp_path / "out",
            fmt=ExportFormat.GGUF,
            dry_run=True,
        )
        assert result.format == ExportFormat.GGUF

    def test_missing_dep_raises_on_real_export(self, tmp_path):
        from xlmtec.export.exporter import ModelExporter

        with patch.object(ModelExporter, "_check_dependencies", side_effect=ImportError("missing")):
            with pytest.raises(ImportError, match="missing"):
                ModelExporter().export(
                    model_dir=self._model_dir(tmp_path),
                    output_dir=tmp_path / "out",
                    fmt=ExportFormat.SAFETENSORS,
                    dry_run=False,
                )


# ---------------------------------------------------------------------------
# export_model CLI logic
# ---------------------------------------------------------------------------


class TestExportModelLogic:
    def test_dry_run_safetensors_returns_0(self, tmp_path):
        from xlmtec.cli.commands.export import export_model

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        code = export_model(model_dir, tmp_path / "out", "safetensors", None, None, dry_run=True)
        assert code == 0

    def test_dry_run_onnx_returns_0(self, tmp_path):
        from xlmtec.cli.commands.export import export_model

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        code = export_model(model_dir, tmp_path / "out", "onnx", None, None, dry_run=True)
        assert code == 0

    def test_dry_run_gguf_returns_0(self, tmp_path):
        from xlmtec.cli.commands.export import export_model

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        code = export_model(model_dir, tmp_path / "out", "gguf", "q4_0", None, dry_run=True)
        assert code == 0

    def test_invalid_format_returns_1(self, tmp_path):
        from xlmtec.cli.commands.export import export_model

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        code = export_model(model_dir, tmp_path / "out", "parquet", None, None, dry_run=True)
        assert code == 1

    def test_missing_model_dir_returns_1(self, tmp_path):
        from xlmtec.cli.commands.export import export_model

        code = export_model(
            tmp_path / "missing", tmp_path / "out", "safetensors", None, None, dry_run=True
        )
        assert code == 1

    def test_invalid_quantize_for_format_returns_1(self, tmp_path):
        from xlmtec.cli.commands.export import export_model

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        code = export_model(model_dir, tmp_path / "out", "onnx", "q4_0", None, dry_run=True)
        assert code == 1
