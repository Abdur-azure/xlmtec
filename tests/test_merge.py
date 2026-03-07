"""
Unit tests for the `lmtool merge` command.

All model loading and PEFT ops are mocked — no GPU, no downloads required.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from typer.testing import CliRunner

from lmtool.cli.main import app

runner = CliRunner()


# ============================================================================
# HELPERS
# ============================================================================

def _make_adapter_dir(tmp_path: Path) -> Path:
    """Create a minimal valid adapter directory."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    # adapter_config.json is the presence check in the merge command
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "gpt2", "peft_type": "LORA"}),
        encoding="utf-8",
    )
    return adapter_dir


def _mock_merge_stack(output_dir: Path):
    """Returns patches that simulate a successful merge without any real models."""
    mock_merged = MagicMock()
    mock_tokenizer = MagicMock()

    # save_pretrained writes config.json so the verify step passes
    def fake_save_pretrained(path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "config.json").write_text("{}", encoding="utf-8")
        (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")

    mock_merged.save_pretrained.side_effect = fake_save_pretrained
    mock_tokenizer.save_pretrained.side_effect = lambda path, **kw: None

    mock_peft_model = MagicMock()
    mock_peft_model.merge_and_unload.return_value = mock_merged

    patches = [
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()),
        patch("peft.PeftModel.from_pretrained", return_value=mock_peft_model),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
    ]
    return patches, mock_merged, mock_tokenizer, mock_peft_model


# ============================================================================
# TESTS
# ============================================================================

class TestMergeCommand:

    def test_merge_succeeds_with_valid_adapter(self, tmp_path):
        """Happy path — exits 0 and output dir is created."""
        adapter_dir = _make_adapter_dir(tmp_path)
        output_dir = tmp_path / "merged"

        patches, _, _, _ = _mock_merge_stack(output_dir)
        with patches[0], patches[1], patches[2]:
            result = runner.invoke(app, [
                "merge",
                str(adapter_dir),
                str(output_dir),
                "--base-model", "gpt2",
            ])

        assert result.exit_code == 0, result.output
        assert output_dir.exists()

    def test_merge_saves_model_and_tokenizer(self, tmp_path):
        """merged.save_pretrained and tokenizer.save_pretrained are both called."""
        adapter_dir = _make_adapter_dir(tmp_path)
        output_dir = tmp_path / "merged"

        patches, mock_merged, mock_tokenizer, _ = _mock_merge_stack(output_dir)
        with patches[0], patches[1], patches[2]:
            runner.invoke(app, [
                "merge", str(adapter_dir), str(output_dir), "--base-model", "gpt2",
            ])

        mock_merged.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()

    def test_merge_calls_merge_and_unload(self, tmp_path):
        """PeftModel.merge_and_unload() must be called — this is the core op."""
        adapter_dir = _make_adapter_dir(tmp_path)
        output_dir = tmp_path / "merged"

        patches, _, _, mock_peft_model = _mock_merge_stack(output_dir)
        with patches[0], patches[1], patches[2]:
            runner.invoke(app, [
                "merge", str(adapter_dir), str(output_dir), "--base-model", "gpt2",
            ])
            # Assert inside the with block — mock only valid while patch is active
            mock_peft_model.merge_and_unload.assert_called_once()

    def test_merge_missing_adapter_dir_exits_1(self, tmp_path):
        """Non-existent adapter_dir exits with code 1."""
        result = runner.invoke(app, [
            "merge",
            str(tmp_path / "does_not_exist"),
            str(tmp_path / "out"),
            "--base-model", "gpt2",
        ])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_merge_missing_adapter_config_exits_1(self, tmp_path):
        """Dir exists but has no adapter_config.json — exits 1 with clear message."""
        bad_dir = tmp_path / "bad_adapter"
        bad_dir.mkdir()
        # No adapter_config.json

        result = runner.invoke(app, [
            "merge",
            str(bad_dir),
            str(tmp_path / "out"),
            "--base-model", "gpt2",
        ])
        assert result.exit_code == 1
        assert "adapter_config.json" in result.output

    def test_merge_base_model_required(self, tmp_path):
        """--base-model is required — typer enforces this at parse time."""
        adapter_dir = _make_adapter_dir(tmp_path)
        result = runner.invoke(app, [
            "merge",
            str(adapter_dir),
            str(tmp_path / "out"),
            # no --base-model
        ])
        assert result.exit_code != 0

    def test_merge_invalid_dtype_exits_1(self, tmp_path):
        """Unknown --dtype value exits with code 1."""
        adapter_dir = _make_adapter_dir(tmp_path)
        patches, _, _, _ = _mock_merge_stack(tmp_path / "out")
        with patches[0], patches[1], patches[2]:
            result = runner.invoke(app, [
                "merge",
                str(adapter_dir),
                str(tmp_path / "out"),
                "--base-model", "gpt2",
                "--dtype", "float99",
            ])
        assert result.exit_code == 1
        assert "dtype" in result.output.lower()

    def test_merge_float16_dtype_accepted(self, tmp_path):
        """float16 is a valid dtype and produces exit 0."""
        adapter_dir = _make_adapter_dir(tmp_path)
        output_dir = tmp_path / "merged_fp16"

        patches, _, _, _ = _mock_merge_stack(output_dir)
        with patches[0], patches[1], patches[2]:
            result = runner.invoke(app, [
                "merge",
                str(adapter_dir),
                str(output_dir),
                "--base-model", "gpt2",
                "--dtype", "float16",
            ])
        assert result.exit_code == 0, result.output
