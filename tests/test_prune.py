"""
Unit tests for the `finetune-cli prune` CLI subcommand.

All model loading and pruning are mocked — no GPU, no real model required.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from finetune_cli.cli.main import app
from finetune_cli.trainers.structured_pruner import PruningResult

runner = CliRunner()

# ============================================================================
# Helpers
# ============================================================================

_LOADER_PATH = "finetune_cli.models.loader.load_model_and_tokenizer"
_PRUNER_PATH = "finetune_cli.trainers.structured_pruner.StructuredPruner"


def _mock_pruning_result(output_dir: Path) -> PruningResult:
    return PruningResult(
        output_dir=output_dir,
        original_param_count=100_000,
        zeroed_param_count=10_000,
        sparsity_achieved=0.10,
        heads_pruned_per_layer={"model.layers[0]": 1},
        pruning_time_seconds=0.5,
    )


def _make_pruner_mock(output_dir: Path) -> MagicMock:
    mock_pruner = MagicMock()
    mock_pruner.prune.return_value = _mock_pruning_result(output_dir)
    return mock_pruner


# ============================================================================
# Tests
# ============================================================================

class TestPruneCommand:

    def test_missing_model_path_exits_one(self, tmp_path):
        """prune exits 1 when model_path does not exist."""
        result = runner.invoke(app, [
            "prune",
            str(tmp_path / "nonexistent"),
            "--output", str(tmp_path / "out"),
        ])
        assert result.exit_code == 1

    def test_invalid_sparsity_exits_one(self, tmp_path):
        """prune exits 1 when --sparsity >= 1.0."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        result = runner.invoke(app, [
            "prune",
            str(model_dir),
            "--output", str(tmp_path / "out"),
            "--sparsity", "1.5",
        ])
        assert result.exit_code == 1

    def test_invalid_method_exits_one(self, tmp_path):
        """prune exits 1 when --method is not 'heads' or 'ffn'."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        result = runner.invoke(app, [
            "prune",
            str(model_dir),
            "--output", str(tmp_path / "out"),
            "--method", "badmethod",
        ])
        assert result.exit_code == 1

    def test_happy_path_exits_zero(self, tmp_path):
        """Successful prune run exits 0."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_dir = tmp_path / "pruned"

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_pruner = _make_pruner_mock(output_dir)

        with patch(_LOADER_PATH,
                   return_value=(mock_model, mock_tokenizer)):
            with patch(_PRUNER_PATH,
                       return_value=mock_pruner):
                result = runner.invoke(app, [
                    "prune",
                    str(model_dir),
                    "--output", str(output_dir),
                    "--sparsity", "0.3",
                ])

        assert result.exit_code == 0, result.output

    def test_sparsity_passed_to_pruning_config(self, tmp_path):
        """--sparsity value is forwarded to PruningConfig."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_dir = tmp_path / "pruned"

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_pruner = _make_pruner_mock(output_dir)
        captured_config = {}

        def capture_pruner(model, tokenizer, config):
            captured_config["config"] = config
            return mock_pruner

        with patch(_LOADER_PATH,
                   return_value=(mock_model, mock_tokenizer)):
            with patch(_PRUNER_PATH,
                       side_effect=capture_pruner):
                runner.invoke(app, [
                    "prune",
                    str(model_dir),
                    "--output", str(output_dir),
                    "--sparsity", "0.25",
                ])

        assert captured_config["config"].sparsity == pytest.approx(0.25)

    def test_method_ffn_accepted(self, tmp_path):
        """--method ffn is a valid value and exits 0."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_dir = tmp_path / "pruned"

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_pruner = _make_pruner_mock(output_dir)

        with patch(_LOADER_PATH,
                   return_value=(mock_model, mock_tokenizer)):
            with patch(_PRUNER_PATH,
                       return_value=mock_pruner):
                result = runner.invoke(app, [
                    "prune",
                    str(model_dir),
                    "--output", str(output_dir),
                    "--method", "ffn",
                ])

        assert result.exit_code == 0, result.output

    def test_output_contains_sparsity_achieved(self, tmp_path):
        """CLI output includes sparsity achieved from PruningResult."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_dir = tmp_path / "pruned"

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_pruner = _make_pruner_mock(output_dir)

        with patch(_LOADER_PATH,
                   return_value=(mock_model, mock_tokenizer)):
            with patch(_PRUNER_PATH,
                       return_value=mock_pruner):
                result = runner.invoke(app, [
                    "prune",
                    str(model_dir),
                    "--output", str(output_dir),
                ])

        assert "10.0%" in result.output or "Sparsity" in result.output
