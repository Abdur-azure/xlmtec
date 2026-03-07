"""
Unit tests for the `finetune-cli wanda` CLI subcommand.

All model loading and pruning are mocked — no GPU, no real model required.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from finetune_cli.cli.main import app
from finetune_cli.trainers.wanda_pruner import WandaResult

runner = CliRunner()

_LOADER_PATH = "finetune_cli.models.loader.load_model_and_tokenizer"
_WANDA_PATH  = "finetune_cli.trainers.wanda_pruner.WandaPruner"


def _mock_result(output_dir: Path) -> WandaResult:
    return WandaResult(
        output_dir=output_dir,
        original_param_count=200_000,
        zeroed_param_count=100_000,
        sparsity_achieved=0.5,
        layers_pruned=12,
        pruning_time_seconds=1.2,
    )


def _make_pruner_mock(output_dir: Path) -> MagicMock:
    m = MagicMock()
    m.prune.return_value = _mock_result(output_dir)
    return m


class TestWandaCommand:

    def test_missing_model_path_exits_one(self, tmp_path):
        result = runner.invoke(app, [
            "wanda",
            str(tmp_path / "nonexistent"),
            "--output", str(tmp_path / "out"),
        ])
        assert result.exit_code == 1

    def test_invalid_sparsity_exits_one(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        result = runner.invoke(app, [
            "wanda",
            str(model_dir),
            "--output", str(tmp_path / "out"),
            "--sparsity", "1.5",
        ])
        assert result.exit_code == 1

    def test_happy_path_exits_zero(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_dir = tmp_path / "pruned"

        mock_pruner = _make_pruner_mock(output_dir)
        with patch(_LOADER_PATH, return_value=(MagicMock(), MagicMock())):
            with patch(_WANDA_PATH, return_value=mock_pruner):
                result = runner.invoke(app, [
                    "wanda",
                    str(model_dir),
                    "--output", str(output_dir),
                    "--sparsity", "0.5",
                ])
        assert result.exit_code == 0, result.output

    def test_sparsity_forwarded_to_config(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_dir = tmp_path / "pruned"
        captured = {}

        def capture(model, tokenizer, config):
            captured["config"] = config
            return _make_pruner_mock(output_dir)

        with patch(_LOADER_PATH, return_value=(MagicMock(), MagicMock())):
            with patch(_WANDA_PATH, side_effect=capture):
                runner.invoke(app, [
                    "wanda",
                    str(model_dir),
                    "--output", str(output_dir),
                    "--sparsity", "0.3",
                ])
        assert captured["config"].sparsity == pytest.approx(0.3)

    def test_output_contains_sparsity_achieved(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_dir = tmp_path / "pruned"

        mock_pruner = _make_pruner_mock(output_dir)
        with patch(_LOADER_PATH, return_value=(MagicMock(), MagicMock())):
            with patch(_WANDA_PATH, return_value=mock_pruner):
                result = runner.invoke(app, [
                    "wanda",
                    str(model_dir),
                    "--output", str(output_dir),
                ])
        assert "50.0%" in result.output or "Sparsity" in result.output

    def test_missing_dataset_exits_one(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        with patch(_LOADER_PATH, return_value=(MagicMock(), MagicMock())):
            result = runner.invoke(app, [
                "wanda",
                str(model_dir),
                "--output", str(tmp_path / "out"),
                "--dataset", str(tmp_path / "nonexistent.jsonl"),
            ])
        assert result.exit_code == 1
