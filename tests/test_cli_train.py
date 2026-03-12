"""
Unit tests for the CLI `train` command flag-to-config wiring.

Verifies that each training method builds the right config from flags
without crashing. No model loading, no GPU required — all heavy ops mocked.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from xlmtec.cli.main import app

runner = CliRunner()


# ============================================================================
# HELPERS
# ============================================================================


def _make_dataset_file(tmp_path: Path) -> Path:
    """Write a minimal JSONL file so path validation passes."""
    f = tmp_path / "data.jsonl"
    f.write_text(json.dumps({"text": "hello world"}) + "\n", encoding="utf-8")
    return f


def _mock_training_stack():
    """Context manager stack that skips model load + training."""
    mock_result = MagicMock()
    mock_result.output_dir = Path("./output")
    mock_result.train_loss = 0.5
    mock_result.steps_completed = 10

    patches = [
        patch(
            "xlmtec.models.loader.load_model_and_tokenizer", return_value=(MagicMock(), MagicMock())
        ),
        patch("xlmtec.data.prepare_dataset", return_value=MagicMock()),
        patch("xlmtec.trainers.TrainerFactory.train", return_value=mock_result),
    ]
    return patches


# ============================================================================
# TESTS
# ============================================================================


class TestTrainCommand:
    def test_lora_via_flags_builds_config(self, tmp_path):
        """Default method (lora) works with minimal flags."""
        ds = _make_dataset_file(tmp_path)
        patches = _mock_training_stack()
        with patches[0], patches[1], patches[2]:
            result = runner.invoke(
                app,
                [
                    "train",
                    "--model",
                    "gpt2",
                    "--dataset",
                    str(ds),
                    "--method",
                    "lora",
                    "--epochs",
                    "1",
                    "--output",
                    str(tmp_path / "out"),
                ],
            )
        assert result.exit_code == 0, result.output

    def test_full_finetuning_requires_no_lora_flags(self, tmp_path):
        """full_finetuning must not crash when no lora flags are given."""
        ds = _make_dataset_file(tmp_path)
        patches = _mock_training_stack()
        with patches[0], patches[1], patches[2]:
            result = runner.invoke(
                app,
                [
                    "train",
                    "--model",
                    "gpt2",
                    "--dataset",
                    str(ds),
                    "--method",
                    "full_finetuning",
                    "--epochs",
                    "1",
                    "--output",
                    str(tmp_path / "out"),
                ],
            )
        assert result.exit_code == 0, result.output

    def test_instruction_tuning_via_flags(self, tmp_path):
        """instruction_tuning builds lora config from flags without crashing."""
        ds = _make_dataset_file(tmp_path)
        patches = _mock_training_stack()
        with patches[0], patches[1], patches[2]:
            result = runner.invoke(
                app,
                [
                    "train",
                    "--model",
                    "gpt2",
                    "--dataset",
                    str(ds),
                    "--method",
                    "instruction_tuning",
                    "--epochs",
                    "1",
                    "--output",
                    str(tmp_path / "out"),
                ],
            )
        assert result.exit_code == 0, result.output

    def test_missing_dataset_exits_with_error(self, tmp_path):
        """No --dataset and no --hf-dataset exits with code 1."""
        result = runner.invoke(
            app,
            [
                "train",
                "--model",
                "gpt2",
                "--method",
                "lora",
            ],
        )
        assert result.exit_code == 1

    def test_config_file_takes_precedence(self, tmp_path):
        """--config file is loaded instead of building from flags."""
        ds = _make_dataset_file(tmp_path)
        out_dir = tmp_path / "out"
        cfg = tmp_path / "config.yaml"
        # Use .as_posix() — backslashes in Windows paths break YAML parsing
        cfg.write_text(
            f"""
model:
  name: gpt2
dataset:
  source: local_file
  path: "{ds.as_posix()}"
tokenization:
  max_length: 64
training:
  method: lora
  output_dir: "{out_dir.as_posix()}"
  num_epochs: 1
  batch_size: 2
lora:
  r: 4
  lora_alpha: 8
""",
            encoding="utf-8",
        )

        patches = _mock_training_stack()
        with patches[0], patches[1], patches[2]:
            result = runner.invoke(app, ["train", "--config", str(cfg)])
        assert result.exit_code == 0, result.output

    def test_full_finetuning_via_flags(self, tmp_path):
        """full_finetuning method exits 0 (no lora_config required)."""
        ds = _make_dataset_file(tmp_path)
        patches = _mock_training_stack()
        with patches[0], patches[1], patches[2]:
            result = runner.invoke(
                app,
                [
                    "train",
                    "--model",
                    "gpt2",
                    "--dataset",
                    str(ds),
                    "--method",
                    "full_finetuning",
                    "--epochs",
                    "1",
                    "--output",
                    str(tmp_path / "out"),
                ],
            )
        assert result.exit_code == 0, result.output

    def test_dpo_via_flags(self, tmp_path):
        """dpo method gets lora config and exits 0."""
        ds = _make_dataset_file(tmp_path)
        patches = _mock_training_stack()
        with patches[0], patches[1], patches[2]:
            result = runner.invoke(
                app,
                [
                    "train",
                    "--model",
                    "gpt2",
                    "--dataset",
                    str(ds),
                    "--method",
                    "dpo",
                    "--epochs",
                    "1",
                    "--output",
                    str(tmp_path / "out"),
                ],
            )
        assert result.exit_code == 0, result.output

    def test_qlora_sets_lora_config(self, tmp_path):
        """qlora method still gets lora config from flags."""
        ds = _make_dataset_file(tmp_path)
        patches = _mock_training_stack()
        with patches[0], patches[1], patches[2]:
            result = runner.invoke(
                app,
                [
                    "train",
                    "--model",
                    "gpt2",
                    "--dataset",
                    str(ds),
                    "--method",
                    "qlora",
                    "--lora-r",
                    "16",
                    "--epochs",
                    "1",
                    "--output",
                    str(tmp_path / "out"),
                ],
            )
        assert result.exit_code == 0, result.output
