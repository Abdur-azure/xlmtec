"""
Unit tests for the `evaluate` CLI subcommand.

All model loading, dataset loading, and evaluation are mocked.
No GPU, no network, no real model files required.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from finetune_cli.cli.main import app

runner = CliRunner()


# ============================================================================
# HELPERS
# ============================================================================

def _mock_eval_result(scores=None):
    result = MagicMock()
    result.scores = scores or {"rougeL": 0.42, "bleu": 0.18}
    return result


def _mock_evaluation_stack():
    """Patch at source module — evaluate uses lazy imports inside the function."""
    return [
        patch("finetune_cli.models.loader.load_model_and_tokenizer",
              return_value=(MagicMock(), MagicMock())),
        patch("finetune_cli.data.quick_load", return_value=MagicMock()),
        patch("finetune_cli.evaluation.BenchmarkRunner"),
    ]


# ============================================================================
# TESTS
# ============================================================================

class TestEvaluateCommand:

    def test_requires_dataset_or_hf_dataset(self, tmp_path):
        """evaluate exits 1 when neither --dataset nor --hf-dataset is given."""
        result = runner.invoke(app, [
            "evaluate",
            str(tmp_path / "model"),
        ])
        assert result.exit_code == 1

    def test_local_dataset_exits_zero(self, tmp_path):
        """evaluate exits 0 with a valid local --dataset."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        ds = tmp_path / "eval.jsonl"
        ds.write_text('{"text": "hello"}\n')

        mock_result = _mock_eval_result()
        patches = _mock_evaluation_stack()

        with patches[0], patches[1], patches[2] as mock_runner_cls:
            mock_runner_cls.return_value.evaluate.return_value = mock_result
            result = runner.invoke(app, [
                "evaluate",
                str(model_dir),
                "--dataset", str(ds),
            ])

        assert result.exit_code == 0, result.output

    def test_hf_dataset_exits_zero(self, tmp_path):
        """evaluate exits 0 when --hf-dataset is provided."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_result = _mock_eval_result()
        patches = _mock_evaluation_stack()

        with patches[0], patches[1], patches[2] as mock_runner_cls:
            mock_runner_cls.return_value.evaluate.return_value = mock_result
            result = runner.invoke(app, [
                "evaluate",
                str(model_dir),
                "--hf-dataset", "wikitext",
            ])

        assert result.exit_code == 0, result.output

    def test_metric_scores_appear_in_output(self, tmp_path):
        """Metric scores from the runner are printed to stdout."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        ds = tmp_path / "eval.jsonl"
        ds.write_text('{"text": "hello"}\n')

        mock_result = _mock_eval_result({"rougeL": 0.4231, "bleu": 0.1876})
        patches = _mock_evaluation_stack()

        with patches[0], patches[1], patches[2] as mock_runner_cls:
            mock_runner_cls.return_value.evaluate.return_value = mock_result
            result = runner.invoke(app, [
                "evaluate",
                str(model_dir),
                "--dataset", str(ds),
                "--metrics", "rougeL,bleu",
            ])

        assert "0.4231" in result.output
        assert "0.1876" in result.output

    def test_unknown_metric_warns_and_exits_one(self, tmp_path):
        """An entirely invalid metric name causes exit 1 with no crash."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        ds = tmp_path / "eval.jsonl"
        ds.write_text('{"text": "hello"}\n')

        result = runner.invoke(app, [
            "evaluate",
            str(model_dir),
            "--dataset", str(ds),
            "--metrics", "not_a_real_metric",
        ])

        assert result.exit_code == 1

    def test_num_samples_flag_accepted(self, tmp_path):
        """--num-samples flag is accepted and passed through."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        ds = tmp_path / "eval.jsonl"
        ds.write_text('{"text": "hello"}\n')

        mock_result = _mock_eval_result()
        patches = _mock_evaluation_stack()

        with patches[0], patches[1], patches[2] as mock_runner_cls:
            mock_runner_cls.return_value.evaluate.return_value = mock_result
            result = runner.invoke(app, [
                "evaluate",
                str(model_dir),
                "--dataset", str(ds),
                "--num-samples", "50",
            ])

        assert result.exit_code == 0, result.output
