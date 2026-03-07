"""
Unit tests for the `benchmark` CLI subcommand.

All model loading, dataset loading, and comparison are mocked.
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

def _mock_report():
    report = MagicMock()
    report.summary.return_value = (
        "Metric       Base      Fine-tuned   Delta\n"
        "rougeL       0.3012    0.4231       0.1219\n"
        "bleu         0.1204    0.1876       0.0672\n"
    )
    return report


def _mock_benchmark_stack():
    """Patch at source module — benchmark uses lazy imports inside the function."""
    return [
        patch("finetune_cli.models.loader.load_model_and_tokenizer",
              return_value=(MagicMock(), MagicMock())),
        patch("finetune_cli.data.quick_load", return_value=MagicMock()),
        patch("finetune_cli.evaluation.BenchmarkRunner"),
    ]


# ============================================================================
# TESTS
# ============================================================================

class TestBenchmarkCommand:

    def test_requires_dataset_or_hf_dataset(self, tmp_path):
        """benchmark exits 1 when neither --dataset nor --hf-dataset is given."""
        result = runner.invoke(app, [
            "benchmark",
            "gpt2",
            str(tmp_path / "finetuned"),
        ])
        assert result.exit_code == 1

    def test_local_dataset_exits_zero(self, tmp_path):
        """benchmark exits 0 with base model, finetuned path, and --dataset."""
        ft_dir = tmp_path / "finetuned"
        ft_dir.mkdir()
        ds = tmp_path / "eval.jsonl"
        ds.write_text('{"text": "hello"}\n')

        mock_report = _mock_report()
        patches = _mock_benchmark_stack()

        with patches[0], patches[1], patches[2] as mock_runner_cls:
            mock_runner_cls.return_value.run_comparison.return_value = mock_report
            result = runner.invoke(app, [
                "benchmark",
                "gpt2",
                str(ft_dir),
                "--dataset", str(ds),
            ])

        assert result.exit_code == 0, result.output

    def test_hf_dataset_exits_zero(self, tmp_path):
        """benchmark exits 0 when --hf-dataset is provided."""
        ft_dir = tmp_path / "finetuned"
        ft_dir.mkdir()

        mock_report = _mock_report()
        patches = _mock_benchmark_stack()

        with patches[0], patches[1], patches[2] as mock_runner_cls:
            mock_runner_cls.return_value.run_comparison.return_value = mock_report
            result = runner.invoke(app, [
                "benchmark",
                "gpt2",
                str(ft_dir),
                "--hf-dataset", "wikitext",
            ])

        assert result.exit_code == 0, result.output

    def test_summary_appears_in_output(self, tmp_path):
        """Summary from BenchmarkReport.summary() is printed to stdout."""
        ft_dir = tmp_path / "finetuned"
        ft_dir.mkdir()
        ds = tmp_path / "eval.jsonl"
        ds.write_text('{"text": "hello"}\n')

        mock_report = _mock_report()
        patches = _mock_benchmark_stack()

        with patches[0], patches[1], patches[2] as mock_runner_cls:
            mock_runner_cls.return_value.run_comparison.return_value = mock_report
            result = runner.invoke(app, [
                "benchmark",
                "gpt2",
                str(ft_dir),
                "--dataset", str(ds),
            ])

        assert "rougeL" in result.output
        assert "0.4231" in result.output

    def test_run_comparison_called_with_two_models(self, tmp_path):
        """BenchmarkRunner.run_comparison is called once."""
        ft_dir = tmp_path / "finetuned"
        ft_dir.mkdir()
        ds = tmp_path / "eval.jsonl"
        ds.write_text('{"text": "hello"}\n')

        mock_report = _mock_report()
        patches = _mock_benchmark_stack()

        with patches[0], patches[1], patches[2] as mock_runner_cls:
            mock_runner_cls.return_value.run_comparison.return_value = mock_report
            result = runner.invoke(app, [
                "benchmark",
                "gpt2",
                str(ft_dir),
                "--dataset", str(ds),
            ])

        mock_runner_cls.return_value.run_comparison.assert_called_once()
        assert result.exit_code == 0, result.output

    def test_num_samples_flag_accepted(self, tmp_path):
        """--num-samples flag is accepted without error."""
        ft_dir = tmp_path / "finetuned"
        ft_dir.mkdir()
        ds = tmp_path / "eval.jsonl"
        ds.write_text('{"text": "hello"}\n')

        mock_report = _mock_report()
        patches = _mock_benchmark_stack()

        with patches[0], patches[1], patches[2] as mock_runner_cls:
            mock_runner_cls.return_value.run_comparison.return_value = mock_report
            result = runner.invoke(app, [
                "benchmark",
                "gpt2",
                str(ft_dir),
                "--dataset", str(ds),
                "--num-samples", "50",
            ])

        assert result.exit_code == 0, result.output
