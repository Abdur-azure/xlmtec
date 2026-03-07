"""
Unit tests for the `recommend` CLI subcommand.

Mocks AutoConfig and torch.cuda so tests run without network or GPU.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from lmtool.cli.main import app

runner = CliRunner()


def _mock_autoconfig(hidden=768, layers=12):
    cfg = MagicMock()
    cfg.hidden_size = hidden
    cfg.num_hidden_layers = layers
    del cfg.n_embd
    del cfg.n_layer
    return cfg


# ============================================================================
# TESTS
# ============================================================================

class TestRecommendCommand:

    @patch("torch.cuda.is_available", return_value=False)
    @patch("transformers.AutoConfig.from_pretrained")
    def test_outputs_valid_yaml(self, mock_cfg, mock_cuda):
        mock_cfg.return_value = _mock_autoconfig(768, 12)
        result = runner.invoke(app, ["recommend", "gpt2"])
        assert result.exit_code == 0
        assert "method:" in result.output

    @patch("torch.cuda.is_available", return_value=False)
    @patch("transformers.AutoConfig.from_pretrained")
    def test_small_model_no_gpu_recommends_lora_or_full(self, mock_cfg, mock_cuda):
        mock_cfg.return_value = _mock_autoconfig(768, 12)  # ~124M
        result = runner.invoke(app, ["recommend", "gpt2"])
        assert result.exit_code == 0
        assert "method:" in result.output

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_large_model_with_low_vram_recommends_qlora(self, mock_cfg, mock_props, mock_cuda):
        mock_cfg.return_value = _mock_autoconfig(hidden=4096, layers=32)  # ~7B
        mock_props.return_value = MagicMock(total_memory=8 * 1024 ** 3)  # 8GB VRAM
        result = runner.invoke(app, ["recommend", "llama"])
        assert result.exit_code == 0
        assert "qlora" in result.output

    @patch("torch.cuda.is_available", return_value=False)
    @patch("transformers.AutoConfig.from_pretrained")
    def test_saves_yaml_to_file(self, mock_cfg, mock_cuda, tmp_path):
        mock_cfg.return_value = _mock_autoconfig(768, 12)
        out_file = tmp_path / "config.yaml"
        result = runner.invoke(app, ["recommend", "gpt2", "--output", str(out_file)])
        assert result.exit_code == 0
        assert out_file.exists()
        parsed = yaml.safe_load(out_file.read_text())
        assert "model" in parsed
        assert "training" in parsed

    @patch("torch.cuda.is_available", return_value=False)
    @patch("transformers.AutoConfig.from_pretrained")
    def test_output_config_has_required_sections(self, mock_cfg, mock_cuda, tmp_path):
        mock_cfg.return_value = _mock_autoconfig(768, 12)
        out_file = tmp_path / "config.yaml"
        runner.invoke(app, ["recommend", "gpt2", "--output", str(out_file)])
        parsed = yaml.safe_load(out_file.read_text())
        for section in ["model", "dataset", "tokenization", "training"]:
            assert section in parsed, f"Missing section: {section}"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("transformers.AutoConfig.from_pretrained")
    def test_output_config_is_valid_pipeline_config(self, mock_cfg, mock_cuda, tmp_path):
        mock_cfg.return_value = _mock_autoconfig(768, 12)
        out_file = tmp_path / "config.yaml"
        runner.invoke(app, ["recommend", "gpt2", "--output", str(out_file)])

        from lmtool.core.config import PipelineConfig
        with patch("pathlib.Path.exists", return_value=True):
            cfg = PipelineConfig.from_yaml(out_file)
        assert cfg.training is not None
        assert cfg.model.name == "gpt2"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("transformers.AutoConfig.from_pretrained", side_effect=Exception("network error"))
    def test_falls_back_gracefully_on_config_error(self, mock_cfg, mock_cuda):
        """Falls back to 124M estimate when AutoConfig.from_pretrained fails."""
        result = runner.invoke(app, ["recommend", "unknown-model"])
        assert result.exit_code == 0
        assert "method:" in result.output

    @patch("torch.cuda.is_available", return_value=False)
    @patch("transformers.AutoConfig.from_pretrained")
    def test_vram_flag_overrides_detection(self, mock_cfg, mock_cuda):
        mock_cfg.return_value = _mock_autoconfig(hidden=4096, layers=32)
        result = runner.invoke(app, ["recommend", "llama", "--vram", "24"])
        assert result.exit_code == 0
        assert "lora" in result.output
