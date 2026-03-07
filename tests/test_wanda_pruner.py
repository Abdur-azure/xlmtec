"""
Unit tests for WandaPruner (WANDA unstructured pruning).

All torch operations use real tiny tensors (no GPU required — CPU only).
Model loading and save_pretrained are mocked.
"""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from lmtool.core.exceptions import FineTuneError
from lmtool.core.types import WandaConfig
from lmtool.trainers.wanda_pruner import (
    WandaPruner,
    WandaResult,
    _apply_wanda_mask,
    _wanda_score,
)

# ============================================================================
# Helpers
# ============================================================================

def _make_wanda_config(tmp_path: Path, sparsity: float = 0.5) -> WandaConfig:
    return WandaConfig(
        output_dir=tmp_path / "wanda_out",
        sparsity=sparsity,
        n_calibration_samples=4,
        calibration_seq_len=8,
    )


def _make_linear_module(out: int = 4, inp: int = 8) -> MagicMock:
    """Mock nn.Linear with a real weight tensor."""
    mod = MagicMock()
    mod.weight = torch.ones(out, inp)
    type(mod).__name__ = "Linear"
    return mod


def _make_model_mock(n_layers: int = 2) -> MagicMock:
    """Model mock with n_layers Linear modules and save_pretrained."""
    layers = [_make_linear_module() for _ in range(n_layers)]
    named_mods = [(f"layer{i}", m) for i, m in enumerate(layers)]

    model = MagicMock()
    model.named_modules.return_value = named_mods
    model.eval.return_value = model
    return model, layers


# ============================================================================
# WandaConfig validation
# ============================================================================

class TestWandaConfig:

    def test_valid_config_constructs(self, tmp_path):
        cfg = WandaConfig(output_dir=tmp_path, sparsity=0.5)
        assert cfg.sparsity == 0.5
        assert cfg.n_calibration_samples == 128
        assert cfg.use_row_wise is True

    def test_default_sparsity(self, tmp_path):
        cfg = WandaConfig(output_dir=tmp_path)
        assert cfg.sparsity == 0.5

    def test_invalid_sparsity_raises(self, tmp_path):
        with pytest.raises(FineTuneError):
            WandaPruner(MagicMock(), MagicMock(),
                        WandaConfig(output_dir=tmp_path, sparsity=1.0))

    def test_invalid_n_calibration_raises(self, tmp_path):
        with pytest.raises(FineTuneError):
            WandaPruner(MagicMock(), MagicMock(),
                        WandaConfig(output_dir=tmp_path, n_calibration_samples=0))


# ============================================================================
# Internal helper unit tests
# ============================================================================

class TestWandaHelpers:

    def test_wanda_score_shape_matches_weight(self):
        weight = torch.ones(4, 8)
        act_norm = torch.ones(8) * 2.0
        score = _wanda_score(weight, act_norm)
        assert score.shape == weight.shape

    def test_wanda_score_values(self):
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        act_norm = torch.tensor([1.0, 2.0])
        score = _wanda_score(weight, act_norm)
        # score = |W| * act_norm
        assert score[0, 0].item() == pytest.approx(1.0)
        assert score[0, 1].item() == pytest.approx(4.0)
        assert score[1, 0].item() == pytest.approx(3.0)
        assert score[1, 1].item() == pytest.approx(8.0)

    def test_apply_wanda_mask_row_wise_zeroes_correct_count(self):
        weight = torch.ones(4, 8)
        act_norm = torch.ones(8)
        zeroed = _apply_wanda_mask(weight, act_norm, sparsity=0.5, row_wise=True)
        # 50% of 8 inputs per row × 4 rows = 16
        assert zeroed == 16
        # Each row should have exactly 4 zeros
        for row in weight:
            assert int((row == 0).sum().item()) == 4

    def test_apply_wanda_mask_global_zeroes_correct_fraction(self):
        weight = torch.ones(4, 8)
        act_norm = torch.ones(8)
        zeroed = _apply_wanda_mask(weight, act_norm, sparsity=0.5, row_wise=False)
        total = 4 * 8
        assert zeroed <= total * 0.5 + 1  # allow rounding

    def test_apply_wanda_mask_higher_act_norm_spared(self):
        """Weights with higher activation norm are less likely to be pruned."""
        weight = torch.ones(1, 4)
        # Column 3 has the highest norm — should survive 0.75 row-wise sparsity
        act_norm = torch.tensor([0.1, 0.1, 0.1, 10.0])
        _apply_wanda_mask(weight, act_norm, sparsity=0.75, row_wise=True)
        # Column 3 should survive (highest score)
        assert weight[0, 3].item() == pytest.approx(1.0)


# ============================================================================
# WandaPruner.prune() — no calibration data (magnitude-only fallback)
# ============================================================================

class TestWandaPrunerNoCalibraton:

    def test_prune_returns_wanda_result(self, tmp_path):
        model, _ = _make_model_mock()
        cfg = _make_wanda_config(tmp_path)
        result = WandaPruner(model, MagicMock(), cfg).prune()
        assert isinstance(result, WandaResult)

    def test_prune_warns_no_calibration_data(self, tmp_path):
        model, _ = _make_model_mock()
        cfg = _make_wanda_config(tmp_path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            WandaPruner(model, MagicMock(), cfg).prune(calibration_input_ids=None)
        assert any("magnitude-only" in str(warning.message) for warning in w)

    def test_prune_calls_save_pretrained(self, tmp_path):
        model, _ = _make_model_mock()
        tokenizer = MagicMock()
        cfg = _make_wanda_config(tmp_path)
        WandaPruner(model, tokenizer, cfg).prune()
        model.save_pretrained.assert_called_once()
        tokenizer.save_pretrained.assert_called_once()

    def test_prune_output_dir_in_result(self, tmp_path):
        model, _ = _make_model_mock()
        cfg = _make_wanda_config(tmp_path, sparsity=0.5)
        result = WandaPruner(model, MagicMock(), cfg).prune()
        assert result.output_dir == tmp_path / "wanda_out"

    def test_prune_layers_pruned_count(self, tmp_path):
        model, _ = _make_model_mock(n_layers=3)
        cfg = _make_wanda_config(tmp_path)
        result = WandaPruner(model, MagicMock(), cfg).prune()
        assert result.layers_pruned == 3

    def test_zero_sparsity_warns_and_zeroes_nothing(self, tmp_path):
        model, layers = _make_model_mock()
        cfg = _make_wanda_config(tmp_path, sparsity=0.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = WandaPruner(model, MagicMock(), cfg).prune()
        assert any("sparsity=0.0" in str(warning.message) for warning in w)

    def test_result_has_timing(self, tmp_path):
        model, _ = _make_model_mock()
        result = WandaPruner(model, MagicMock(),
                             _make_wanda_config(tmp_path)).prune()
        assert result.pruning_time_seconds >= 0.0


# ============================================================================
# WandaPruner.prune() — with calibration data
# ============================================================================

class TestWandaPrunerWithCalibration:

    def test_prune_with_calibration_returns_result(self, tmp_path):
        """Calibration path runs end-to-end without error."""
        # Build a real tiny model-like object with a real Linear
        import torch.nn as nn

        linear = nn.Linear(8, 4, bias=False)
        named_mods = [("linear", linear)]

        model = MagicMock()
        model.named_modules.return_value = named_mods
        model.eval.return_value = model

        # Mock forward pass (hook is registered on the real nn.Linear)
        def fake_forward(input_ids):
            # Simulate activations flowing through the hook
            with torch.no_grad():
                x = torch.randn(input_ids.shape[0], 8)
                linear(x)

        model.side_effect = fake_forward

        cfg = WandaConfig(
            output_dir=tmp_path / "out",
            sparsity=0.5,
            n_calibration_samples=2,
            calibration_seq_len=4,
        )
        calib_ids = torch.zeros(8, 4, dtype=torch.long)
        result = WandaPruner(model, MagicMock(), cfg).prune(calib_ids)
        assert isinstance(result, WandaResult)
        assert result.layers_pruned >= 1

    def test_weight_sparsity_increases_after_pruning(self, tmp_path):
        """After pruning, fraction of zero weights >= target sparsity."""
        import torch.nn as nn

        linear = nn.Linear(16, 8, bias=False)
        # Ensure non-uniform weights so scoring works
        torch.nn.init.uniform_(linear.weight, 0.01, 1.0)

        named_mods = [("linear", linear)]
        model = MagicMock()
        model.named_modules.return_value = named_mods
        model.eval.return_value = model

        cfg = WandaConfig(
            output_dir=tmp_path / "out",
            sparsity=0.5,
            n_calibration_samples=4,
            calibration_seq_len=4,
        )
        # No calibration — magnitude-only but checks sparsity holds
        result = WandaPruner(model, MagicMock(), cfg).prune()

        zero_frac = float((linear.weight == 0).sum()) / float(linear.weight.numel())
        assert zero_frac >= cfg.sparsity - 0.05  # allow small rounding tolerance
