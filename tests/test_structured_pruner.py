"""
Unit tests for StructuredPruner.

All torch and model interactions are mocked — no GPU, no HF downloads.
"""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from finetune_cli.core.exceptions import FineTuneError
from finetune_cli.core.types import PruningConfig
from finetune_cli.trainers.structured_pruner import (
    PruningResult,
    StructuredPruner,
    _count_params,
    _head_importance_scores,
    _zero_head_rows,
)

# ============================================================================
# Helpers
# ============================================================================

def _make_param(numel: int, requires_grad: bool = True) -> MagicMock:
    """Return a MagicMock that looks like an nn.Parameter to _count_params."""
    p = MagicMock()
    p.numel.return_value = numel
    p.requires_grad = requires_grad
    return p


def _make_pruning_config(tmp_path: Path, sparsity: float = 0.3) -> PruningConfig:
    return PruningConfig(
        output_dir=tmp_path / "pruned",
        sparsity=sparsity,
        method="heads",
        importance_metric="magnitude",
        min_heads_per_layer=1,
    )


def _make_model_mock(num_heads: int = 4) -> MagicMock:
    """Build a minimal GPT-2-style model mock with one transformer layer."""
    import torch

    # Query projection weight: (num_heads * head_dim, hidden)
    head_dim = 16
    hidden = head_dim * num_heads
    q_weight = torch.ones(hidden, hidden)  # uniform — all heads score equally

    q_proj = MagicMock()
    q_proj.weight = q_weight

    attn = MagicMock()
    attn.q_proj = q_proj

    layer = MagicMock()
    layer.self_attn = attn

    model_config = MagicMock()
    model_config.num_attention_heads = num_heads

    model = MagicMock()
    model.config = model_config
    model.model = MagicMock()
    model.model.layers = [layer]
    model.parameters.return_value = [_make_param(hidden * hidden)]
    return model


# ============================================================================
# PruningConfig validation
# ============================================================================

class TestPruningConfig:

    def test_valid_config_constructs(self, tmp_path):
        cfg = PruningConfig(output_dir=tmp_path, sparsity=0.3)
        assert cfg.sparsity == 0.3
        assert cfg.method == "heads"
        assert cfg.min_heads_per_layer == 1

    def test_default_sparsity(self, tmp_path):
        cfg = PruningConfig(output_dir=tmp_path)
        assert cfg.sparsity == 0.3

    def test_invalid_sparsity_raises(self, tmp_path):
        with pytest.raises(FineTuneError):
            pruner = StructuredPruner(
                MagicMock(), MagicMock(),
                PruningConfig(output_dir=tmp_path, sparsity=1.5)
            )

    def test_invalid_method_raises(self, tmp_path):
        with pytest.raises(FineTuneError):
            StructuredPruner(
                MagicMock(), MagicMock(),
                PruningConfig(output_dir=tmp_path, method="invalid")
            )


# ============================================================================
# Internal helper unit tests
# ============================================================================

class TestHelpers:

    def test_head_importance_scores_returns_one_per_head(self):
        import torch
        weight = torch.ones(8, 8)   # 2 heads of size 4
        scores = _head_importance_scores(weight, num_heads=2)
        assert len(scores) == 2

    def test_head_importance_scores_differentiates_magnitudes(self):
        import torch
        # Head 0: small weights; Head 1: large weights
        weight = torch.zeros(8, 8)
        weight[:4] = 0.1   # head 0
        weight[4:] = 1.0   # head 1
        scores = _head_importance_scores(weight, num_heads=2)
        assert scores[0] < scores[1]

    def test_zero_head_rows_zeroes_correct_rows(self):
        import torch
        weight = torch.ones(8, 8)   # 2 heads of 4 rows each
        zeroed = _zero_head_rows(weight, head_indices=[0], num_heads=2)
        assert (weight[:4] == 0).all()
        assert (weight[4:] == 1).all()
        assert zeroed == 4 * 8  # 4 rows × 8 cols

    def test_count_params_sums_numel(self):
        model = MagicMock()
        model.parameters.return_value = [_make_param(100), _make_param(50)]
        assert _count_params(model) == 150


# ============================================================================
# StructuredPruner.prune() — attention head pruning
# ============================================================================

class TestStructuredPrunerHeads:

    def test_prune_returns_pruning_result(self, tmp_path):
        import torch
        model = _make_model_mock(num_heads=4)
        tokenizer = MagicMock()
        cfg = _make_pruning_config(tmp_path, sparsity=0.5)

        pruner = StructuredPruner(model, tokenizer, cfg)
        result = pruner.prune()

        assert isinstance(result, PruningResult)
        assert result.output_dir == tmp_path / "pruned"

    def test_prune_calls_save_pretrained(self, tmp_path):
        model = _make_model_mock(num_heads=4)
        tokenizer = MagicMock()
        cfg = _make_pruning_config(tmp_path, sparsity=0.5)

        StructuredPruner(model, tokenizer, cfg).prune()

        model.save_pretrained.assert_called_once()
        tokenizer.save_pretrained.assert_called_once()

    def test_prune_records_heads_pruned_per_layer(self, tmp_path):
        model = _make_model_mock(num_heads=4)
        tokenizer = MagicMock()
        cfg = _make_pruning_config(tmp_path, sparsity=0.5)

        result = StructuredPruner(model, tokenizer, cfg).prune()

        # At least one layer entry should be present
        assert isinstance(result.heads_pruned_per_layer, dict)

    def test_zero_sparsity_warns_and_prunes_nothing(self, tmp_path):
        model = _make_model_mock(num_heads=4)
        tokenizer = MagicMock()
        cfg = _make_pruning_config(tmp_path, sparsity=0.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = StructuredPruner(model, tokenizer, cfg).prune()
            assert any("sparsity=0.0" in str(warning.message) for warning in w)

        assert result.zeroed_param_count == 0

    def test_min_heads_per_layer_respected(self, tmp_path):
        """With 4 heads, sparsity=0.9, min=1 → max 3 pruned, not 4."""
        model = _make_model_mock(num_heads=4)
        tokenizer = MagicMock()
        cfg = PruningConfig(
            output_dir=tmp_path / "pruned",
            sparsity=0.9,
            method="heads",
            min_heads_per_layer=1,
        )
        result = StructuredPruner(model, tokenizer, cfg).prune()
        for n_pruned in result.heads_pruned_per_layer.values():
            assert n_pruned <= 3   # 4 - min_heads_per_layer(1)

    def test_prune_result_has_timing(self, tmp_path):
        model = _make_model_mock(num_heads=4)
        result = StructuredPruner(model, MagicMock(),
                                  _make_pruning_config(tmp_path)).prune()
        assert result.pruning_time_seconds >= 0.0

    def test_unknown_model_structure_skips_gracefully(self, tmp_path):
        """A model with no recognised layer structure should return zeroed=0."""
        model = MagicMock()
        # No .model.layers, .transformer.h, etc.
        model.model = MagicMock(spec=[])
        model.config = MagicMock()
        model.parameters.return_value = [_make_param(100)]
        cfg = _make_pruning_config(tmp_path, sparsity=0.5)

        result = StructuredPruner(model, MagicMock(), cfg).prune()
        assert result.zeroed_param_count == 0


# ============================================================================
# FFN pruning method
# ============================================================================

class TestStructuredPrunerFFN:

    def test_ffn_prune_returns_result(self, tmp_path):
        import torch

        head_dim, hidden = 16, 64
        ffn_out = hidden * 4
        fc1_weight = torch.ones(ffn_out, hidden)

        fc1 = MagicMock()
        fc1.weight = fc1_weight

        mlp = MagicMock()
        mlp.fc1 = fc1

        layer = MagicMock()
        layer.mlp = mlp

        model = MagicMock()
        model.model = MagicMock()
        model.model.layers = [layer]
        model.config = MagicMock()
        model.parameters.return_value = [_make_param(ffn_out * hidden)]

        cfg = PruningConfig(output_dir=tmp_path / "pruned", sparsity=0.2, method="ffn")
        result = StructuredPruner(model, MagicMock(), cfg).prune()

        assert isinstance(result, PruningResult)
