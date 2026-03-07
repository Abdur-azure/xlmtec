"""
Unit tests for FullFineTuner.

HuggingFace Trainer is mocked — no GPU required.
No real torch tensors — all params are MagicMock per lessons.md rule.
"""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from lmtool.core.types import TrainingConfig, TrainingMethod
from lmtool.trainers import TrainerFactory
from lmtool.trainers.base import TrainingResult
from lmtool.trainers.full_trainer import _VRAM_WARNING_THRESHOLD, FullFineTuner

# ============================================================================
# HELPERS
# ============================================================================

def _make_param(numel: int, requires_grad: bool = True) -> MagicMock:
    """Pure MagicMock parameter — no real tensors, no torch import required."""
    param = MagicMock()
    param.numel.return_value = numel
    param.requires_grad = requires_grad
    return param


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def full_training_config(tmp_path) -> TrainingConfig:
    return TrainingConfig(
        method=TrainingMethod.FULL_FINETUNING,
        output_dir=tmp_path / "output",
        num_epochs=1,
        batch_size=2,
    )


@pytest.fixture
def small_dataset() -> Dataset:
    return Dataset.from_dict({
        "input_ids": [[1, 2, 3, 4]] * 8,
        "attention_mask": [[1, 1, 1, 1]] * 8,
        "labels": [[1, 2, 3, 4]] * 8,
    })


# ============================================================================
# TESTS
# ============================================================================

class TestFullFineTuner:

    def test_all_params_trainable_after_setup(
        self, mock_model, mock_tokenizer, full_training_config
    ):
        params = [_make_param(100), _make_param(50)]
        mock_model.parameters.side_effect = lambda: iter(params)
        trainer = FullFineTuner(mock_model, mock_tokenizer, full_training_config)
        result_model = trainer._setup_peft(mock_model)
        assert result_model is mock_model

    def test_vram_warning_for_large_model(self, mock_tokenizer, full_training_config):
        big_param = _make_param(_VRAM_WARNING_THRESHOLD + 1)

        mock_model = MagicMock()
        mock_model.parameters.side_effect = lambda: iter([big_param])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FullFineTuner(mock_model, mock_tokenizer, full_training_config)
            resource_warnings = [x for x in w if issubclass(x.category, ResourceWarning)]
            assert len(resource_warnings) == 1
            assert "VRAM" in str(resource_warnings[0].message)

    def test_no_warning_for_small_model(self, mock_tokenizer, full_training_config):
        small_param = _make_param(100)

        mock_model = MagicMock()
        mock_model.parameters.side_effect = lambda: iter([small_param])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FullFineTuner(mock_model, mock_tokenizer, full_training_config)
            resource_warnings = [x for x in w if issubclass(x.category, ResourceWarning)]
            assert len(resource_warnings) == 0

    @patch("lmtool.trainers.base.BaseTrainer._build_hf_trainer")
    @patch("lmtool.trainers.base.BaseTrainer._build_training_args")
    def test_train_returns_result(
        self, mock_args, mock_hf_trainer, mock_model, mock_tokenizer,
        full_training_config, small_dataset
    ):
        # Pure MagicMock param — no torch required
        mock_model.parameters.side_effect = lambda: iter([_make_param(16)])

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = MagicMock(
            training_loss=0.5,
            global_step=10,
            metrics={"epoch": 1},
        )
        mock_trainer.state.log_history = []
        mock_hf_trainer.return_value = mock_trainer
        mock_args.return_value = MagicMock()

        trainer = FullFineTuner(mock_model, mock_tokenizer, full_training_config)
        result = trainer.train(small_dataset)

        assert isinstance(result, TrainingResult)
        assert result.output_dir == full_training_config.output_dir

    def test_factory_creates_full_finetuner(
        self, mock_model, mock_tokenizer, full_training_config
    ):
        # Pure MagicMock param — no torch required
        mock_model.parameters.side_effect = lambda: iter([_make_param(16)])

        trainer = TrainerFactory.create(
            model=mock_model,
            tokenizer=mock_tokenizer,
            training_config=full_training_config,
        )
        assert isinstance(trainer, FullFineTuner)
