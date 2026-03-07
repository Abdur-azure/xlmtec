"""
Unit tests for FeatureDistillationTrainer.

Teacher loading, hidden-state extraction, and HF Trainer are all mocked.
No GPU, no real model downloads. Absolute imports per lessons.md.
"""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset, DatasetDict

from lmtool.core.exceptions import MissingConfigError, TrainingError
from lmtool.core.types import (
    FeatureDistillationConfig,
    TrainingConfig,
    TrainingMethod,
)
from lmtool.trainers.factory import TrainerFactory
from lmtool.trainers.feature_distillation_trainer import (
    _VRAM_WARNING_THRESHOLD,
    FeatureDistillationTrainer,
    _map_teacher_layer,
    _select_layers,
)

# ============================================================================
# HELPERS
# ============================================================================


def _make_param(numel: int = 1_000_000, requires_grad: bool = True) -> MagicMock:
    param = MagicMock()
    param.numel.return_value = numel
    param.requires_grad = requires_grad
    return param


def _mock_train_output() -> MagicMock:
    out = MagicMock()
    out.training_loss = 0.28
    out.global_step = 30
    out.metrics = {"epoch": 3}
    return out


def _mock_fd_trainer() -> MagicMock:
    hf_trainer = MagicMock()
    hf_trainer.train.return_value = _mock_train_output()
    hf_trainer.state.log_history = [
        {"loss": 0.40, "step": 15},
        {"eval_loss": 0.33, "step": 30},
    ]
    return hf_trainer


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def fd_config() -> FeatureDistillationConfig:
    return FeatureDistillationConfig(
        teacher_model_name="gpt2-medium",
        temperature=2.0,
        alpha=0.3,
        beta=0.2,
        feature_layers=None,
        feature_loss_weight=1.0,
    )


@pytest.fixture
def training_config(tmp_output_dir) -> TrainingConfig:
    return TrainingConfig(
        method=TrainingMethod.FEATURE_DISTILLATION,
        output_dir=tmp_output_dir,
        num_epochs=3,
        batch_size=2,
        learning_rate=1e-4,
    )


@pytest.fixture
def small_dataset() -> Dataset:
    return Dataset.from_dict({
        "input_ids": [[1, 2, 3, 4]] * 8,
        "attention_mask": [[1, 1, 1, 1]] * 8,
        "labels": [[1, 2, 3, 4]] * 8,
    })


@pytest.fixture
def dataset_dict(small_dataset) -> DatasetDict:
    return DatasetDict({"train": small_dataset, "validation": small_dataset})


# ============================================================================
# UNIT: _select_layers
# ============================================================================


class TestSelectLayers:
    def test_explicit_layers_returned(self):
        assert _select_layers(12, [0, 4, 8]) == [0, 4, 8]

    def test_out_of_range_layers_filtered(self):
        result = _select_layers(6, [0, 5, 6, 99])
        assert result == [0, 5]

    def test_none_returns_auto_evenly_spaced(self):
        result = _select_layers(12, None)
        assert len(result) <= 4
        assert all(0 <= i < 12 for i in result)

    def test_auto_fewer_layers_than_4(self):
        result = _select_layers(2, None)
        assert len(result) <= 2

    def test_all_invalid_raises(self):
        with pytest.raises(ValueError):
            _select_layers(4, [10, 20, 30])


# ============================================================================
# UNIT: _map_teacher_layer
# ============================================================================


class TestMapTeacherLayer:
    def test_same_depth_maps_identity(self):
        assert _map_teacher_layer(0, 12, 12) == 0
        assert _map_teacher_layer(6, 12, 12) == 6

    def test_teacher_deeper_maps_proportionally(self):
        # student layer 0 → teacher layer 0
        assert _map_teacher_layer(0, 24, 12) == 0
        # student layer 6 → teacher layer 12
        assert _map_teacher_layer(6, 24, 12) == 12

    def test_clamps_to_teacher_max(self):
        result = _map_teacher_layer(100, 12, 12)
        assert result <= 11


# ============================================================================
# FACTORY DISPATCH
# ============================================================================


class TestFactoryDispatch:
    def test_creates_feature_distillation_trainer(
        self, mock_model, mock_tokenizer, training_config, fd_config
    ):
        trainer = TrainerFactory.create(
            model=mock_model,
            tokenizer=mock_tokenizer,
            training_config=training_config,
            feature_distillation_config=fd_config,
        )
        assert isinstance(trainer, FeatureDistillationTrainer)

    def test_missing_config_raises(
        self, mock_model, mock_tokenizer, training_config
    ):
        with pytest.raises(MissingConfigError):
            TrainerFactory.create(
                model=mock_model,
                tokenizer=mock_tokenizer,
                training_config=training_config,
                feature_distillation_config=None,
            )


# ============================================================================
# INIT
# ============================================================================


class TestInit:
    def test_stores_fd_config(
        self, mock_model, mock_tokenizer, training_config, fd_config
    ):
        trainer = FeatureDistillationTrainer(
            mock_model, mock_tokenizer, training_config, fd_config
        )
        assert trainer.fd_config is fd_config

    def test_vram_warning_large_model(
        self, mock_tokenizer, training_config, fd_config
    ):
        big_param = _make_param(numel=_VRAM_WARNING_THRESHOLD + 1)
        big_model = MagicMock()
        big_model.parameters.side_effect = lambda: iter([big_param])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            FeatureDistillationTrainer(
                big_model, mock_tokenizer, training_config, fd_config
            )

        resource_warns = [w for w in caught if issubclass(w.category, ResourceWarning)]
        assert len(resource_warns) == 1

    def test_no_vram_warning_small_model(
        self, mock_model, mock_tokenizer, training_config, fd_config
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            FeatureDistillationTrainer(
                mock_model, mock_tokenizer, training_config, fd_config
            )

        resource_warns = [w for w in caught if issubclass(w.category, ResourceWarning)]
        assert len(resource_warns) == 0


# ============================================================================
# _setup_peft
# ============================================================================


class TestSetupPeft:
    def test_all_params_trainable(
        self, mock_model, mock_tokenizer, training_config, fd_config
    ):
        trainer = FeatureDistillationTrainer(
            mock_model, mock_tokenizer, training_config, fd_config
        )
        returned = trainer._setup_peft(mock_model)
        for p in returned.parameters():
            assert p.requires_grad is True

    def test_returns_same_model(
        self, mock_model, mock_tokenizer, training_config, fd_config
    ):
        trainer = FeatureDistillationTrainer(
            mock_model, mock_tokenizer, training_config, fd_config
        )
        assert trainer._setup_peft(mock_model) is mock_model


# ============================================================================
# train()
# ============================================================================

_FD_TRAINER_PATH = (
    "lmtool.trainers.feature_distillation_trainer._FeatureDistillationTrainer"
)
_AUTO_MODEL_PATH = (
    "lmtool.trainers.feature_distillation_trainer.AutoModelForCausalLM"
)


def _mock_teacher():
    t = MagicMock()
    t.config.num_hidden_layers = 24
    t.to.return_value = t   # .to(device) must return the same mock
    return t


class TestTrain:
    @patch(_FD_TRAINER_PATH)
    @patch(_AUTO_MODEL_PATH)
    def test_returns_training_result(
        self,
        mock_auto,
        mock_trainer_cls,
        mock_model,
        mock_tokenizer,
        training_config,
        fd_config,
        small_dataset,
    ):
        from lmtool.trainers.base import TrainingResult

        mock_auto.from_pretrained.return_value = _mock_teacher()
        mock_trainer_cls.return_value = _mock_fd_trainer()
        mock_model.config.num_hidden_layers = 12

        trainer = FeatureDistillationTrainer(
            mock_model, mock_tokenizer, training_config, fd_config
        )
        result = trainer.train(small_dataset)

        assert isinstance(result, TrainingResult)
        assert result.train_loss == pytest.approx(0.28)
        assert result.steps_completed == 30
        assert result.epochs_completed == 3

    @patch(_FD_TRAINER_PATH)
    @patch(_AUTO_MODEL_PATH)
    def test_eval_loss_extracted(
        self,
        mock_auto,
        mock_trainer_cls,
        mock_model,
        mock_tokenizer,
        training_config,
        fd_config,
        dataset_dict,
    ):
        mock_auto.from_pretrained.return_value = _mock_teacher()
        mock_trainer_cls.return_value = _mock_fd_trainer()
        mock_model.config.num_hidden_layers = 12

        trainer = FeatureDistillationTrainer(
            mock_model, mock_tokenizer, training_config, fd_config
        )
        result = trainer.train(dataset_dict)
        assert result.eval_loss == pytest.approx(0.33)

    @patch(_FD_TRAINER_PATH)
    @patch(_AUTO_MODEL_PATH)
    def test_model_saved_to_output_dir(
        self,
        mock_auto,
        mock_trainer_cls,
        mock_model,
        mock_tokenizer,
        training_config,
        fd_config,
        small_dataset,
        tmp_output_dir,
    ):
        mock_auto.from_pretrained.return_value = _mock_teacher()
        hf_trainer = _mock_fd_trainer()
        mock_trainer_cls.return_value = hf_trainer
        mock_model.config.num_hidden_layers = 12

        trainer = FeatureDistillationTrainer(
            mock_model, mock_tokenizer, training_config, fd_config
        )
        trainer.train(small_dataset)
        hf_trainer.save_model.assert_called_once_with(str(tmp_output_dir))

    @patch(_FD_TRAINER_PATH)
    @patch(_AUTO_MODEL_PATH)
    def test_teacher_loaded_and_frozen(
        self,
        mock_auto,
        mock_trainer_cls,
        mock_model,
        mock_tokenizer,
        training_config,
        fd_config,
        small_dataset,
    ):
        mock_teacher = _mock_teacher()
        mock_auto.from_pretrained.return_value = mock_teacher
        mock_trainer_cls.return_value = _mock_fd_trainer()
        mock_model.config.num_hidden_layers = 12

        trainer = FeatureDistillationTrainer(
            mock_model, mock_tokenizer, training_config, fd_config
        )
        trainer.train(small_dataset)

        mock_auto.from_pretrained.assert_called_once_with(
            fd_config.teacher_model_name,
            output_hidden_states=True,
        )
        mock_teacher.eval.assert_called_once()

    @patch(_AUTO_MODEL_PATH)
    def test_teacher_load_failure_raises_training_error(
        self,
        mock_auto,
        mock_model,
        mock_tokenizer,
        training_config,
        fd_config,
        small_dataset,
    ):
        mock_auto.from_pretrained.side_effect = OSError("not found")
        mock_model.config.num_hidden_layers = 12

        trainer = FeatureDistillationTrainer(
            mock_model, mock_tokenizer, training_config, fd_config
        )
        with pytest.raises(TrainingError):
            trainer.train(small_dataset)

    @patch(_FD_TRAINER_PATH)
    @patch(_AUTO_MODEL_PATH)
    def test_output_dir_in_result(
        self,
        mock_auto,
        mock_trainer_cls,
        mock_model,
        mock_tokenizer,
        training_config,
        fd_config,
        small_dataset,
        tmp_output_dir,
    ):
        mock_auto.from_pretrained.return_value = _mock_teacher()
        mock_trainer_cls.return_value = _mock_fd_trainer()
        mock_model.config.num_hidden_layers = 12

        trainer = FeatureDistillationTrainer(
            mock_model, mock_tokenizer, training_config, fd_config
        )
        result = trainer.train(small_dataset)
        assert result.output_dir == tmp_output_dir
