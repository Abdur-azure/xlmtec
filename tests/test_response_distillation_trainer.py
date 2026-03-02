"""
Unit tests for ResponseDistillationTrainer.

Teacher loading and HuggingFace Trainer are mocked — no GPU, no downloads.
All imports are absolute per lessons.md.
"""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from datasets import Dataset, DatasetDict

from finetune_cli.core.types import TrainingConfig, DistillationConfig, TrainingMethod
from finetune_cli.core.exceptions import MissingConfigError, TrainingError
from finetune_cli.trainers.response_distillation_trainer import (
    ResponseDistillationTrainer,
    _VRAM_WARNING_THRESHOLD,
)
from finetune_cli.trainers.factory import TrainerFactory


# ============================================================================
# HELPERS
# ============================================================================


def _make_param(numel: int = 1_000_000, requires_grad: bool = True) -> MagicMock:
    """Pure MagicMock parameter — no real tensors (lessons.md rule)."""
    param = MagicMock()
    param.numel.return_value = numel
    param.requires_grad = requires_grad
    return param


def _mock_train_output() -> MagicMock:
    out = MagicMock()
    out.training_loss = 0.31
    out.global_step = 20
    out.metrics = {"epoch": 2}
    return out


def _mock_distillation_trainer() -> MagicMock:
    """Fake _DistillationTrainer (the inner HF Trainer subclass)."""
    hf_trainer = MagicMock()
    hf_trainer.train.return_value = _mock_train_output()
    hf_trainer.state.log_history = [
        {"loss": 0.45, "step": 10},
        {"eval_loss": 0.38, "step": 20},
    ]
    return hf_trainer


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def distillation_config() -> DistillationConfig:
    return DistillationConfig(
        teacher_model_name="gpt2-medium",
        temperature=2.0,
        alpha=0.5,
    )


@pytest.fixture
def training_config(tmp_output_dir) -> TrainingConfig:
    return TrainingConfig(
        method=TrainingMethod.VANILLA_DISTILLATION,
        output_dir=tmp_output_dir,
        num_epochs=2,
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
# FACTORY DISPATCH
# ============================================================================


class TestFactoryDispatch:
    """TrainerFactory correctly selects ResponseDistillationTrainer."""

    def test_creates_response_distillation_trainer(
        self, mock_model, mock_tokenizer, training_config, distillation_config
    ):
        trainer = TrainerFactory.create(
            model=mock_model,
            tokenizer=mock_tokenizer,
            training_config=training_config,
            distillation_config=distillation_config,
        )
        assert isinstance(trainer, ResponseDistillationTrainer)

    def test_missing_distillation_config_raises(
        self, mock_model, mock_tokenizer, training_config
    ):
        """VANILLA_DISTILLATION without distillation_config raises MissingConfigError."""
        with pytest.raises(MissingConfigError):
            TrainerFactory.create(
                model=mock_model,
                tokenizer=mock_tokenizer,
                training_config=training_config,
                distillation_config=None,
            )


# ============================================================================
# INIT / CONFIG STORED
# ============================================================================


class TestInit:
    """ResponseDistillationTrainer stores config and issues VRAM warning when needed."""

    def test_stores_distillation_config(
        self, mock_model, mock_tokenizer, training_config, distillation_config
    ):
        trainer = ResponseDistillationTrainer(
            mock_model, mock_tokenizer, training_config, distillation_config
        )
        assert trainer.distillation_config is distillation_config
        assert trainer.distillation_config.teacher_model_name == "gpt2-medium"
        assert trainer.distillation_config.temperature == 2.0
        assert trainer.distillation_config.alpha == 0.5

    def test_stores_training_config(
        self, mock_model, mock_tokenizer, training_config, distillation_config
    ):
        trainer = ResponseDistillationTrainer(
            mock_model, mock_tokenizer, training_config, distillation_config
        )
        assert trainer.training_config is training_config

    def test_vram_warning_for_large_model(
        self, mock_tokenizer, training_config, distillation_config
    ):
        """Models above threshold trigger a ResourceWarning."""
        big_param = _make_param(numel=_VRAM_WARNING_THRESHOLD + 1)
        big_model = MagicMock()
        big_model.parameters.side_effect = lambda: iter([big_param])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ResponseDistillationTrainer(
                big_model, mock_tokenizer, training_config, distillation_config
            )

        resource_warns = [w for w in caught if issubclass(w.category, ResourceWarning)]
        assert len(resource_warns) == 1
        assert "teacher" in str(resource_warns[0].message).lower()

    def test_no_vram_warning_for_small_model(
        self, mock_model, mock_tokenizer, training_config, distillation_config
    ):
        """Models below threshold do not trigger a warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ResponseDistillationTrainer(
                mock_model, mock_tokenizer, training_config, distillation_config
            )

        resource_warns = [w for w in caught if issubclass(w.category, ResourceWarning)]
        assert len(resource_warns) == 0


# ============================================================================
# _setup_peft — all params trainable
# ============================================================================


class TestSetupPeft:
    """_setup_peft enables requires_grad on all parameters."""

    def test_all_parameters_require_grad(
        self, mock_model, mock_tokenizer, training_config, distillation_config
    ):
        trainer = ResponseDistillationTrainer(
            mock_model, mock_tokenizer, training_config, distillation_config
        )
        returned_model = trainer._setup_peft(mock_model)
        # All params should have requires_grad set to True
        for param in returned_model.parameters():
            assert param.requires_grad is True

    def test_returns_model(
        self, mock_model, mock_tokenizer, training_config, distillation_config
    ):
        trainer = ResponseDistillationTrainer(
            mock_model, mock_tokenizer, training_config, distillation_config
        )
        returned = trainer._setup_peft(mock_model)
        assert returned is mock_model


# ============================================================================
# train() — full flow mocked
# ============================================================================

_DISTILLATION_TRAINER_PATH = (
    "finetune_cli.trainers.response_distillation_trainer._DistillationTrainer"
)
_AUTO_MODEL_PATH = (
    "finetune_cli.trainers.response_distillation_trainer.AutoModelForCausalLM"
)


class TestTrain:
    """train() orchestrates teacher load → distillation → save → TrainingResult."""

    @patch(_DISTILLATION_TRAINER_PATH)
    @patch(_AUTO_MODEL_PATH)
    def test_returns_training_result(
        self,
        mock_auto_model,
        mock_dist_trainer_cls,
        mock_model,
        mock_tokenizer,
        training_config,
        distillation_config,
        small_dataset,
    ):
        from finetune_cli.trainers.base import TrainingResult

        mock_auto_model.from_pretrained.return_value = MagicMock()
        mock_dist_trainer_cls.return_value = _mock_distillation_trainer()

        trainer = ResponseDistillationTrainer(
            mock_model, mock_tokenizer, training_config, distillation_config
        )
        result = trainer.train(small_dataset)

        assert isinstance(result, TrainingResult)
        assert result.train_loss == pytest.approx(0.31)
        assert result.steps_completed == 20
        assert result.epochs_completed == 2

    @patch(_DISTILLATION_TRAINER_PATH)
    @patch(_AUTO_MODEL_PATH)
    def test_eval_loss_extracted_from_logs(
        self,
        mock_auto_model,
        mock_dist_trainer_cls,
        mock_model,
        mock_tokenizer,
        training_config,
        distillation_config,
        dataset_dict,
    ):
        mock_auto_model.from_pretrained.return_value = MagicMock()
        mock_dist_trainer_cls.return_value = _mock_distillation_trainer()

        trainer = ResponseDistillationTrainer(
            mock_model, mock_tokenizer, training_config, distillation_config
        )
        result = trainer.train(dataset_dict)

        assert result.eval_loss == pytest.approx(0.38)

    @patch(_DISTILLATION_TRAINER_PATH)
    @patch(_AUTO_MODEL_PATH)
    def test_model_saved_to_output_dir(
        self,
        mock_auto_model,
        mock_dist_trainer_cls,
        mock_model,
        mock_tokenizer,
        training_config,
        distillation_config,
        small_dataset,
        tmp_output_dir,
    ):
        mock_auto_model.from_pretrained.return_value = MagicMock()
        hf_trainer = _mock_distillation_trainer()
        mock_dist_trainer_cls.return_value = hf_trainer

        trainer = ResponseDistillationTrainer(
            mock_model, mock_tokenizer, training_config, distillation_config
        )
        trainer.train(small_dataset)

        hf_trainer.save_model.assert_called_once_with(str(tmp_output_dir))

    @patch(_DISTILLATION_TRAINER_PATH)
    @patch(_AUTO_MODEL_PATH)
    def test_teacher_loaded_and_frozen(
        self,
        mock_auto_model,
        mock_dist_trainer_cls,
        mock_model,
        mock_tokenizer,
        training_config,
        distillation_config,
        small_dataset,
    ):
        mock_teacher = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_teacher
        mock_dist_trainer_cls.return_value = _mock_distillation_trainer()

        trainer = ResponseDistillationTrainer(
            mock_model, mock_tokenizer, training_config, distillation_config
        )
        trainer.train(small_dataset)

        mock_auto_model.from_pretrained.assert_called_once_with(
            distillation_config.teacher_model_name
        )
        mock_teacher.eval.assert_called_once()

    @patch(_AUTO_MODEL_PATH)
    def test_teacher_load_failure_raises_training_error(
        self,
        mock_auto_model,
        mock_model,
        mock_tokenizer,
        training_config,
        distillation_config,
        small_dataset,
    ):
        mock_auto_model.from_pretrained.side_effect = OSError("model not found")

        trainer = ResponseDistillationTrainer(
            mock_model, mock_tokenizer, training_config, distillation_config
        )
        with pytest.raises(TrainingError):
            trainer.train(small_dataset)

    @patch(_DISTILLATION_TRAINER_PATH)
    @patch(_AUTO_MODEL_PATH)
    def test_output_dir_in_result(
        self,
        mock_auto_model,
        mock_dist_trainer_cls,
        mock_model,
        mock_tokenizer,
        training_config,
        distillation_config,
        small_dataset,
        tmp_output_dir,
    ):
        mock_auto_model.from_pretrained.return_value = MagicMock()
        mock_dist_trainer_cls.return_value = _mock_distillation_trainer()

        trainer = ResponseDistillationTrainer(
            mock_model, mock_tokenizer, training_config, distillation_config
        )
        result = trainer.train(small_dataset)

        assert result.output_dir == tmp_output_dir