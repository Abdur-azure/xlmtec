"""
Unit tests for QLoRATrainer.

All HuggingFace and PEFT calls are mocked — no GPU, no downloads.
"""

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset, DatasetDict

from xlmtec.core.exceptions import MissingConfigError
from xlmtec.core.types import LoRAConfig, ModelConfig, TrainingConfig, TrainingMethod
from xlmtec.trainers.factory import TrainerFactory
from xlmtec.trainers.qlora_trainer import QLoRATrainer

# ============================================================================
# FIXTURES (mock_model, mock_tokenizer, tmp_output_dir come from conftest.py)
# ============================================================================


@pytest.fixture
def training_config(tmp_output_dir) -> TrainingConfig:
    return TrainingConfig(
        method=TrainingMethod.QLORA,
        output_dir=tmp_output_dir,
        num_epochs=1,
        batch_size=2,
        learning_rate=2e-4,
    )


@pytest.fixture
def lora_config() -> LoRAConfig:
    return LoRAConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(name="gpt2", load_in_4bit=True)


@pytest.fixture
def model_config_no_4bit() -> ModelConfig:
    return ModelConfig(name="gpt2", load_in_4bit=False)


@pytest.fixture
def small_dataset() -> DatasetDict:
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4]] * 8,
            "attention_mask": [[1, 1, 1, 1]] * 8,
            "labels": [[1, 2, 3, 4]] * 8,
        }
    )
    return DatasetDict({"train": ds, "validation": ds})


# ============================================================================
# FACTORY DISPATCH
# ============================================================================


class TestFactoryDispatchesQLoRA:
    """TrainerFactory correctly selects QLoRATrainer."""

    def test_factory_creates_qlora_trainer(
        self, mock_model, mock_tokenizer, training_config, lora_config, model_config
    ):
        trainer = TrainerFactory.create(
            model=mock_model,
            tokenizer=mock_tokenizer,
            training_config=training_config,
            lora_config=lora_config,
            model_config=model_config,
        )
        assert isinstance(trainer, QLoRATrainer)

    def test_qlora_without_lora_config_raises(
        self, mock_model, mock_tokenizer, training_config, model_config
    ):
        with pytest.raises(MissingConfigError):
            TrainerFactory.create(
                model=mock_model,
                tokenizer=mock_tokenizer,
                training_config=training_config,
                lora_config=None,
                model_config=model_config,
            )

    def test_qlora_without_model_config_raises(
        self, mock_model, mock_tokenizer, training_config, lora_config
    ):
        with pytest.raises(MissingConfigError):
            TrainerFactory.create(
                model=mock_model,
                tokenizer=mock_tokenizer,
                training_config=training_config,
                lora_config=lora_config,
                model_config=None,
            )


# ============================================================================
# QLoRATrainer UNIT TESTS
# ============================================================================


class TestQLoRATrainerInit:
    """QLoRATrainer stores config and warns when model is not 4-bit."""

    def test_init_stores_model_config(
        self, mock_model, mock_tokenizer, training_config, lora_config, model_config
    ):
        trainer = QLoRATrainer(
            mock_model, mock_tokenizer, training_config, lora_config, model_config
        )
        assert trainer.model_config is model_config
        assert trainer.lora_config is lora_config

    def test_warns_when_not_4bit(
        self, mock_model, mock_tokenizer, training_config, lora_config, model_config_no_4bit, caplog
    ):
        import logging

        with caplog.at_level(logging.WARNING):
            QLoRATrainer(
                mock_model, mock_tokenizer, training_config, lora_config, model_config_no_4bit
            )
        assert any("4-bit" in r.message for r in caplog.records)

    def test_no_warning_when_4bit(
        self, mock_model, mock_tokenizer, training_config, lora_config, model_config, caplog
    ):
        import logging

        with caplog.at_level(logging.WARNING):
            QLoRATrainer(mock_model, mock_tokenizer, training_config, lora_config, model_config)
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("4-bit" in m for m in warning_msgs)


class TestQLoRASetupPeft:
    """_setup_peft calls prepare_model_for_kbit_training then LoRA setup."""

    @patch("xlmtec.trainers.qlora_trainer.prepare_model_for_kbit_training")
    @patch("xlmtec.trainers.lora_trainer.get_peft_model")
    @patch("xlmtec.trainers.lora_trainer.detect_target_modules", return_value=["q_proj"])
    def test_kbit_prep_called_before_lora(
        self,
        mock_detect,
        mock_get_peft,
        mock_kbit_prep,
        mock_model,
        mock_tokenizer,
        training_config,
        lora_config,
        model_config,
    ):
        prepared_model = MagicMock()
        prepared_model.parameters.return_value = iter([])
        mock_kbit_prep.return_value = prepared_model
        mock_get_peft.return_value = prepared_model

        trainer = QLoRATrainer(
            mock_model, mock_tokenizer, training_config, lora_config, model_config
        )
        trainer._setup_peft(mock_model)

        mock_kbit_prep.assert_called_once()
        mock_get_peft.assert_called_once()
        # kbit must be called before get_peft
        assert mock_kbit_prep.call_args[0][0] is mock_model

    @patch("xlmtec.trainers.qlora_trainer.prepare_model_for_kbit_training")
    @patch("xlmtec.trainers.lora_trainer.get_peft_model")
    @patch("xlmtec.trainers.lora_trainer.detect_target_modules", return_value=["q_proj"])
    def test_gradient_checkpointing_passed_to_kbit(
        self,
        mock_detect,
        mock_get_peft,
        mock_kbit_prep,
        mock_model,
        mock_tokenizer,
        lora_config,
        model_config,
        tmp_output_dir,
    ):
        cfg = TrainingConfig(
            method=TrainingMethod.QLORA,
            output_dir=tmp_output_dir,
            gradient_checkpointing=True,
        )
        prepared = MagicMock()
        prepared.parameters.return_value = iter([])
        mock_kbit_prep.return_value = prepared
        mock_get_peft.return_value = prepared

        trainer = QLoRATrainer(mock_model, mock_tokenizer, cfg, lora_config, model_config)
        trainer._setup_peft(mock_model)

        _, kwargs = mock_kbit_prep.call_args
        assert kwargs.get("use_gradient_checkpointing") is True


# ============================================================================
# FULL TRAIN INTEGRATION (mocked HF Trainer)
# ============================================================================


class TestQLoRATrainerTrain:
    """End-to-end train() call with all HF components mocked."""

    @patch("xlmtec.trainers.qlora_trainer.prepare_model_for_kbit_training")
    @patch("xlmtec.trainers.lora_trainer.get_peft_model")
    @patch("xlmtec.trainers.lora_trainer.detect_target_modules", return_value=["q_proj"])
    def test_train_returns_result(
        self,
        mock_detect,
        mock_get_peft,
        mock_kbit_prep,
        mock_model,
        mock_tokenizer,
        training_config,
        lora_config,
        model_config,
        small_dataset,
        tmp_output_dir,
    ):
        peft_model = MagicMock()
        peft_model.parameters.return_value = iter([])
        mock_kbit_prep.return_value = peft_model
        mock_get_peft.return_value = peft_model

        # Trainer is lazy-imported inside _build_hf_trainer — patch the method directly
        hf_instance = MagicMock()
        hf_instance.train.return_value = MagicMock(
            training_loss=0.42,
            metrics={"epoch": 1},
            global_step=10,
        )
        hf_instance.state.log_history = [{"train_loss": 0.42}]

        trainer = QLoRATrainer(
            mock_model, mock_tokenizer, training_config, lora_config, model_config
        )

        with patch.object(trainer, "_build_hf_trainer", return_value=hf_instance):
            result = trainer.train(small_dataset)

        assert result is not None
        assert result.output_dir == tmp_output_dir
        hf_instance.train.assert_called_once()
