"""
Unit tests for DPOTrainer.

All HF / TRL / PEFT ops are mocked — no GPU, no downloads, no trl install required.
"""

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from xlmtec.core.exceptions import DatasetError, TrainingError
from xlmtec.core.types import LoRAConfig, TrainingConfig, TrainingMethod
from xlmtec.trainers.dpo_trainer import DPOTrainer, validate_dpo_dataset

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def training_config(tmp_path):
    return TrainingConfig(
        method=TrainingMethod.DPO,
        output_dir=str(tmp_path / "dpo_out"),
        num_epochs=1,
        batch_size=2,
        learning_rate=5e-5,
        gradient_accumulation_steps=1,
        warmup_ratio=0.0,
        weight_decay=0.0,
        fp16=False,
        bf16=False,
        logging_steps=10,
        save_steps=None,
        save_strategy="epoch",
        evaluation_strategy="no",
        load_best_model_at_end=False,
        seed=42,
        gradient_checkpointing=False,
        lr_scheduler_type="linear",
    )


@pytest.fixture
def lora_config():
    return LoRAConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        fan_in_fan_out=False,
        init_lora_weights=True,
    )


@pytest.fixture
def dpo_dataset():
    """Minimal valid DPO dataset with required columns."""
    return Dataset.from_list(
        [
            {
                "prompt": "What is the capital of France?",
                "chosen": "The capital of France is Paris.",
                "rejected": "I don't know.",
            },
            {
                "prompt": "Explain gravity.",
                "chosen": "Gravity is a fundamental force that attracts objects with mass.",
                "rejected": "Gravity is magic.",
            },
        ]
    )


@pytest.fixture
def mock_model():
    m = MagicMock()
    param = MagicMock()
    param.numel.return_value = 1000
    param.requires_grad = True
    m.parameters.return_value = [param]
    return m


@pytest.fixture
def mock_tokenizer():
    t = MagicMock()
    t.pad_token = None
    t.eos_token = "<eos>"
    return t


# ============================================================================
# validate_dpo_dataset
# ============================================================================


class TestValidateDpoDataset:
    def test_valid_dataset_passes(self, dpo_dataset):
        """No exception raised for a valid dataset."""
        validate_dpo_dataset(dpo_dataset)  # should not raise

    def test_missing_chosen_raises(self):
        bad = Dataset.from_list([{"prompt": "q", "rejected": "bad"}])
        with pytest.raises(DatasetError, match="chosen"):
            validate_dpo_dataset(bad)

    def test_missing_rejected_raises(self):
        bad = Dataset.from_list([{"prompt": "q", "chosen": "good"}])
        with pytest.raises(DatasetError, match="rejected"):
            validate_dpo_dataset(bad)

    def test_missing_prompt_raises(self):
        bad = Dataset.from_list([{"chosen": "good", "rejected": "bad"}])
        with pytest.raises(DatasetError, match="prompt"):
            validate_dpo_dataset(bad)

    def test_missing_all_raises(self):
        bad = Dataset.from_list([{"text": "something"}])
        with pytest.raises(DatasetError):
            validate_dpo_dataset(bad)


# ============================================================================
# DPOTrainer
# ============================================================================


class TestDPOTrainer:
    def _make_trainer(self, mock_model, mock_tokenizer, training_config, lora_config):
        return DPOTrainer(mock_model, mock_tokenizer, training_config, lora_config)

    def test_init_stores_beta(self, mock_model, mock_tokenizer, training_config, lora_config):
        trainer = DPOTrainer(mock_model, mock_tokenizer, training_config, lora_config, beta=0.2)
        assert trainer.beta == 0.2

    def test_init_default_beta(self, mock_model, mock_tokenizer, training_config, lora_config):
        trainer = self._make_trainer(mock_model, mock_tokenizer, training_config, lora_config)
        assert trainer.beta == 0.1

    def test_train_raises_without_trl(
        self, mock_model, mock_tokenizer, training_config, lora_config, dpo_dataset
    ):
        """If trl is not installed, TrainingError is raised with install hint."""
        trainer = self._make_trainer(mock_model, mock_tokenizer, training_config, lora_config)
        with patch("builtins.__import__", side_effect=ImportError("No module named 'trl'")):
            with pytest.raises((TrainingError, ImportError)):
                trainer.train(dpo_dataset)

    def test_train_validates_columns(
        self, mock_model, mock_tokenizer, training_config, lora_config
    ):
        """Dataset missing required columns raises DatasetError before training starts."""
        bad_dataset = Dataset.from_list([{"text": "hello"}])
        trainer = self._make_trainer(mock_model, mock_tokenizer, training_config, lora_config)

        # Patch trl so the import succeeds — error should come from column validation
        mock_trl = MagicMock()
        with patch.dict("sys.modules", {"trl": mock_trl}):
            with pytest.raises(DatasetError):
                trainer.train(bad_dataset)

    def test_train_returns_training_result(
        self, mock_model, mock_tokenizer, training_config, lora_config, dpo_dataset, tmp_path
    ):
        """Happy path — mocked TRL DPOTrainer, returns valid TrainingResult."""
        trainer = self._make_trainer(mock_model, mock_tokenizer, training_config, lora_config)

        mock_train_output = MagicMock()
        mock_train_output.training_loss = 0.42
        mock_train_output.global_step = 10
        mock_train_output.metrics = {"epoch": 1}

        mock_state = MagicMock()
        mock_state.log_history = [{"train_loss": 0.42}]

        mock_trl_trainer_instance = MagicMock()
        mock_trl_trainer_instance.train.return_value = mock_train_output
        mock_trl_trainer_instance.state = mock_state

        mock_dpo_config = MagicMock()
        mock_trl_module = MagicMock()
        mock_trl_module.DPOTrainer.return_value = mock_trl_trainer_instance
        mock_trl_module.DPOConfig.return_value = mock_dpo_config

        with patch.dict("sys.modules", {"trl": mock_trl_module}):
            with patch("xlmtec.trainers.lora_trainer.get_peft_model", return_value=mock_model):
                with patch(
                    "xlmtec.trainers.lora_trainer.detect_target_modules", return_value=["q_proj"]
                ):
                    result = trainer.train(dpo_dataset)

        assert result.train_loss == pytest.approx(0.42)
        assert result.steps_completed == 10
        assert result.output_dir is not None

    def test_setup_peft_calls_get_peft_model(
        self, mock_model, mock_tokenizer, training_config, lora_config
    ):
        trainer = self._make_trainer(mock_model, mock_tokenizer, training_config, lora_config)
        with patch(
            "xlmtec.trainers.lora_trainer.get_peft_model", return_value=mock_model
        ) as mock_gpm:
            with patch(
                "xlmtec.trainers.lora_trainer.detect_target_modules", return_value=["q_proj"]
            ):
                trainer._setup_peft(mock_model)
        mock_gpm.assert_called_once()

    def test_factory_creates_dpo_trainer(
        self, mock_model, mock_tokenizer, training_config, lora_config
    ):
        """TrainerFactory.create() returns a DPOTrainer for method=dpo."""
        from xlmtec.trainers.factory import TrainerFactory

        trainer = TrainerFactory.create(
            model=mock_model,
            tokenizer=mock_tokenizer,
            training_config=training_config,
            lora_config=lora_config,
        )
        assert isinstance(trainer, DPOTrainer)

    def test_factory_dpo_requires_lora_config(self, mock_model, mock_tokenizer, training_config):
        """TrainerFactory.create() raises MissingConfigError if lora_config is None."""
        from xlmtec.core.exceptions import MissingConfigError
        from xlmtec.trainers.factory import TrainerFactory

        with pytest.raises(MissingConfigError):
            TrainerFactory.create(
                model=mock_model,
                tokenizer=mock_tokenizer,
                training_config=training_config,
                lora_config=None,
            )
