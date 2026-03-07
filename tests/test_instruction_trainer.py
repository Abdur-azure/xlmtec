"""
Unit tests for InstructionTrainer and format_instruction_dataset.

No GPU required — training is mocked.
"""

import pytest
from datasets import Dataset, DatasetDict

from lmtool.core.exceptions import MissingConfigError
from lmtool.core.types import LoRAConfig, TrainingConfig, TrainingMethod
from lmtool.trainers import TrainerFactory
from lmtool.trainers.instruction_trainer import InstructionTrainer, format_instruction_dataset

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def instruction_dataset() -> Dataset:
    return Dataset.from_dict({
        "instruction": ["Translate to French", "Summarize this text", "Answer the question"],
        "input": ["Hello world", "", "What is AI?"],
        "response": ["Bonjour le monde", "Short summary.", "AI is artificial intelligence."],
    })


@pytest.fixture
def instruction_with_no_input() -> Dataset:
    return Dataset.from_dict({
        "instruction": ["Write a poem about the ocean", "Explain quantum computing"],
        "input": ["", ""],
        "response": ["Waves crash...", "Quantum computing uses qubits..."],
    })


@pytest.fixture
def pre_formatted_dataset() -> Dataset:
    return Dataset.from_dict({
        "text": ["### Instruction:\nDo something\n\n### Response:\nDone."] * 3
    })


@pytest.fixture
def instruction_training_config(tmp_path) -> TrainingConfig:
    return TrainingConfig(
        method=TrainingMethod.INSTRUCTION_TUNING,
        output_dir=tmp_path / "output",
        num_epochs=1,
        batch_size=2,
    )


@pytest.fixture
def lora_config() -> LoRAConfig:
    return LoRAConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])


# ============================================================================
# format_instruction_dataset TESTS
# ============================================================================

class TestFormatInstructionDataset:

    def test_produces_text_column(self, instruction_dataset):
        result = format_instruction_dataset(instruction_dataset)
        assert "text" in result.column_names

    def test_removes_original_columns(self, instruction_dataset):
        result = format_instruction_dataset(instruction_dataset)
        assert result.column_names == ["text"]

    def test_with_input_uses_input_template(self, instruction_dataset):
        result = format_instruction_dataset(instruction_dataset)
        first = result[0]["text"]
        assert "### Input:" in first
        assert "Hello world" in first

    def test_without_input_skips_input_section(self, instruction_with_no_input):
        result = format_instruction_dataset(instruction_with_no_input)
        for row in result:
            assert "### Input:" not in row["text"]

    def test_instruction_and_response_present(self, instruction_dataset):
        result = format_instruction_dataset(instruction_dataset)
        for row in result:
            assert "### Instruction:" in row["text"]
            assert "### Response:" in row["text"]

    def test_preserves_sample_count(self, instruction_dataset):
        result = format_instruction_dataset(instruction_dataset)
        assert len(result) == len(instruction_dataset)

    def test_pre_formatted_dataset_unchanged(self, pre_formatted_dataset):
        result = format_instruction_dataset(pre_formatted_dataset)
        assert result is pre_formatted_dataset

    def test_missing_columns_raises(self):
        bad = Dataset.from_dict({"question": ["q"], "answer": ["a"]})
        with pytest.raises(ValueError, match="instruction"):
            format_instruction_dataset(bad)


# ============================================================================
# InstructionTrainer TESTS
# ============================================================================

class TestInstructionTrainer:

    def test_formats_dataset_before_training(
        self, mock_model, mock_tokenizer,
        instruction_training_config, lora_config, instruction_dataset
    ):
        trainer = InstructionTrainer(
            mock_model, mock_tokenizer,
            instruction_training_config, lora_config
        )
        formatted = trainer._maybe_format(instruction_dataset)
        assert "text" in formatted.column_names
        assert "instruction" not in formatted.column_names

    def test_skips_formatting_when_text_present(
        self, mock_model, mock_tokenizer,
        instruction_training_config, lora_config, pre_formatted_dataset
    ):
        trainer = InstructionTrainer(
            mock_model, mock_tokenizer,
            instruction_training_config, lora_config
        )
        result = trainer._maybe_format(pre_formatted_dataset)
        assert result is pre_formatted_dataset

    def test_formats_dataset_dict(
        self, mock_model, mock_tokenizer,
        instruction_training_config, lora_config, instruction_dataset
    ):
        trainer = InstructionTrainer(
            mock_model, mock_tokenizer,
            instruction_training_config, lora_config
        )
        ds_dict = DatasetDict({"train": instruction_dataset, "validation": instruction_dataset})
        import unittest.mock as mock
        with mock.patch.object(type(trainer).__bases__[0], "train", return_value=None):
            trainer.train(ds_dict)

    def test_factory_creates_instruction_trainer(
        self, mock_model, mock_tokenizer,
        instruction_training_config, lora_config
    ):
        trainer = TrainerFactory.create(
            model=mock_model,
            tokenizer=mock_tokenizer,
            training_config=instruction_training_config,
            lora_config=lora_config,
        )
        assert isinstance(trainer, InstructionTrainer)

    def test_factory_requires_lora_config(
        self, mock_model, mock_tokenizer, instruction_training_config
    ):
        with pytest.raises(MissingConfigError):
            TrainerFactory.create(
                model=mock_model,
                tokenizer=mock_tokenizer,
                training_config=instruction_training_config,
                lora_config=None,
            )
