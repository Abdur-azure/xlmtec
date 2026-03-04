"""
Instruction fine-tuning trainer.

Extends LoRATrainer with alpaca-style dataset formatting.
Converts {"instruction": ..., "input": ..., "response": ...} rows
into a single formatted prompt string before tokenization.
"""

from typing import Optional

from datasets import Dataset, DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.types import TrainingConfig, LoRAConfig as LoRAConfigType
from .lora_trainer import LoRATrainer


# Default prompt template — matches the Stanford Alpaca format
_PROMPT_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{response}"
)

_PROMPT_WITHOUT_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}"
)


def format_instruction_dataset(
    dataset: Dataset,
    instruction_col: str = "instruction",
    input_col: str = "input",
    response_col: str = "response",
    text_col: str = "text",
) -> Dataset:
    """
    Reformat a dataset of instruction/response pairs into a single text column.

    Supports rows with or without an ``input`` field. Rows that already
    have a ``text`` column are returned unchanged.

    Args:
        dataset: Raw dataset with instruction/response columns.
        instruction_col: Column name for the instruction.
        input_col: Column name for the optional input context.
        response_col: Column name for the expected response.
        text_col: Output column name for the formatted prompt.

    Returns:
        Dataset with a single ``text`` column ready for tokenization.
    """
    # If already has a text column, don't reformat
    if text_col in dataset.column_names:
        return dataset

    cols = dataset.column_names
    has_instruction = instruction_col in cols
    has_response = response_col in cols

    if not has_instruction or not has_response:
        raise ValueError(
            f"Instruction dataset must have '{instruction_col}' and '{response_col}' columns. "
            f"Found: {cols}. "
            "Pass a pre-formatted dataset with a 'text' column to skip reformatting."
        )

    has_input = input_col in cols

    def _format(row):
        if has_input and row.get(input_col, "").strip():
            text = _PROMPT_WITH_INPUT.format(
                instruction=row[instruction_col],
                input=row[input_col],
                response=row[response_col],
            )
        else:
            text = _PROMPT_WITHOUT_INPUT.format(
                instruction=row[instruction_col],
                response=row[response_col],
            )
        return {text_col: text}

    # Map and keep only the text column
    formatted = dataset.map(_format, remove_columns=cols)
    return formatted


class InstructionTrainer(LoRATrainer):
    """
    LoRA trainer with alpaca-style instruction formatting.

    Accepts datasets with ``instruction`` / ``input`` / ``response`` columns
    and reformats them into a single prompt string before passing to the
    base LoRA training loop.

    Also accepts pre-formatted datasets that already have a ``text`` column
    — in that case no reformatting is applied.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_config: TrainingConfig,
        lora_config: LoRAConfigType,
        instruction_col: str = "instruction",
        input_col: str = "input",
        response_col: str = "response",
    ):
        super().__init__(model, tokenizer, training_config, lora_config)
        self.instruction_col = instruction_col
        self.input_col = input_col
        self.response_col = response_col

    # ------------------------------------------------------------------
    # Override train to reformat dataset before passing to base class
    # ------------------------------------------------------------------

    def train(self, dataset):
        """Reformat dataset then delegate to LoRATrainer.train."""
        if isinstance(dataset, DatasetDict):
            formatted = DatasetDict({
                split: self._maybe_format(ds)
                for split, ds in dataset.items()
            })
        else:
            formatted = self._maybe_format(dataset)

        return super().train(formatted)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_format(self, dataset: Dataset) -> Dataset:
        """Format if needed; skip if already tokenized or pre-formatted."""
        # Already tokenized — nothing to format
        if "input_ids" in dataset.column_names:
            self.logger.info("Dataset already tokenized — skipping reformatting.")
            return dataset

        # Already has a text column — no reformatting needed
        if "text" in dataset.column_names:
            self.logger.info("Dataset already has 'text' column — skipping reformatting.")
            return dataset

        self.logger.info(
            f"Formatting instruction dataset using columns: "
            f"'{self.instruction_col}', '{self.input_col}', '{self.response_col}'"
        )
        formatted = format_instruction_dataset(
            dataset,
            instruction_col=self.instruction_col,
            input_col=self.input_col,
            response_col=self.response_col,
        )
        self.logger.info(f"Formatted {len(formatted)} samples.")
        return formatted