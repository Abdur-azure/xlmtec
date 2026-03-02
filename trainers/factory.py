"""
Trainer factory — selects and constructs the right trainer for a given method.

Usage::

    result = TrainerFactory.train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_config=config.training.to_config(),
        lora_config=config.lora.to_config(),
    )
"""

from typing import Optional
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.types import (
    TrainingConfig,
    LoRAConfig,
    ModelConfig,
    DistillationConfig,
    TrainingMethod,
)
from ..core.exceptions import MissingConfigError
from .base import BaseTrainer, TrainingResult
from .lora_trainer import LoRATrainer
from .qlora_trainer import QLoRATrainer
from .full_trainer import FullFineTuner
from .instruction_trainer import InstructionTrainer
from .dpo_trainer import DPOTrainer, validate_dpo_dataset
from .response_distillation_trainer import ResponseDistillationTrainer


# ============================================================================
# FACTORY
# ============================================================================


class TrainerFactory:
    """
    Creates the appropriate trainer for a given ``TrainingMethod``.

    Validates that method-specific config objects are present before
    constructing the trainer.
    """

    @staticmethod
    def create(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_config: TrainingConfig,
        lora_config: Optional[LoRAConfig] = None,
        model_config: Optional[ModelConfig] = None,
        distillation_config: Optional[DistillationConfig] = None,
    ) -> BaseTrainer:
        """
        Instantiate the correct trainer.

        Args:
            model: The (possibly quantized) model to fine-tune.
            tokenizer: Corresponding tokenizer.
            training_config: Core training hyper-parameters.
            lora_config: Required for LORA / QLORA / INSTRUCTION_TUNING / DPO.
            model_config: Required for QLORA (to check quantization flags).
            distillation_config: Required for VANILLA_DISTILLATION.

        Returns:
            A configured ``BaseTrainer`` subclass instance.

        Raises:
            MissingConfigError: If a required config object is absent.
            NotImplementedError: If the method is not yet supported.
        """
        method = training_config.method

        if method == TrainingMethod.LORA:
            if lora_config is None:
                raise MissingConfigError("lora_config", "LoRA training")
            return LoRATrainer(model, tokenizer, training_config, lora_config)

        if method == TrainingMethod.QLORA:
            if lora_config is None:
                raise MissingConfigError("lora_config", "QLoRA training")
            if model_config is None:
                raise MissingConfigError("model_config", "QLoRA training")
            return QLoRATrainer(model, tokenizer, training_config, lora_config, model_config)

        if method == TrainingMethod.FULL_FINETUNING:
            return FullFineTuner(model, tokenizer, training_config)

        if method == TrainingMethod.INSTRUCTION_TUNING:
            if lora_config is None:
                raise MissingConfigError("lora_config", "instruction tuning")
            return InstructionTrainer(model, tokenizer, training_config, lora_config)

        if method == TrainingMethod.DPO:
            if lora_config is None:
                raise MissingConfigError("lora_config", "DPO training")
            return DPOTrainer(model, tokenizer, training_config, lora_config)

        if method == TrainingMethod.VANILLA_DISTILLATION:
            if distillation_config is None:
                raise MissingConfigError("distillation_config", "response distillation")
            return ResponseDistillationTrainer(
                model, tokenizer, training_config, distillation_config
            )

        raise NotImplementedError(
            f"Training method '{method.value}' is not yet implemented. "
            f"Supported methods: lora, qlora, full_finetuning, "
            f"instruction_tuning, dpo, vanilla_distillation."
        )

    @staticmethod
    def train(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset | DatasetDict,
        training_config: TrainingConfig,
        lora_config: Optional[LoRAConfig] = None,
        model_config: Optional[ModelConfig] = None,
        distillation_config: Optional[DistillationConfig] = None,
    ) -> TrainingResult:
        """
        One-call convenience: create trainer and run training.

        Args:
            model: Model to fine-tune.
            tokenizer: Tokenizer.
            dataset: Train dataset or DatasetDict with 'train'/'validation'.
            training_config: Training hyper-parameters.
            lora_config: Required for LoRA / QLoRA / Instruction / DPO.
            model_config: Required for QLoRA.
            distillation_config: Required for vanilla distillation.

        Returns:
            ``TrainingResult`` with metrics and saved model path.
        """
        trainer = TrainerFactory.create(
            model=model,
            tokenizer=tokenizer,
            training_config=training_config,
            lora_config=lora_config,
            model_config=model_config,
            distillation_config=distillation_config,
        )
        return trainer.train(dataset)