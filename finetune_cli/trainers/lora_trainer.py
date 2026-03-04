"""
LoRA (Low-Rank Adaptation) trainer.

Applies PEFT LoRA adapters to the model, training only adapter
weights while the base model remains frozen.
"""

from typing import Optional

from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from ..core.types import TrainingConfig, LoRAConfig as LoRAConfigType
from ..models.loader import detect_target_modules
from .base import BaseTrainer


class LoRATrainer(BaseTrainer):
    """
    Trainer for LoRA fine-tuning.

    Attaches low-rank adapter matrices to the specified target modules.
    Only the adapter parameters are updated during training.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_config: TrainingConfig,
        lora_config: LoRAConfigType,
    ):
        super().__init__(model, tokenizer, training_config)
        self.lora_config = lora_config

        self.logger.info(
            f"LoRA config — r={lora_config.r}, alpha={lora_config.lora_alpha}, "
            f"dropout={lora_config.lora_dropout}, "
            f"target_modules={lora_config.target_modules or 'auto-detect'}"
        )

    # ------------------------------------------------------------------
    # BaseTrainer hook
    # ------------------------------------------------------------------

    def _setup_peft(self, model: PreTrainedModel) -> PreTrainedModel:
        """Attach LoRA adapters and freeze base model weights."""
        # Auto-detect target modules if not specified
        target_modules = self.lora_config.target_modules or detect_target_modules(model)
        self.logger.info(f"Target modules: {target_modules}")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=target_modules,
            bias=self.lora_config.bias,
            fan_in_fan_out=self.lora_config.fan_in_fan_out,
            init_lora_weights=self.lora_config.init_lora_weights,
        )

        peft_model = get_peft_model(model, peft_config)

        # Log trainable parameter count
        trainable, total = self._count_parameters(peft_model)
        self.logger.info(
            f"Trainable parameters: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

        return peft_model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_parameters(model):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total if total > 0 else 1