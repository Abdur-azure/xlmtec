"""
QLoRA (Quantized LoRA) trainer.

Extends LoRATrainer by preparing the base model for 4-bit
quantized training via BitsAndBytes before attaching adapters.
"""

from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import prepare_model_for_kbit_training

from ..core.types import TrainingConfig, LoRAConfig as LoRAConfigType, ModelConfig
from .lora_trainer import LoRATrainer


class QLoRATrainer(LoRATrainer):
    """
    Trainer for QLoRA fine-tuning (4-bit quantized LoRA).

    The base model must have been loaded with ``load_in_4bit=True``
    (BitsAndBytes config) before passing it to this trainer. This
    class calls ``prepare_model_for_kbit_training`` and then
    delegates the rest of setup to ``LoRATrainer``.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_config: TrainingConfig,
        lora_config: LoRAConfigType,
        model_config: ModelConfig,
    ):
        super().__init__(model, tokenizer, training_config, lora_config)
        self.model_config = model_config

        if not model_config.load_in_4bit:
            self.logger.warning(
                "QLoRATrainer expects a 4-bit quantized model "
                "(load_in_4bit=True). Proceeding anyway, but results "
                "may not reflect true QLoRA behaviour."
            )

    # ------------------------------------------------------------------
    # Override hook — prepare for k-bit then attach LoRA
    # ------------------------------------------------------------------

    def _setup_peft(self, model: PreTrainedModel) -> PreTrainedModel:
        """Prepare model for k-bit training, then apply LoRA adapters."""
        self.logger.info("Preparing model for k-bit (4-bit QLoRA) training...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=self.training_config.gradient_checkpointing,
        )
        # Delegate LoRA adapter attachment to parent
        return super()._setup_peft(model)