"""
Full fine-tuning trainer.

Trains ALL model parameters — no PEFT adapters.
Best for small models or when maximum adaptation is needed.
Issues a VRAM warning for models with >1B parameters.
"""

import warnings
from typing import Optional

from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.types import TrainingConfig
from .base import BaseTrainer


# Parameter count above which we warn about VRAM
_VRAM_WARNING_THRESHOLD = 1_000_000_000  # 1B


class FullFineTuner(BaseTrainer):
    """
    Trains every model parameter without PEFT adapters.

    All weights are updated on each step. Suitable for:
    - Small models (<1B params) where LoRA would limit capacity
    - Domain adaptation requiring deep weight changes
    - Cases where the adapter must be merged for deployment anyway

    Memory note: requires approximately 16× the model size in VRAM
    (weights + gradients + optimiser states in FP32).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_config: TrainingConfig,
    ):
        super().__init__(model, tokenizer, training_config)

        total = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Full fine-tuning — {total:,} total parameters")

        if total > _VRAM_WARNING_THRESHOLD:
            warnings.warn(
                f"Full fine-tuning a model with {total:,} parameters "
                f"(>{_VRAM_WARNING_THRESHOLD // 1_000_000_000}B). "
                "This requires significant VRAM. Consider LoRA or QLoRA instead.",
                ResourceWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # BaseTrainer hook — no PEFT, just unfreeze everything
    # ------------------------------------------------------------------

    def _setup_peft(self, model: PreTrainedModel) -> PreTrainedModel:
        """Ensure all parameters are trainable (no-op for full fine-tuning)."""
        for param in model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"Trainable parameters: {trainable:,} / {total:,} (100% — full fine-tuning)"
        )
        return model