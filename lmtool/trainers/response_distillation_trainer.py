"""
Response Distillation trainer (Vanilla KD).

The student model learns to mimic the teacher's output distribution via
KL-divergence loss, blended with the standard cross-entropy loss on
ground-truth labels.

Loss formula:
    L = alpha * KL(softmax(teacher/T) || softmax(student/T)) * T²
        + (1 - alpha) * CE(student, labels)

where T is the temperature and alpha controls the distillation weight.

No PEFT adapters are attached — all student parameters are updated.
If you want a parameter-efficient student, run QLoRA or LoRA first,
then distil into the resulting adapter.

Reference: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015).
"""

import time
import warnings
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from ..core.exceptions import MissingConfigError, TrainingError
from ..core.types import DistillationConfig, TrainingConfig
from ..utils.logging import get_logger
from .base import BaseTrainer, TrainingResult

# Parameter count above which we warn about VRAM
_VRAM_WARNING_THRESHOLD = 1_000_000_000  # 1B


class ResponseDistillationTrainer(BaseTrainer):
    """
    Knowledge distillation trainer using KL divergence on output logits.

    Loads the teacher model internally on ``train()`` so the caller only
    needs to supply the *student* model (plus tokenizer and configs).

    Args:
        model: Student model to train.
        tokenizer: Shared tokenizer (teacher and student must use the same vocab).
        training_config: Core training hyper-parameters.
        distillation_config: Distillation-specific settings (teacher name,
            temperature, alpha).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_config: TrainingConfig,
        distillation_config: DistillationConfig,
    ):
        super().__init__(model, tokenizer, training_config)
        self.distillation_config = distillation_config

        total = sum(p.numel() for p in model.parameters())
        if total > _VRAM_WARNING_THRESHOLD:
            warnings.warn(
                f"Response distillation with a {total:,}-parameter student model "
                f"requires holding both student AND teacher in memory simultaneously. "
                "Ensure sufficient VRAM or use gradient checkpointing.",
                ResourceWarning,
                stacklevel=2,
            )

        self.logger.info(
            f"ResponseDistillationTrainer — teacher: {distillation_config.teacher_model_name}, "
            f"T={distillation_config.temperature}, alpha={distillation_config.alpha}"
        )

    # ------------------------------------------------------------------
    # BaseTrainer hook — no PEFT for response distillation
    # ------------------------------------------------------------------

    def _setup_peft(self, model: PreTrainedModel) -> PreTrainedModel:
        """No PEFT adapters — all parameters are trainable."""
        for param in model.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"Student trainable parameters: {trainable:,} / {total:,} "
            f"(100% — full student update)"
        )
        return model

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        dataset: Union[Dataset, DatasetDict],
    ) -> TrainingResult:
        """
        Run response distillation end-to-end.

        Loads the teacher model, applies KL + CE blended loss via a custom
        HuggingFace Trainer subclass, saves the student, returns TrainingResult.

        Args:
            dataset: Dataset or DatasetDict with 'train' / optional 'validation'.

        Returns:
            TrainingResult with loss, steps, and output path.
        """
        cfg = self.training_config
        dcfg = self.distillation_config

        # Unpack splits
        if isinstance(dataset, DatasetDict):
            train_dataset = dataset["train"]
            eval_dataset = dataset.get("validation")
        else:
            train_dataset = dataset
            eval_dataset = None

        # Load teacher (frozen, eval mode)
        self.logger.info(f"Loading teacher model: {dcfg.teacher_model_name}")
        try:
            teacher = AutoModelForCausalLM.from_pretrained(dcfg.teacher_model_name)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
            # Move teacher to same device as student
            device = next(self.model.parameters()).device
            teacher = teacher.to(device)
        except Exception as exc:
            raise TrainingError(
                "response_distillation",
                f"Failed to load teacher model '{dcfg.teacher_model_name}': {exc}",
            ) from exc

        # Setup student
        self.model = self._setup_peft(self.model)

        # Build training args
        training_args = self._build_training_args()

        # Build custom distillation trainer
        distillation_trainer = _DistillationTrainer(
            teacher=teacher,
            temperature=dcfg.temperature,
            alpha=dcfg.alpha,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            ),
        )

        # Train
        start = time.time()
        try:
            train_output = distillation_trainer.train()
        except Exception as exc:
            raise TrainingError("response_distillation", str(exc)) from exc
        elapsed = time.time() - start

        # Save student
        output_dir = Path(cfg.output_dir)
        distillation_trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        self.logger.info(f"Student model saved to {output_dir}")

        logs = distillation_trainer.state.log_history
        eval_loss = self._extract_last(logs, "eval_loss")

        return TrainingResult(
            output_dir=output_dir,
            train_loss=train_output.training_loss,
            eval_loss=eval_loss,
            epochs_completed=int(
                train_output.metrics.get("epoch", cfg.num_epochs)
            ),
            steps_completed=train_output.global_step,
            training_time_seconds=elapsed,
            trainer_logs={str(i): entry for i, entry in enumerate(logs)},
        )


# ============================================================================
# Internal: custom HF Trainer with KL + CE blended loss
# ============================================================================


class _DistillationTrainer(Trainer):
    """HuggingFace Trainer subclass that blends KL distillation with CE loss."""

    def __init__(self, teacher: PreTrainedModel, temperature: float, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        # Student forward pass
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits  # (B, T, V)

        # CE loss (standard next-token prediction)
        ce_loss = student_outputs.loss

        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            teacher_outputs = self.teacher(**teacher_inputs)
            teacher_logits = teacher_outputs.logits  # (B, T, V)

        # KL divergence loss over soft targets
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        # Mean over batch and sequence, scaled by T²
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (T ** 2)

        # Blended loss
        loss = self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss

        return (loss, student_outputs) if return_outputs else loss
