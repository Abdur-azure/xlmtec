"""
Feature Distillation trainer (Intermediate Layer KD).

The student learns from two signals simultaneously:
  1. MSE loss between student and teacher hidden states at selected layers.
  2. KL divergence on output logits (same as response distillation).
  3. Standard CE loss on ground-truth labels.

Total loss:
    L = alpha * CE
      + beta  * KL(softmax(teacher_logits/T) || softmax(student_logits/T)) * T²
      + (1 - alpha - beta) * mean_MSE(student_hidden[i], teacher_hidden[i])

Layer selection:
    feature_layers = None  →  auto-select evenly-spaced student layers
    feature_layers = [0, 4, 11]  →  match those specific student layer indices

If teacher has fewer layers than needed, mapping falls back to the nearest
available teacher layer (floor division).

Reference:
    Romero et al., "FitNets: Hints for Thin Deep Nets" (2015).
    Sun et al., "Patient Knowledge Distillation for BERT" (2019).
"""

import time
import warnings
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
)

from ..core.exceptions import TrainingError
from ..core.types import FeatureDistillationConfig, TrainingConfig
from .base import BaseTrainer, TrainingResult

_VRAM_WARNING_THRESHOLD = 1_000_000_000  # 1B params


# ============================================================================
# Layer selection helper
# ============================================================================


def _select_layers(n_student_layers: int, requested: Optional[List[int]]) -> List[int]:
    """Return student layer indices to distil from.

    Args:
        n_student_layers: Total number of transformer layers in the student.
        requested: Explicit list, or None for auto evenly-spaced selection.

    Returns:
        List of valid student layer indices.
    """
    if requested is not None:
        valid = [i for i in requested if 0 <= i < n_student_layers]
        if not valid:
            raise ValueError(
                f"None of the requested feature_layers {requested} are valid "
                f"for a student with {n_student_layers} layers."
            )
        return valid
    # Auto: pick 4 evenly-spaced layers (or fewer if model is small)
    n = min(4, n_student_layers)
    step = max(1, n_student_layers // n)
    return [i * step for i in range(n)]


def _map_teacher_layer(student_idx: int, n_teacher_layers: int, n_student_layers: int) -> int:
    """Map a student layer index to the corresponding teacher layer index."""
    ratio = n_teacher_layers / max(n_student_layers, 1)
    return min(int(student_idx * ratio), n_teacher_layers - 1)


# ============================================================================
# Trainer
# ============================================================================


class FeatureDistillationTrainer(BaseTrainer):
    """
    Knowledge distillation trainer using intermediate hidden-state MSE loss
    combined with output KL divergence and ground-truth CE loss.

    Args:
        model: Student model to train.
        tokenizer: Shared tokenizer.
        training_config: Core training hyper-parameters.
        feature_distillation_config: Feature-distillation settings.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_config: TrainingConfig,
        feature_distillation_config: FeatureDistillationConfig,
    ):
        super().__init__(model, tokenizer, training_config)
        self.fd_config = feature_distillation_config

        total = sum(p.numel() for p in model.parameters())
        if total > _VRAM_WARNING_THRESHOLD:
            warnings.warn(
                f"Feature distillation with a {total:,}-parameter student requires "
                "holding student + teacher + hidden states in memory simultaneously. "
                "Ensure sufficient VRAM or reduce feature_layers.",
                ResourceWarning,
                stacklevel=2,
            )

        self.logger.info(
            f"FeatureDistillationTrainer — teacher: {feature_distillation_config.teacher_model_name}, "
            f"T={feature_distillation_config.temperature}, "
            f"alpha={feature_distillation_config.alpha}, "
            f"beta={feature_distillation_config.beta}, "
            f"layers={feature_distillation_config.feature_layers or 'auto'}"
        )

    # ------------------------------------------------------------------
    # BaseTrainer hook
    # ------------------------------------------------------------------

    def _setup_peft(self, model: PreTrainedModel) -> PreTrainedModel:
        """All student parameters are trainable for feature distillation."""
        for param in model.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Student trainable parameters: {trainable:,} / {total:,} (100%)")
        return model

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, dataset: Union[Dataset, DatasetDict]) -> TrainingResult:
        """
        Run feature distillation end-to-end.

        Loads the teacher (frozen), resolves layer mapping, builds a custom
        HF Trainer that applies MSE + KL + CE loss, saves the student.

        Args:
            dataset: Dataset or DatasetDict with 'train' / optional 'validation'.

        Returns:
            TrainingResult with loss, steps, and output path.
        """
        cfg = self.training_config
        fdcfg = self.fd_config

        # Unpack splits
        if isinstance(dataset, DatasetDict):
            train_dataset = dataset["train"]
            eval_dataset = dataset.get("validation")
        else:
            train_dataset = dataset
            eval_dataset = None

        # Load teacher
        self.logger.info(f"Loading teacher: {fdcfg.teacher_model_name}")
        try:
            teacher = AutoModelForCausalLM.from_pretrained(
                fdcfg.teacher_model_name,
                output_hidden_states=True,
            )
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
            device = next(self.model.parameters()).device
            teacher = teacher.to(device)
        except Exception as exc:
            raise TrainingError(
                "feature_distillation",
                f"Failed to load teacher '{fdcfg.teacher_model_name}': {exc}",
            ) from exc

        # Resolve layer indices
        try:
            n_student = self.model.config.num_hidden_layers
        except AttributeError:
            n_student = 12  # sensible fallback
        try:
            n_teacher = teacher.config.num_hidden_layers
        except AttributeError:
            n_teacher = 12

        student_layers = _select_layers(n_student, fdcfg.feature_layers)
        teacher_layers = [_map_teacher_layer(sl, n_teacher, n_student) for sl in student_layers]
        self.logger.info(f"Layer mapping — student {student_layers} → teacher {teacher_layers}")

        # Setup student
        self.model = self._setup_peft(self.model)

        # Enable hidden state output on student
        try:
            self.model.config.output_hidden_states = True
        except Exception:
            pass  # some model types don't support this via config

        training_args = self._build_training_args()

        fd_trainer = _FeatureDistillationTrainer(
            teacher=teacher,
            student_layers=student_layers,
            teacher_layers=teacher_layers,
            temperature=fdcfg.temperature,
            alpha=fdcfg.alpha,
            beta=fdcfg.beta,
            feature_loss_weight=fdcfg.feature_loss_weight,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
        )

        start = time.time()
        try:
            train_output = fd_trainer.train()
        except Exception as exc:
            raise TrainingError("feature_distillation", str(exc)) from exc
        elapsed = time.time() - start

        output_dir = Path(cfg.output_dir)
        fd_trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        self.logger.info(f"Student saved to {output_dir}")

        logs = fd_trainer.state.log_history
        eval_loss = self._extract_last(logs, "eval_loss")

        return TrainingResult(
            output_dir=output_dir,
            train_loss=train_output.training_loss,
            eval_loss=eval_loss,
            epochs_completed=int(train_output.metrics.get("epoch", cfg.num_epochs)),
            steps_completed=train_output.global_step,
            training_time_seconds=elapsed,
            trainer_logs={str(i): e for i, e in enumerate(logs)},
        )


# ============================================================================
# Internal: custom HF Trainer with MSE + KL + CE loss
# ============================================================================


class _FeatureDistillationTrainer(Trainer):
    """HF Trainer subclass combining feature MSE, output KL, and CE losses."""

    def __init__(
        self,
        teacher: PreTrainedModel,
        student_layers: List[int],
        teacher_layers: List[int],
        temperature: float,
        alpha: float,
        beta: float,
        feature_loss_weight: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher = teacher
        self.student_layers = student_layers
        self.teacher_layers = teacher_layers
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.feature_loss_weight = feature_loss_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Student forward — request hidden states
        student_outputs = model(**inputs, output_hidden_states=True)
        student_logits = student_outputs.logits
        ce_loss = student_outputs.loss

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            teacher_outputs = self.teacher(**teacher_inputs, output_hidden_states=True)
            teacher_logits = teacher_outputs.logits

        # ── Output KL loss ──────────────────────────────────────────────
        T = self.temperature
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T**2)

        # ── Feature MSE loss ────────────────────────────────────────────
        mse_losses = []
        s_hidden = student_outputs.hidden_states  # tuple: (n_layers+1, B, T, H)
        t_hidden = teacher_outputs.hidden_states

        for sl, tl in zip(self.student_layers, self.teacher_layers):
            if sl + 1 >= len(s_hidden) or tl + 1 >= len(t_hidden):
                continue
            s_feat = s_hidden[sl + 1]  # +1 because index 0 is embedding
            t_feat = t_hidden[tl + 1]

            # If hidden dims differ, project via mean pooling over dim=-1
            if s_feat.shape[-1] != t_feat.shape[-1]:
                # Simple dimension-agnostic fallback: normalise then MSE
                s_feat = F.normalize(s_feat, dim=-1)
                t_feat = F.normalize(t_feat, dim=-1)

            mse_losses.append(F.mse_loss(s_feat, t_feat.detach()))

        feature_loss = torch.stack(mse_losses).mean() if mse_losses else torch.tensor(0.0)

        # ── Combined loss ───────────────────────────────────────────────
        gamma = max(0.0, 1.0 - self.alpha - self.beta)
        loss = (
            self.alpha * ce_loss
            + self.beta * kl_loss
            + gamma * self.feature_loss_weight * feature_loss
        )

        return (loss, student_outputs) if return_outputs else loss
