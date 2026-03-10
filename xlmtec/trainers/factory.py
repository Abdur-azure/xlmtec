"""
xlmtec.trainers.factory
~~~~~~~~~~~~~~~~~~~~~~~~
TrainerFactory — single entry point for all training.

Uses a registry dict + lazy imports so that heavy deps (peft, trl, torch)
are only imported when a specific trainer is actually needed, keeping
`xlmtec --help` fast even without the [ml] extra installed.

Usage:
    result = TrainerFactory.train(
        model=model, tokenizer=tokenizer, dataset=dataset,
        training_config=config.training.to_config(),
        lora_config=config.lora.to_config(),
    )
"""

from __future__ import annotations

from typing import Optional, Union

from datasets import Dataset, DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.exceptions import MissingConfigError
from ..core.types import (
    DistillationConfig,
    FeatureDistillationConfig,
    LoRAConfig,
    ModelConfig,
    TrainingConfig,
    TrainingMethod,
)
from .base import BaseTrainer, TrainingResult

# ---------------------------------------------------------------------------
# Registry — maps TrainingMethod → (module_path, class_name)
# Trainers are imported lazily inside create() so heavy deps load on demand.
# ---------------------------------------------------------------------------

_REGISTRY: dict[TrainingMethod, tuple[str, str]] = {
    TrainingMethod.LORA:                 ("xlmtec.trainers.lora_trainer",                    "LoRATrainer"),
    TrainingMethod.QLORA:                ("xlmtec.trainers.qlora_trainer",                   "QLoRATrainer"),
    TrainingMethod.FULL_FINETUNING:      ("xlmtec.trainers.full_trainer",                    "FullFineTuner"),
    TrainingMethod.INSTRUCTION_TUNING:   ("xlmtec.trainers.instruction_trainer",             "InstructionTrainer"),
    TrainingMethod.DPO:                  ("xlmtec.trainers.dpo_trainer",                     "DPOTrainer"),
    TrainingMethod.VANILLA_DISTILLATION: ("xlmtec.trainers.response_distillation_trainer",   "ResponseDistillationTrainer"),
    TrainingMethod.FEATURE_DISTILLATION: ("xlmtec.trainers.feature_distillation_trainer",    "FeatureDistillationTrainer"),
}

# Methods that require lora_config
_LORA_METHODS = {
    TrainingMethod.LORA,
    TrainingMethod.QLORA,
    TrainingMethod.INSTRUCTION_TUNING,
    TrainingMethod.DPO,
}


def _load_trainer_class(method: TrainingMethod):
    """Import and return the trainer class for *method*."""
    if method not in _REGISTRY:
        known = ", ".join(m.value for m in _REGISTRY)
        raise NotImplementedError(
            f"No trainer registered for '{method.value}'. "
            f"Supported: {known}"
        )
    module_path, class_name = _REGISTRY[method]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class TrainerFactory:
    """Creates the right trainer for a given TrainingMethod."""

    @staticmethod
    def create(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_config: TrainingConfig,
        lora_config: Optional[LoRAConfig] = None,
        model_config: Optional[ModelConfig] = None,
        distillation_config: Optional[DistillationConfig] = None,
        feature_distillation_config: Optional[FeatureDistillationConfig] = None,
    ) -> BaseTrainer:
        """Validate required configs then instantiate the correct trainer.

        Raises:
            MissingConfigError: If a method-specific config is absent.
            NotImplementedError: If method has no registered trainer.
        """
        method = training_config.method
        trainer_cls = _load_trainer_class(method)

        # Validate + construct per method
        if method in _LORA_METHODS:
            if lora_config is None:
                raise MissingConfigError("lora_config", method.value)

        if method == TrainingMethod.QLORA:
            if model_config is None:
                raise MissingConfigError("model_config", method.value)
            return trainer_cls(model, tokenizer, training_config, lora_config, model_config)

        if method in {TrainingMethod.LORA, TrainingMethod.INSTRUCTION_TUNING}:
            return trainer_cls(model, tokenizer, training_config, lora_config)

        if method == TrainingMethod.DPO:
            return trainer_cls(model, tokenizer, training_config, lora_config)

        if method == TrainingMethod.VANILLA_DISTILLATION:
            if distillation_config is None:
                raise MissingConfigError("distillation_config", method.value)
            return trainer_cls(model, tokenizer, training_config, distillation_config)

        if method == TrainingMethod.FEATURE_DISTILLATION:
            if feature_distillation_config is None:
                raise MissingConfigError("feature_distillation_config", method.value)
            return trainer_cls(model, tokenizer, training_config, feature_distillation_config)

        # FULL_FINETUNING and STRUCTURED_PRUNING — no extra config needed
        return trainer_cls(model, tokenizer, training_config)

    @staticmethod
    def train(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Union[Dataset, DatasetDict],
        training_config: TrainingConfig,
        lora_config: Optional[LoRAConfig] = None,
        model_config: Optional[ModelConfig] = None,
        distillation_config: Optional[DistillationConfig] = None,
        feature_distillation_config: Optional[FeatureDistillationConfig] = None,
    ) -> TrainingResult:
        """One-call convenience: create trainer and run training."""
        trainer = TrainerFactory.create(
            model=model,
            tokenizer=tokenizer,
            training_config=training_config,
            lora_config=lora_config,
            model_config=model_config,
            distillation_config=distillation_config,
            feature_distillation_config=feature_distillation_config,
        )
        return trainer.train(dataset)