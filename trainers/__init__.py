"""Trainer system for the fine-tuning framework."""

from .base import BaseTrainer, TrainingResult
from .lora_trainer import LoRATrainer
from .qlora_trainer import QLoRATrainer
from .full_trainer import FullFineTuner
from .instruction_trainer import InstructionTrainer, format_instruction_dataset
from .dpo_trainer import DPOTrainer, validate_dpo_dataset
from .response_distillation_trainer import ResponseDistillationTrainer
from .factory import TrainerFactory

__all__ = [
    "BaseTrainer",
    "TrainingResult",
    "LoRATrainer",
    "QLoRATrainer",
    "FullFineTuner",
    "InstructionTrainer",
    "format_instruction_dataset",
    "DPOTrainer",
    "validate_dpo_dataset",
    "ResponseDistillationTrainer",
    "TrainerFactory",
]