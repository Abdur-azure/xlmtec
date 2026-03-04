"""
Abstract base trainer and shared result types.

All concrete trainers extend BaseTrainer and return TrainingResult.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset, DatasetDict

from ..core.types import TrainingConfig, LoRAConfig
from ..core.exceptions import TrainingError
from ..utils.logging import get_logger

@dataclass
class TrainingState:
    """Mutable training state tracked during a run."""
    current_epoch: int = 0
    current_step: int = 0
    best_loss: float = float("inf")
    is_complete: bool = False

# ============================================================================
# RESULT TYPE
# ============================================================================


@dataclass(frozen=True)
class TrainingResult:
    """Immutable result returned by any trainer."""

    output_dir: Path
    """Directory where the model/adapter was saved."""

    train_loss: float
    """Final training loss."""

    eval_loss: Optional[float]
    """Final evaluation loss (None if no validation set was provided)."""

    epochs_completed: int
    """Number of epochs actually run."""

    steps_completed: int
    """Total optimizer steps completed."""

    training_time_seconds: float
    """Wall-clock training time."""

    trainer_logs: Dict[str, Any] = field(default_factory=dict)
    """Raw log history from the HF Trainer."""


# ============================================================================
# ABSTRACT BASE TRAINER
# ============================================================================


class BaseTrainer(ABC):
    """
    Abstract base for all trainers.

    Subclasses implement ``_setup_peft`` to configure PEFT adapters and
    ``_build_training_args`` to customise HuggingFace TrainingArguments.
    The ``train`` method orchestrates the full flow.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_config: TrainingConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_config = training_config
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        dataset: Union[Dataset, DatasetDict],
    ) -> TrainingResult:
        """
        Run training end-to-end.

        Args:
            dataset: Either a single Dataset (train only) or a DatasetDict
                     with 'train' and optional 'validation' keys.

        Returns:
            TrainingResult with metrics and output path.
        """
        import time

        self.logger.info(f"Starting training — method: {self.training_config.method.value}")

        # Unpack splits
        if isinstance(dataset, DatasetDict):
            train_dataset = dataset["train"]
            eval_dataset = dataset.get("validation")
        else:
            train_dataset = dataset
            eval_dataset = None

        # Setup PEFT adapters (subclass responsibility)
        self.model = self._setup_peft(self.model)

        # Build HF TrainingArguments
        training_args = self._build_training_args()

        # Build HF Trainer
        hf_trainer = self._build_hf_trainer(
            training_args, train_dataset, eval_dataset
        )

        # Train
        start = time.time()
        try:
            train_output = hf_trainer.train()
        except Exception as exc:
            raise TrainingError(
                self.training_config.method.value,
                str(exc),
            ) from exc

        elapsed = time.time() - start

        # Persist
        output_dir = Path(self.training_config.output_dir)
        hf_trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        self.logger.info(f"Model saved to {output_dir}")

        # Extract metrics
        logs = hf_trainer.state.log_history
        eval_loss = self._extract_last(logs, "eval_loss")

        return TrainingResult(
            output_dir=output_dir,
            train_loss=train_output.training_loss,
            eval_loss=eval_loss,
            epochs_completed=int(train_output.metrics.get("epoch", self.training_config.num_epochs)),
            steps_completed=train_output.global_step,
            training_time_seconds=elapsed,
            trainer_logs={str(i): entry for i, entry in enumerate(logs)},
        )

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _setup_peft(self, model: PreTrainedModel) -> PreTrainedModel:
        """Configure and attach PEFT adapters to the model."""

    def _build_training_args(self):
        """Build HuggingFace TrainingArguments from training_config."""
        from transformers import TrainingArguments

        cfg = self.training_config
        do_eval = cfg.evaluation_strategy != "no"
        return TrainingArguments(
            output_dir=str(cfg.output_dir),
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type=cfg.lr_scheduler_type,
            weight_decay=cfg.weight_decay,
            fp16=cfg.fp16,
            bf16=cfg.bf16,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps or 500,
            save_strategy=cfg.save_strategy,
            eval_strategy=cfg.evaluation_strategy,
            load_best_model_at_end=do_eval and cfg.load_best_model_at_end,
            seed=cfg.seed,
            gradient_checkpointing=cfg.gradient_checkpointing,
            report_to="none",
            remove_unused_columns=False,
        )

    def _build_hf_trainer(self, training_args, train_dataset, eval_dataset):
        """Construct the HuggingFace Trainer."""
        from transformers import Trainer, DataCollatorForLanguageModeling

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_last(logs: list, key: str) -> Optional[float]:
        """Return the last value for *key* from the HF log history."""
        for entry in reversed(logs):
            if key in entry:
                return float(entry[key])
        return None