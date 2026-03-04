"""
DPO (Direct Preference Optimization) trainer.

Trains a model on preference data — {prompt, chosen, rejected} triples —
using TRL's DPOTrainer. Attaches a LoRA adapter; the base model acts as
the implicit reference model.

Dataset requirements:
    Each example must have three string columns:
        - ``prompt``   — the instruction / input context
        - ``chosen``   — the preferred completion
        - ``rejected`` — the dispreferred completion

Usage::

    trainer = DPOTrainer(model, tokenizer, training_config, lora_config)
    result = trainer.train(dataset)
"""

import warnings
from typing import Optional, Union

from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType

from ..core.types import TrainingConfig, LoRAConfig as LoRAConfigType
from ..core.exceptions import TrainingError, DatasetError
from ..models.loader import detect_target_modules
from ..utils.logging import get_logger
from .base import BaseTrainer, TrainingResult


# Required columns for DPO datasets
_DPO_REQUIRED_COLUMNS = {"prompt", "chosen", "rejected"}

logger = get_logger(__name__)


def validate_dpo_dataset(dataset: Dataset) -> None:
    """
    Raise ``DatasetError`` if the dataset is missing DPO-required columns.

    Args:
        dataset: HuggingFace Dataset to validate.

    Raises:
        DatasetError: If any of prompt / chosen / rejected columns are absent.
    """
    missing = _DPO_REQUIRED_COLUMNS - set(dataset.column_names)
    if missing:
        raise DatasetError(
            f"DPO dataset is missing required columns: {sorted(missing)}. "
            f"Expected: prompt, chosen, rejected. "
            f"Got: {sorted(dataset.column_names)}"
        )


class DPOTrainer(BaseTrainer):
    """
    Trainer for Direct Preference Optimization (DPO).

    Wraps TRL's ``DPOTrainer`` with LoRA adapter setup. The base model
    serves as the implicit reference model — no separate reference copy
    is required in memory.

    Requires ``trl>=0.7.0``.
    """

    # TRL minimum version required
    _TRL_MIN_VERSION = (0, 7, 0)

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_config: TrainingConfig,
        lora_config: LoRAConfigType,
        beta: float = 0.1,
    ):
        """
        Args:
            model: Base model to fine-tune.
            tokenizer: Corresponding tokenizer.
            training_config: Core training hyper-parameters.
            lora_config: LoRA adapter configuration.
            beta: DPO temperature — controls deviation from reference policy.
                  Lower = closer to reference. Default 0.1 is a sensible start.
        """
        super().__init__(model, tokenizer, training_config)
        self.lora_config = lora_config
        self.beta = beta

        self.logger.info(
            f"DPO config — beta={beta}, r={lora_config.r}, "
            f"alpha={lora_config.lora_alpha}, "
            f"target_modules={lora_config.target_modules or 'auto-detect'}"
        )

    # ------------------------------------------------------------------
    # BaseTrainer hooks
    # ------------------------------------------------------------------

    def _setup_peft(self, model: PreTrainedModel) -> PreTrainedModel:
        """Attach LoRA adapters for DPO training."""
        target_modules = self.lora_config.target_modules or detect_target_modules(model)
        self.logger.info(f"DPO LoRA target modules: {target_modules}")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=target_modules,
            bias=self.lora_config.bias,
        )
        peft_model = get_peft_model(model, peft_config)

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        self.logger.info(
            f"DPO trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )
        return peft_model

    def train(
        self,
        dataset: Union[Dataset, DatasetDict],
    ) -> TrainingResult:
        """
        Run DPO training end-to-end.

        Validates dataset columns, sets up LoRA, builds TRL DPOTrainer,
        runs training, saves adapter.

        Args:
            dataset: Dataset or DatasetDict with 'train' / optional 'validation'.
                     Must have ``prompt``, ``chosen``, ``rejected`` columns.

        Returns:
            TrainingResult with loss, steps, and output path.
        """
        import time

        try:
            from trl import DPOTrainer as TRLDPOTrainer, DPOConfig
        except ImportError as exc:
            raise TrainingError(
                "dpo",
                "trl is not installed. Install it with: pip install trl>=0.7.0",
            ) from exc

        # Unpack splits
        if isinstance(dataset, DatasetDict):
            train_dataset = dataset["train"]
            eval_dataset = dataset.get("validation")
        else:
            train_dataset = dataset
            eval_dataset = None

        # Validate columns
        validate_dpo_dataset(train_dataset)
        if eval_dataset is not None:
            validate_dpo_dataset(eval_dataset)

        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.info("pad_token set to eos_token for DPO training")

        # Setup PEFT
        self.model = self._setup_peft(self.model)

        # Build DPOConfig (TRL's replacement for TrainingArguments in DPO)
        cfg = self.training_config
        dpo_config = DPOConfig(
            output_dir=str(cfg.output_dir),
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            fp16=cfg.fp16,
            bf16=cfg.bf16,
            logging_steps=cfg.logging_steps,
            save_strategy=cfg.save_strategy,
            evaluation_strategy="epoch" if eval_dataset is not None else "no",
            seed=cfg.seed,
            gradient_checkpointing=cfg.gradient_checkpointing,
            beta=self.beta,
            report_to="none",
            remove_unused_columns=False,
        )

        # Build TRL DPOTrainer
        trl_trainer = TRLDPOTrainer(
            model=self.model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        # Train
        self.logger.info("Starting DPO training...")
        start = time.time()
        try:
            train_output = trl_trainer.train()
        except Exception as exc:
            raise TrainingError("dpo", str(exc)) from exc
        elapsed = time.time() - start

        # Save
        from pathlib import Path
        output_dir = Path(cfg.output_dir)
        trl_trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        self.logger.info(f"DPO adapter saved to {output_dir}")

        logs = trl_trainer.state.log_history
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
            trainer_logs={str(i): e for i, e in enumerate(logs)},
        )