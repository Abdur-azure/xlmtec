"""
xlmtec.trainers.dpo_trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DPO (Direct Preference Optimization) trainer.

Extends LoRATrainer — inherits _setup_peft so there is no duplicate
LoRA setup code. Only the training loop is overridden because DPO uses
TRL's DPOTrainer instead of the standard HF Trainer.

Dataset requirements:
    Each example must have three string columns:
        - ``prompt``   — the instruction / input context
        - ``chosen``   — the preferred completion
        - ``rejected`` — the dispreferred completion

Usage:
    trainer = DPOTrainer(model, tokenizer, training_config, lora_config)
    result = trainer.train(dataset)
"""

from __future__ import annotations

from typing import Union

from datasets import Dataset, DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.exceptions import DatasetError, TrainingError
from ..core.types import LoRAConfig as LoRAConfigType
from ..core.types import TrainingConfig
from .base import TrainingResult
from .lora_trainer import LoRATrainer

_DPO_REQUIRED_COLUMNS = {"prompt", "chosen", "rejected"}


def validate_dpo_dataset(dataset: Dataset) -> None:
    """Raise DatasetError if the dataset is missing DPO-required columns."""
    missing = _DPO_REQUIRED_COLUMNS - set(dataset.column_names)
    if missing:
        raise DatasetError(
            f"DPO dataset is missing required columns: {sorted(missing)}. "
            f"Expected: prompt, chosen, rejected. Got: {sorted(dataset.column_names)}"
        )


class DPOTrainer(LoRATrainer):
    """
    DPO trainer — extends LoRATrainer.

    Inherits _setup_peft from LoRATrainer (no duplicate LoRA code).
    Overrides train() to use TRL's DPOTrainer instead of HF Trainer.

    Requires trl>=0.7.0 — install with: pip install xlmtec[dpo]
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_config: TrainingConfig,
        lora_config: LoRAConfigType,
        beta: float = 0.1,
    ) -> None:
        super().__init__(model, tokenizer, training_config, lora_config)
        self.beta = beta
        self.logger.info(f"DPO beta={beta}")

    # _setup_peft is inherited from LoRATrainer — no duplication needed

    def train(self, dataset: Union[Dataset, DatasetDict]) -> TrainingResult:
        """Run DPO training using TRL's DPOTrainer."""
        import time
        from pathlib import Path

        try:
            from trl import DPOConfig
            from trl import DPOTrainer as TRLDPOTrainer
        except ImportError as exc:
            raise TrainingError(
                "dpo",
                "trl is not installed. Install with: pip install xlmtec[dpo]",
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

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Attach LoRA adapters (from LoRATrainer._setup_peft)
        self.model = self._setup_peft(self.model)

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

        trl_trainer = TRLDPOTrainer(
            model=self.model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        self.logger.info("Starting DPO training...")
        start = time.time()
        try:
            train_output = trl_trainer.train()
        except Exception as exc:
            raise TrainingError("dpo", str(exc)) from exc
        elapsed = time.time() - start

        output_dir = Path(cfg.output_dir)
        trl_trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        self.logger.info(f"DPO adapter saved to {output_dir}")

        logs = trl_trainer.state.log_history
        return TrainingResult(
            output_dir=output_dir,
            train_loss=train_output.training_loss,
            eval_loss=self._extract_last(logs, "eval_loss"),
            epochs_completed=int(train_output.metrics.get("epoch", cfg.num_epochs)),
            steps_completed=train_output.global_step,
            training_time_seconds=elapsed,
            trainer_logs={str(i): e for i, e in enumerate(logs)},
        )
