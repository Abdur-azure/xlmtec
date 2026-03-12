"""
xlmtec.inference.predictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~
BatchPredictor — loads a fine-tuned model and runs batched text generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from xlmtec.inference.data_loader import DataLoader, InferenceRecord
from xlmtec.inference.writer import PredictionRecord, PredictionWriter


@dataclass
class PredictConfig:
    """Configuration for a batch inference run."""

    model_dir: Path
    data_path: Path
    output_path: Path
    output_format: str = "jsonl"  # jsonl or csv
    text_column: str | None = None
    batch_size: int = 8
    max_new_tokens: int = 128
    temperature: float = 1.0
    device: str = "auto"  # auto, cpu, cuda


@dataclass
class PredictResult:
    """Summary of a completed inference run."""

    total_records: int
    output_path: Path
    model_dir: Path
    text_column: str
    errors: list[str] = field(default_factory=list)


class BatchPredictor:
    """Load a model and run batch inference over an input dataset."""

    def predict(self, cfg: PredictConfig) -> PredictResult:
        """Run batch inference.

        Args:
            cfg: PredictConfig with all settings.

        Returns:
            PredictResult with summary info.

        Raises:
            FileNotFoundError: If model_dir or data_path not found.
            ImportError: If transformers/torch not installed.
        """
        if not cfg.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {cfg.model_dir}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Batch inference requires torch and transformers.\n"
                "Install with: pip install xlmtec[ml]\n"
                f"Original error: {exc}"
            ) from exc

        # Load data
        loader = DataLoader(cfg.data_path, text_column=cfg.text_column)
        records = loader.load()
        text_col = loader.text_column or loader.detect_column()

        # Load model
        device = self._resolve_device(cfg.device)
        tokenizer = AutoTokenizer.from_pretrained(str(cfg.model_dir))
        model = AutoModelForCausalLM.from_pretrained(
            str(cfg.model_dir),
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        ).to(device)
        model.eval()

        # Set pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Run batched inference
        predictions = self._run_batches(
            records=records,
            model=model,
            tokenizer=tokenizer,
            batch_size=cfg.batch_size,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            device=device,
        )

        # Write output
        writer = PredictionWriter(cfg.output_path, fmt=cfg.output_format)
        writer.write(predictions)

        return PredictResult(
            total_records=len(records),
            output_path=cfg.output_path,
            model_dir=cfg.model_dir,
            text_column=text_col,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_batches(
        self,
        records: list[InferenceRecord],
        model,
        tokenizer,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        device: str,
    ) -> list[PredictionRecord]:
        import torch

        predictions = []
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            texts = [r.text for r in batch]

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature != 1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode only the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            for record, output_ids in zip(batch, outputs):
                new_ids = output_ids[input_len:]
                decoded = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                predictions.append(
                    PredictionRecord(
                        index=record.index,
                        input_text=record.text,
                        prediction=decoded,
                        source=record.source,
                    )
                )

        return predictions

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
