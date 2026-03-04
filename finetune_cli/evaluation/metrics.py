"""
Evaluation metrics: ROUGE, BLEU, Perplexity.

Each metric follows a consistent interface::

    score = MetricClass().compute(predictions, references)

``predictions`` and ``references`` are lists of strings.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.types import EvaluationMetric
from ..core.exceptions import EvaluationError
from ..utils.logging import get_logger


logger = get_logger(__name__)


# ============================================================================
# BASE
# ============================================================================


class Metric(ABC):
    """Abstract base for all metrics."""

    @property
    @abstractmethod
    def name(self) -> EvaluationMetric:
        """Metric identifier."""

    @abstractmethod
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute metric score.

        Args:
            predictions: Model-generated text.
            references: Ground-truth text.

        Returns:
            Scalar score (higher = better, except Perplexity).
        """


# ============================================================================
# ROUGE
# ============================================================================


class RougeMetric(Metric):
    """ROUGE-N and ROUGE-L via the rouge_score library."""

    _VARIANT_MAP = {
        EvaluationMetric.ROUGE_1: "rouge1",
        EvaluationMetric.ROUGE_2: "rouge2",
        EvaluationMetric.ROUGE_L: "rougeL",
    }

    def __init__(self, variant: EvaluationMetric = EvaluationMetric.ROUGE_L):
        if variant not in self._VARIANT_MAP:
            raise ValueError(f"Unknown ROUGE variant: {variant}")
        self._variant = variant
        self._rouge_key = self._VARIANT_MAP[variant]

    @property
    def name(self) -> EvaluationMetric:
        return self._variant

    def compute(self, predictions: List[str], references: List[str]) -> float:
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            raise EvaluationError(
                "rouge_score not installed. Run: pip install rouge-score"
            )

        scorer = rouge_scorer.RougeScorer([self._rouge_key], use_stemmer=True)
        scores = [
            scorer.score(ref, pred)[self._rouge_key].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        return sum(scores) / len(scores) if scores else 0.0


# ============================================================================
# BLEU
# ============================================================================


class BleuMetric(Metric):
    """Corpus-level BLEU via nltk."""

    @property
    def name(self) -> EvaluationMetric:
        return EvaluationMetric.BLEU

    def compute(self, predictions: List[str], references: List[str]) -> float:
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            from nltk.tokenize import word_tokenize
        except ImportError:
            raise EvaluationError(
                "nltk not installed. Run: pip install nltk"
            )

        smoothing = SmoothingFunction().method1
        hypothesis_list = [word_tokenize(p.lower()) for p in predictions]
        reference_list = [[word_tokenize(r.lower())] for r in references]

        try:
            return corpus_bleu(reference_list, hypothesis_list, smoothing_function=smoothing)
        except Exception as exc:
            raise EvaluationError(f"BLEU computation failed: {exc}") from exc


# ============================================================================
# PERPLEXITY
# ============================================================================


class PerplexityMetric(Metric):
    """
    Token-level perplexity computed from model log-likelihoods.

    Lower perplexity = better.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> EvaluationMetric:
        return EvaluationMetric.PERPLEXITY

    def compute(self, predictions: List[str], references: List[str]) -> float:
        """Compute average perplexity over *predictions* (references ignored)."""
        self._model.eval()
        total_loss = 0.0
        total_tokens = 0

        for i in range(0, len(predictions), self._batch_size):
            batch_texts = predictions[i : i + self._batch_size]
            encodings = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
                padding=True,
            ).to(self._device)

            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]

            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                # outputs.loss is mean cross-entropy over non-padded tokens
                batch_tokens = int(attention_mask.sum().item())
                total_loss += outputs.loss.item() * batch_tokens
                total_tokens += batch_tokens

        if total_tokens == 0:
            return float("inf")

        avg_loss = total_loss / total_tokens
        return float(torch.exp(torch.tensor(avg_loss)).item())


# ============================================================================
# REGISTRY
# ============================================================================


class MetricRegistry:
    """Returns the correct Metric instance for an EvaluationMetric enum."""

    @staticmethod
    def get(
        metric: EvaluationMetric,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ) -> Metric:
        if metric in (EvaluationMetric.ROUGE_1, EvaluationMetric.ROUGE_2, EvaluationMetric.ROUGE_L):
            return RougeMetric(variant=metric)
        if metric == EvaluationMetric.BLEU:
            return BleuMetric()
        if metric == EvaluationMetric.PERPLEXITY:
            if model is None or tokenizer is None:
                raise ValueError("Perplexity requires model and tokenizer.")
            return PerplexityMetric(model=model, tokenizer=tokenizer, **kwargs)
        raise NotImplementedError(f"Metric '{metric.value}' not yet implemented.")