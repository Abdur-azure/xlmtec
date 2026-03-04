"""
Abstract base classes and interfaces for evaluation system.

Defines protocols for metrics, evaluators, and benchmarkers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset

from ..core.types import EvaluationConfig, EvaluationResult, EvaluationMetric
from ..core.exceptions import EvaluationError
from ..utils.logging import get_logger


logger = get_logger(__name__)


# ============================================================================
# EVALUATION RESULT TYPES
# ============================================================================


@dataclass
class MetricResult:
    """Result from a single metric evaluation."""
    
    metric_name: str
    score: float
    samples_evaluated: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric': self.metric_name,
            'score': self.score,
            'samples': self.samples_evaluated,
            'metadata': self.metadata
        }


@dataclass
class ComparisonResult:
    """Result from comparing two models."""
    
    base_metrics: Dict[str, float]
    finetuned_metrics: Dict[str, float]
    improvements: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'base_model': self.base_metrics,
            'finetuned_model': self.finetuned_metrics,
            'improvements': self.improvements,
            'timestamp': self.timestamp.isoformat()
        }
    
    def get_average_improvement(self) -> float:
        """Calculate average improvement across all metrics."""
        if not self.improvements:
            return 0.0
        return sum(self.improvements.values()) / len(self.improvements)


# ============================================================================
# ABSTRACT METRIC
# ============================================================================


class Metric(ABC):
    """
    Abstract base class for evaluation metrics.
    
    Each metric implementation computes a specific score
    (ROUGE, BLEU, perplexity, etc.)
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> float:
        """
        Compute metric score.
        
        Args:
            predictions: Model predictions
            references: Reference texts
            **kwargs: Additional metric-specific arguments
        
        Returns:
            Metric score
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get metric name."""
        pass
    
    def get_range(self) -> tuple:
        """
        Get valid score range.
        
        Returns:
            Tuple of (min_score, max_score)
        """
        return (0.0, 1.0)
    
    def is_higher_better(self) -> bool:
        """
        Whether higher scores are better.
        
        Returns:
            True if higher is better, False otherwise
        """
        return True


# ============================================================================
# ABSTRACT EVALUATOR
# ============================================================================


class Evaluator(ABC):
    """
    Abstract base class for model evaluation.
    
    Handles generation, metric computation, and result aggregation.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: EvaluationConfig
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            config: Evaluation configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Metrics registry
        self.metrics: Dict[str, Metric] = {}
    
    @abstractmethod
    def evaluate(
        self,
        dataset: Dataset,
        metrics: Optional[List[EvaluationMetric]] = None
    ) -> EvaluationResult:
        """
        Evaluate model on dataset.
        
        Args:
            dataset: Evaluation dataset
            metrics: Optional list of metrics to compute
        
        Returns:
            Evaluation results
        """
        pass
    
    @abstractmethod
    def generate_predictions(
        self,
        inputs: List[str],
        batch_size: Optional[int] = None
    ) -> List[str]:
        """
        Generate predictions for inputs.
        
        Args:
            inputs: Input texts
            batch_size: Optional batch size
        
        Returns:
            Generated texts
        """
        pass
    
    def add_metric(self, metric: Metric) -> None:
        """
        Add metric to evaluator.
        
        Args:
            metric: Metric instance
        """
        self.metrics[metric.get_name()] = metric
        self.logger.debug(f"Added metric: {metric.get_name()}")
    
    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute multiple metrics.
        
        Args:
            predictions: Model predictions
            references: Reference texts
            metric_names: Optional list of metric names to compute
        
        Returns:
            Dictionary of metric scores
        """
        if metric_names is None:
            metric_names = list(self.metrics.keys())
        
        results = {}
        for name in metric_names:
            if name not in self.metrics:
                self.logger.warning(f"Metric '{name}' not available, skipping")
                continue
            
            try:
                score = self.metrics[name].compute(predictions, references)
                results[name] = score
                self.logger.debug(f"{name}: {score:.4f}")
            except Exception as e:
                self.logger.error(f"Error computing {name}: {e}")
                results[name] = 0.0
        
        return results


# ============================================================================
# BENCHMARKER
# ============================================================================


class Benchmarker(ABC):
    """
    Abstract base class for benchmarking models.
    
    Compares base model vs fine-tuned model performance.
    """
    
    @abstractmethod
    def benchmark(
        self,
        base_model: PreTrainedModel,
        finetuned_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
        config: EvaluationConfig
    ) -> ComparisonResult:
        """
        Benchmark base vs fine-tuned model.
        
        Args:
            base_model: Original model
            finetuned_model: Fine-tuned model
            tokenizer: Tokenizer
            dataset: Evaluation dataset
            config: Evaluation configuration
        
        Returns:
            Comparison results
        """
        pass


# ============================================================================
# METRIC REGISTRY
# ============================================================================


class MetricRegistry:
    """
    Registry for metric implementations.
    
    Enables dynamic metric selection and custom metric registration.
    """
    
    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
    
    def register(self, metric: Metric) -> None:
        """
        Register a metric.
        
        Args:
            metric: Metric instance
        """
        name = metric.get_name()
        self._metrics[name] = metric
        logger.debug(f"Registered metric: {name}")
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """
        Get metric by name.
        
        Args:
            name: Metric name
        
        Returns:
            Metric instance or None
        """
        return self._metrics.get(name)
    
    def get_all_metrics(self) -> List[Metric]:
        """Get all registered metrics."""
        return list(self._metrics.values())
    
    def is_available(self, name: str) -> bool:
        """Check if metric is available."""
        return name in self._metrics
    
    def list_available(self) -> List[str]:
        """List available metric names."""
        return list(self._metrics.keys())


# Global registry instance
_metric_registry = MetricRegistry()


def register_metric(metric: Metric) -> None:
    """Register a custom metric."""
    _metric_registry.register(metric)


def get_metric(name: str) -> Optional[Metric]:
    """Get metric by name."""
    return _metric_registry.get_metric(name)


def get_all_metrics() -> List[Metric]:
    """Get all registered metrics."""
    return _metric_registry.get_all_metrics()


def list_available_metrics() -> List[str]:
    """List available metric names."""
    return _metric_registry.list_available()