"""
Model evaluator implementation.

Handles generation, metric computation, and comprehensive evaluation.
"""

from typing import List, Optional, Dict
import time

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from ..core.types import EvaluationConfig, EvaluationResult, EvaluationMetric
from ..core.exceptions import EvaluationError, MetricComputationError
from ..utils.logging import get_logger, LogProgress
from .base import Evaluator, MetricResult
from .metrics import create_metric, PerplexityMetric


logger = get_logger(__name__)


# ============================================================================
# STANDARD EVALUATOR
# ============================================================================


class StandardEvaluator(Evaluator):
    """
    Standard implementation of model evaluation.
    
    Generates predictions and computes metrics on evaluation dataset.
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
            tokenizer: Tokenizer
            config: Evaluation configuration
        """
        super().__init__(model, tokenizer, config)
        
        # Set model to eval mode
        self.model.eval()
        
        # Get device
        self.device = next(model.parameters()).device
        
        # Initialize metrics
        self._initialize_metrics()
    
    def _initialize_metrics(self) -> None:
        """Initialize metrics from config."""
        for metric_name in self.config.metrics:
            try:
                # Special handling for perplexity
                if metric_name == EvaluationMetric.PERPLEXITY:
                    metric = PerplexityMetric(self.model, self.tokenizer)
                else:
                    metric = create_metric(metric_name.value)
                
                self.add_metric(metric)
            except Exception as e:
                self.logger.warning(f"Could not initialize {metric_name}: {e}")
    
    def evaluate(
        self,
        dataset: Dataset,
        metrics: Optional[List[EvaluationMetric]] = None
    ) -> EvaluationResult:
        """
        Evaluate model on dataset.
        
        Args:
            dataset: Evaluation dataset
            metrics: Optional list of metrics (uses config if None)
        
        Returns:
            Evaluation results
        """
        with LogProgress(self.logger, "Evaluating model"):
            start_time = time.time()
            
            # Determine metrics to use
            if metrics is None:
                metrics = [m.value for m in self.config.metrics]
            else:
                metrics = [m.value for m in metrics]
            
            # Limit samples if configured
            num_samples = len(dataset)
            if self.config.num_samples and self.config.num_samples < num_samples:
                dataset = dataset.select(range(self.config.num_samples))
                num_samples = self.config.num_samples
            
            self.logger.info(f"Evaluating on {num_samples} samples...")
            
            # Extract inputs and references
            inputs, references = self._prepare_data(dataset)
            
            # Generate predictions
            predictions = self.generate_predictions(inputs, self.config.batch_size)
            
            # Compute metrics
            metric_scores = self.compute_metrics(predictions, references, metrics)
            
            # Calculate evaluation time
            eval_time = time.time() - start_time
            
            # Build result
            result = EvaluationResult(
                metrics=metric_scores,
                num_samples=num_samples,
                evaluation_time_seconds=eval_time
            )
            
            self.logger.info("Evaluation complete!")
            self.logger.info(f"  Samples: {num_samples}")
            self.logger.info(f"  Time: {eval_time:.2f}s")
            for metric, score in metric_scores.items():
                self.logger.info(f"  {metric}: {score:.4f}")
            
            return result
    
    def _prepare_data(self, dataset: Dataset) -> tuple:
        """
        Extract inputs and references from dataset.
        
        Args:
            dataset: Evaluation dataset
        
        Returns:
            Tuple of (inputs, references)
        """
        # Try to detect input/output columns
        column_names = dataset.column_names
        
        # Common patterns
        input_candidates = ['text', 'input', 'prompt', 'question', 'instruction']
        output_candidates = ['label', 'output', 'response', 'answer', 'completion']
        
        input_col = None
        output_col = None
        
        # Find input column
        for candidate in input_candidates:
            if candidate in column_names:
                input_col = candidate
                break
        
        # Find output column
        for candidate in output_candidates:
            if candidate in column_names:
                output_col = candidate
                break
        
        # If not found, use first column as input
        if input_col is None:
            input_col = column_names[0]
            self.logger.warning(f"Using '{input_col}' as input column")
        
        # If output not found, use input as reference (for language modeling)
        if output_col is None:
            output_col = input_col
            self.logger.warning("No output column found, using input as reference")
        
        # Extract data
        inputs = [str(item) for item in dataset[input_col]]
        references = [str(item) for item in dataset[output_col]]
        
        return inputs, references
    
    def generate_predictions(
        self,
        inputs: List[str],
        batch_size: Optional[int] = None
    ) -> List[str]:
        """
        Generate predictions for inputs.
        
        Args:
            inputs: Input texts
            batch_size: Batch size for generation
        
        Returns:
            Generated texts
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        self.logger.info(f"Generating predictions (batch_size={batch_size})...")
        
        predictions = []
        
        # Process in batches
        for i in tqdm(range(0, len(inputs), batch_size), desc="Generating"):
            batch = inputs[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **encoded,
                    max_new_tokens=self.config.generation_max_length,
                    temperature=self.config.generation_temperature,
                    top_p=self.config.generation_top_p,
                    do_sample=self.config.generation_do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            batch_predictions = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            predictions.extend(batch_predictions)
        
        return predictions


# ============================================================================
# QUICK EVALUATOR
# ============================================================================


class QuickEvaluator:
    """
    Quick evaluation without full configuration.
    
    Useful for rapid testing and debugging.
    """
    
    @staticmethod
    def evaluate(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        test_inputs: List[str],
        test_references: List[str],
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Quick evaluation on test samples.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            test_inputs: Test inputs
            test_references: Expected outputs
            metrics: List of metric names (default: ['rouge1', 'rouge2', 'rougeL'])
        
        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = ['rouge1', 'rouge2', 'rougeL']
        
        # Create minimal config
        from ..core.types import EvaluationConfig, EvaluationMetric
        
        config = EvaluationConfig(
            metrics=[EvaluationMetric(m) for m in metrics],
            batch_size=8,
            generation_max_length=50
        )
        
        # Create evaluator
        evaluator = StandardEvaluator(model, tokenizer, config)
        
        # Generate predictions
        predictions = evaluator.generate_predictions(test_inputs)
        
        # Compute metrics
        scores = evaluator.compute_metrics(predictions, test_references, metrics)
        
        return scores


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    config: EvaluationConfig
) -> EvaluationResult:
    """
    Convenience function for model evaluation.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset: Evaluation dataset
        config: Evaluation configuration
    
    Returns:
        Evaluation results
    
    Example:
        >>> from finetune_cli.core.config import ConfigBuilder
        >>> from finetune_cli.core.types import EvaluationMetric
        >>> from finetune_cli.models.loader import load_model_and_tokenizer
        >>> from finetune_cli.evaluation import evaluate_model
        >>> 
        >>> config = ConfigBuilder() \\
        ...     .with_model("gpt2") \\
        ...     .with_evaluation(
        ...         metrics=[EvaluationMetric.ROUGE_1, EvaluationMetric.ROUGE_L],
        ...         batch_size=8
        ...     ) \\
        ...     .build()
        >>> 
        >>> model, tokenizer = load_model_and_tokenizer(config.model.to_config())
        >>> result = evaluate_model(
        ...     model, tokenizer, test_dataset,
        ...     config.evaluation.to_config()
        ... )
        >>> print(f"ROUGE-1: {result.metrics['rouge1']:.4f}")
    """
    evaluator = StandardEvaluator(model, tokenizer, config)
    return evaluator.evaluate(dataset)


def quick_evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Quick evaluation without configuration.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        inputs: Test inputs
        references: Expected outputs
    
    Returns:
        Dictionary of scores
    
    Example:
        >>> from finetune_cli.evaluation import quick_evaluate
        >>> 
        >>> scores = quick_evaluate(
        ...     model, tokenizer,
        ...     ["Hello, how are you?", "What is AI?"],
        ...     ["I'm doing well!", "AI is artificial intelligence."]
        ... )
        >>> print(f"ROUGE-1: {scores['rouge1']:.4f}")
    """
    return QuickEvaluator.evaluate(model, tokenizer, inputs, references)