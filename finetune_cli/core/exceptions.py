"""Exception hierarchy for the fine-tuning framework."""

from typing import List, Optional


class FineTuneError(Exception):
    """Base exception for all framework errors."""


# ============================================================================
# CONFIGURATION ERRORS
# ============================================================================

class ConfigurationError(FineTuneError):
    """Base for configuration errors."""


class InvalidConfigError(ConfigurationError):
    def __init__(self, message: str):
        super().__init__(f"Invalid configuration: {message}")


class MissingConfigError(ConfigurationError):
    def __init__(self, field: str, context: str = ""):
        ctx = f" for {context}" if context else ""
        super().__init__(f"Missing required config field '{field}'{ctx}")


class IncompatibleConfigError(ConfigurationError):
    def __init__(self, message: str, fields: Optional[List[str]] = None):
        fields_str = f" (fields: {fields})" if fields else ""
        super().__init__(f"Incompatible configuration: {message}{fields_str}")


# ============================================================================
# MODEL ERRORS
# ============================================================================

class ModelError(FineTuneError):
    """Base for model errors."""


class ModelLoadError(ModelError):
    def __init__(self, model_name: str, reason: str = ""):
        super().__init__(f"Failed to load model '{model_name}': {reason}")


class ModelNotFoundError(ModelError):
    def __init__(self, model_name: str):
        super().__init__(f"Model not found: '{model_name}'")


class UnsupportedModelError(ModelError):
    def __init__(self, model_name: str, reason: str = ""):
        super().__init__(f"Unsupported model '{model_name}': {reason}")


class CUDANotAvailableError(ModelError):
    def __init__(self):
        super().__init__("CUDA is not available but was requested.")


class TargetModulesNotFoundError(ModelError):
    def __init__(self, model_name: str, patterns: List[str]):
        super().__init__(
            f"No suitable LoRA target modules found in '{model_name}'. "
            f"Tried patterns: {patterns}"
        )


# ============================================================================
# DATASET ERRORS
# ============================================================================

class DatasetError(FineTuneError):
    """Base for dataset errors."""


class DatasetLoadError(DatasetError):
    def __init__(self, path: str, reason: str = ""):
        super().__init__(f"Failed to load dataset '{path}': {reason}")


class DatasetNotFoundError(DatasetError):
    def __init__(self, path: str):
        super().__init__(f"Dataset not found: '{path}'")


class NoTextColumnsError(DatasetError):
    def __init__(self, columns: List[str]):
        super().__init__(f"No text columns found. Available columns: {columns}")


class EmptyDatasetError(DatasetError):
    def __init__(self, path: str = ""):
        super().__init__(f"Dataset is empty: {path}")


# ============================================================================
# TRAINING ERRORS
# ============================================================================

class TrainingError(FineTuneError):
    def __init__(self, method: str, reason: str = ""):
        super().__init__(f"Training failed (method={method}): {reason}")


class OutOfMemoryError(TrainingError):
    def __init__(self):
        FineTuneError.__init__(self, "Out of GPU memory. Reduce batch size or use QLoRA.")


class NaNLossError(TrainingError):
    def __init__(self, step: int):
        FineTuneError.__init__(self, f"NaN loss detected at step {step}. Reduce learning rate.")


class CheckpointError(TrainingError):
    def __init__(self, path: str, reason: str = ""):
        FineTuneError.__init__(self, f"Checkpoint error at '{path}': {reason}")


class InsufficientVRAMError(TrainingError):
    def __init__(self, required_gb: float, available_gb: float):
        FineTuneError.__init__(
            self,
            f"Insufficient VRAM: need {required_gb:.1f}GB, have {available_gb:.1f}GB"
        )


class MethodNotImplementedError(FineTuneError):
    def __init__(self, method: str):
        super().__init__(f"Training method '{method}' is not yet implemented.")


# ============================================================================
# EVALUATION ERRORS
# ============================================================================

class EvaluationError(FineTuneError):
    def __init__(self, message: str):
        super().__init__(f"Evaluation error: {message}")


class MetricComputationError(EvaluationError):
    def __init__(self, metric: str, reason: str = ""):
        super().__init__(f"Failed to compute metric '{metric}': {reason}")