"""
xlmtec.core.types
~~~~~~~~~~~~~~~~~~
Enums and frozen dataclasses — the single source of truth for all data
shapes across the repo.

Rules:
- Zero imports from other xlmtec subpackages.
- No torch at module level — torch is an optional ML dep.
- Only add a TrainingMethod value when its trainer is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, Union

# ============================================================================
# ENUMS
# ============================================================================


class TrainingMethod(str, Enum):
    """Only lists methods that have a concrete trainer implementation."""

    FULL_FINETUNING = "full_finetuning"
    LORA = "lora"
    QLORA = "qlora"
    INSTRUCTION_TUNING = "instruction_tuning"
    DPO = "dpo"
    VANILLA_DISTILLATION = "vanilla_distillation"
    FEATURE_DISTILLATION = "feature_distillation"
    STRUCTURED_PRUNING = "structured_pruning"


class DatasetSource(str, Enum):
    LOCAL_FILE = "local_file"
    HUGGINGFACE_HUB = "huggingface_hub"


class FileFormat(str, Enum):
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    TXT = "txt"


class EvaluationMetric(str, Enum):
    """Only lists metrics actually computed by xlmtec.evaluation."""

    ROUGE_1 = "rouge1"
    ROUGE_2 = "rouge2"
    ROUGE_L = "rougeL"
    BLEU = "bleu"
    PERPLEXITY = "perplexity"
    ACCURACY = "accuracy"


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


# ============================================================================
# FROZEN DATACLASSES
# ============================================================================


@dataclass(frozen=True)
class ModelConfig:
    name: str
    device: DeviceType = DeviceType.AUTO
    torch_dtype: Optional[str] = None  # "float32" | "float16" | "bfloat16" | "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = False
    trust_remote_code: bool = False
    revision: Optional[str] = None
    cache_dir: Optional[Path] = None


@dataclass(frozen=True)
class DatasetConfig:
    source: DatasetSource
    path: str
    split: str = "train"
    config_name: Optional[str] = None
    data_files: Optional[Union[str, List[str]]] = None
    text_columns: Optional[List[str]] = None
    max_samples: Optional[int] = None
    streaming: bool = False
    shuffle: bool = True
    seed: int = 42


@dataclass(frozen=True)
class TokenizationConfig:
    max_length: int = 512
    truncation: bool = True
    padding: Literal["max_length", "longest", "do_not_pad"] = "max_length"
    add_special_tokens: bool = True
    return_attention_mask: bool = True


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: Literal["none", "all", "lora_only"] = "none"
    fan_in_fan_out: bool = False
    init_lora_weights: Union[bool, Literal["gaussian", "loftq"]] = True


@dataclass(frozen=True)
class TrainingConfig:
    method: TrainingMethod
    output_dir: str
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 10
    save_steps: Optional[int] = None
    save_strategy: str = "epoch"
    evaluation_strategy: str = "no"
    load_best_model_at_end: bool = False
    seed: int = 42


@dataclass(frozen=True)
class EvaluationConfig:
    metrics: List[EvaluationMetric] = field(default_factory=lambda: [EvaluationMetric.ROUGE_L])
    batch_size: int = 8
    num_samples: Optional[int] = None
    generation_max_length: int = 100
    generation_temperature: float = 0.7
    generation_top_p: float = 0.9
    generation_do_sample: bool = True


@dataclass(frozen=True)
class DistillationConfig:
    """Response (vanilla) distillation — student learns from teacher logits."""

    teacher_model_name: str
    temperature: float = 2.0
    alpha: float = 0.5


@dataclass(frozen=True)
class FeatureDistillationConfig:
    """Feature distillation — student learns from teacher hidden states."""

    teacher_model_name: str
    teacher_layers: Optional[List[int]] = None
    student_layers: Optional[List[int]] = None
    feature_layers: Optional[List[int]] = None
    temperature: float = 2.0
    alpha: float = 0.5
    beta: float = 0.1
    feature_loss_weight: float = 1.0


@dataclass(frozen=True)
class PruningConfig:
    """Structured (magnitude) pruning — attention-head or FFN rows."""

    output_dir: Union[str, Path]
    sparsity: float = 0.3
    method: str = "heads"
    importance_metric: str = "magnitude"
    min_heads_per_layer: int = 1


@dataclass(frozen=True)
class WandaConfig:
    """WANDA unstructured pruning — |W_ij| x ||X_j||2."""

    output_dir: Union[str, Path]
    sparsity: float = 0.5
    n_calibration_samples: int = 128
    calibration_seq_len: int = 512
    use_row_wise: bool = True
    layer_types: Optional[List[str]] = None
