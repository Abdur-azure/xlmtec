"""Core type definitions — enums, dataclasses, protocols."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from dataclasses import dataclass, field
from typing import Dict

import torch


class TrainingMethod(str, Enum):
    FULL_FINETUNING = "full_finetuning"
    LORA = "lora"
    QLORA = "qlora"
    ADALORA = "adalora"
    PREFIX_TUNING = "prefix_tuning"
    P_TUNING = "p_tuning"
    PROMPT_TUNING = "prompt_tuning"
    IA3 = "ia3"
    INSTRUCTION_TUNING = "instruction_tuning"
    RLHF = "rlhf"
    DPO = "dpo"
    RLAIF = "rlaif"
    SIMCSE = "simcse"
    CONTRASTIVE_ALIGNMENT = "contrastive_alignment"
    VANILLA_DISTILLATION = "vanilla_distillation"
    FEATURE_DISTILLATION = "feature_distillation"
    SELF_DISTILLATION = "self_distillation"
    QAT_INT8 = "qat_int8"
    QAT_INT4 = "qat_int4"
    MAGNITUDE_PRUNING = "magnitude_pruning"
    STRUCTURED_PRUNING = "structured_pruning"


class DatasetSource(str, Enum):
    LOCAL_FILE = "local_file"
    HUGGINGFACE_HUB = "huggingface_hub"
    CUSTOM = "custom"


class FileFormat(str, Enum):
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    TXT = "txt"


class EvaluationMetric(str, Enum):
    ROUGE_1 = "rouge1"
    ROUGE_2 = "rouge2"
    ROUGE_L = "rougeL"
    BLEU = "bleu"
    PERPLEXITY = "perplexity"
    ACCURACY = "accuracy"
    F1 = "f1"
    EXACT_MATCH = "exact_match"


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


@dataclass(frozen=True)
class ModelConfig:
    name: str
    device: DeviceType = DeviceType.AUTO
    torch_dtype: Optional[Any] = None
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
class DistillationConfig:
    """Configuration for knowledge distillation training.

    Attributes:
        teacher_model_name: HuggingFace model id of the teacher.
        temperature: Softmax temperature for softening distributions.
            Higher = softer targets. Recommended range: 2.0–6.0.
        alpha: Distillation loss weight.
            Loss = alpha * KL_loss + (1 - alpha) * CE_loss.
    """
    teacher_model_name: str
    temperature: float = 2.0
    alpha: float = 0.5

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
    output_dir: Path
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 10
    save_steps: Optional[int] = None
    save_strategy: Literal["no", "epoch", "steps"] = "epoch"
    evaluation_strategy: Literal["no", "epoch", "steps"] = "no"
    load_best_model_at_end: bool = False
    seed: int = 42
    gradient_checkpointing: bool = False


@dataclass(frozen=True)
class EvaluationConfig:
    metrics: List[EvaluationMetric]
    batch_size: int = 8
    num_samples: Optional[int] = None
    generation_max_length: int = 100
    generation_temperature: float = 0.7
    generation_top_p: float = 0.9
    generation_do_sample: bool = True

@dataclass(frozen=True)
class EvaluationResult:
    model_label: str
    scores: Dict[str, float]
    num_samples: int
    elapsed_seconds: float