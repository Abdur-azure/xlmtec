"""Core type definitions — enums, dataclasses, protocols."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

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
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: Literal["none", "all", "lora_only"] = "none"
    fan_in_fan_out: bool = False
    init_lora_weights: Union[bool, Literal["gaussian", "loftq"]] = True


@dataclass(frozen=True)
class DistillationConfig:
    """Configuration for knowledge distillation training.

    Attributes:
        teacher_model_name: HuggingFace model id of the teacher.
        temperature: Softmax temperature for softening distributions.
            Higher = softer targets. Recommended range: 2.0-6.0.
        alpha: Distillation loss weight.
            Loss = alpha * KL_loss + (1 - alpha) * CE_loss.
    """
    teacher_model_name: str
    temperature: float = 2.0
    alpha: float = 0.5


@dataclass(frozen=True)
class FeatureDistillationConfig:
    """Configuration for feature (intermediate layer) distillation.

    Attributes:
        teacher_model_name: HuggingFace model id of the teacher.
        temperature: Softmax temperature for output-level KL component.
        alpha: Weight for CE loss.
            Total = alpha*CE + beta*KL_output + (1-alpha-beta)*MSE_features.
        beta: Weight for output KL loss component (set 0.0 to skip).
        feature_layers: Student layer indices to match against teacher.
            None = auto-select evenly-spaced layers across the student depth.
        feature_loss_weight: Scalar multiplier for the MSE feature loss term.
    """
    teacher_model_name: str
    temperature: float = 2.0
    alpha: float = 0.3
    beta: float = 0.2
    feature_layers: Optional[List[int]] = None
    feature_loss_weight: float = 1.0




@dataclass(frozen=True)
class WandaConfig:
    """Configuration for WANDA (Weight AND Activation) unstructured pruning.

    WANDA scores each weight by |W_ij| * ||X_j||_2 where W is the weight
    matrix and X is the input activation norm collected on a small calibration
    dataset.  Weights with the lowest scores are zeroed — no gradient pass or
    retraining is required.

    Attributes:
        output_dir:        Where the pruned model will be saved.
        sparsity:          Target fraction of weights to zero per linear layer.
                           0.0 = no pruning; 0.5 = half the weights zeroed.
                           Recommended: 0.3–0.5.
        n_calibration_samples: Number of calibration forward passes used to
                           collect activation norms. More = more accurate
                           scoring but slower. Default: 128.
        calibration_seq_len: Token sequence length for calibration inputs.
        layer_types:       Which nn.Module class names to prune. Default covers
                           all linear projections in standard transformer models.
        use_row_wise:      If True, apply sparsity row-by-row (each output
                           neuron pruned independently). If False, apply
                           globally across the full weight matrix.
    """
    output_dir: Path
    sparsity: float = 0.5
    n_calibration_samples: int = 128
    calibration_seq_len: int = 128
    layer_types: Optional[List[str]] = None
    use_row_wise: bool = True

@dataclass(frozen=True)
class PruningConfig:
    """Configuration for structured pruning.

    Structured pruning removes entire attention heads (or FFN neurons)
    whose weight magnitudes rank below a sparsity threshold.
    No retraining is required — the model is modified in-place and saved.

    Attributes:
        output_dir:           Where the pruned model will be saved.
        sparsity:             Fraction of attention heads to prune per layer.
                              0.0 = no pruning; 1.0 = prune everything (clamped
                              by min_heads_per_layer). Recommended: 0.1–0.4.
        method:               Pruning target — "heads" (attention heads) or
                              "ffn" (feed-forward neurons). Default: "heads".
        importance_metric:    How to score head/neuron importance.
                              Currently only "magnitude" (mean |w|) is supported.
        min_heads_per_layer:  Safety floor — never prune a layer below this many
                              heads. Prevents complete layer collapse. Default: 1.
    """
    output_dir: Path
    sparsity: float = 0.3
    method: str = "heads"
    importance_metric: str = "magnitude"
    min_heads_per_layer: int = 1

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
