"""Configuration management — Pydantic models + ConfigBuilder."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel, Field, field_validator, model_validator

from .types import (
    TrainingMethod, DatasetSource, DeviceType, EvaluationMetric,
    ModelConfig, DatasetConfig, TokenizationConfig, LoRAConfig,
    TrainingConfig, EvaluationConfig,
)
from .exceptions import InvalidConfigError, MissingConfigError, IncompatibleConfigError


# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class ModelConfigModel(BaseModel):
    name: str
    device: DeviceType = DeviceType.AUTO
    torch_dtype: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = False
    trust_remote_code: bool = False
    revision: Optional[str] = None
    cache_dir: Optional[str] = None

    @field_validator("torch_dtype", mode="before")
    @classmethod
    def validate_dtype(cls, v):
        if v is None:
            return None
        valid = ["float32", "float16", "bfloat16", "auto"]
        if v not in valid:
            raise InvalidConfigError(f"Invalid torch_dtype '{v}'. Must be one of {valid}")
        return v

    @model_validator(mode="after")
    def validate_quantization(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise IncompatibleConfigError(
                "Cannot enable both 8-bit and 4-bit quantization",
                ["load_in_8bit", "load_in_4bit"],
            )
        return self

    def to_config(self) -> ModelConfig:
        dtype_map = {"float32": torch.float32, "float16": torch.float16,
                     "bfloat16": torch.bfloat16, "auto": None}
        return ModelConfig(
            name=self.name,
            device=self.device,
            torch_dtype=dtype_map.get(self.torch_dtype) if self.torch_dtype else None,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            use_flash_attention=self.use_flash_attention,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            cache_dir=Path(self.cache_dir) if self.cache_dir else None,
        )


class DatasetConfigModel(BaseModel):
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

    @model_validator(mode="after")
    def validate_source_path(self):
        if self.source == DatasetSource.LOCAL_FILE:
            if not Path(self.path).exists():
                raise InvalidConfigError(f"Local file not found: {self.path}")
        return self

    def to_config(self) -> DatasetConfig:
        return DatasetConfig(
            source=self.source, path=self.path, split=self.split,
            config_name=self.config_name, data_files=self.data_files,
            text_columns=self.text_columns, max_samples=self.max_samples,
            streaming=self.streaming, shuffle=self.shuffle, seed=self.seed,
        )


class TokenizationConfigModel(BaseModel):
    max_length: int = Field(512, ge=1, le=8192)
    truncation: bool = True
    padding: str = "max_length"
    add_special_tokens: bool = True
    return_attention_mask: bool = True

    @field_validator("padding")
    @classmethod
    def validate_padding(cls, v):
        valid = ["max_length", "longest", "do_not_pad"]
        if v not in valid:
            raise InvalidConfigError(f"Invalid padding '{v}'")
        return v

    def to_config(self) -> TokenizationConfig:
        return TokenizationConfig(
            max_length=self.max_length, truncation=self.truncation,
            padding=self.padding, add_special_tokens=self.add_special_tokens,
            return_attention_mask=self.return_attention_mask,
        )


class LoRAConfigModel(BaseModel):
    r: int = Field(8, ge=1, le=256)
    lora_alpha: int = Field(32, ge=1)
    lora_dropout: float = Field(0.1, ge=0.0, le=0.5)
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    fan_in_fan_out: bool = False
    init_lora_weights: Union[bool, str] = True

    @field_validator("bias")
    @classmethod
    def validate_bias(cls, v):
        valid = ["none", "all", "lora_only"]
        if v not in valid:
            raise InvalidConfigError(f"Invalid bias '{v}'")
        return v

    def to_config(self) -> LoRAConfig:
        return LoRAConfig(
            r=self.r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout,
            target_modules=self.target_modules, bias=self.bias,  # type: ignore
            fan_in_fan_out=self.fan_in_fan_out, init_lora_weights=self.init_lora_weights,  # type: ignore
        )


class TrainingConfigModel(BaseModel):
    method: TrainingMethod
    output_dir: str
    num_epochs: int = Field(3, ge=1)
    batch_size: int = Field(4, ge=1)
    gradient_accumulation_steps: int = Field(4, ge=1)
    learning_rate: float = Field(2e-4, gt=0.0)
    weight_decay: float = Field(0.01, ge=0.0)
    warmup_ratio: float = Field(0.1, ge=0.0)
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = Field(1.0, gt=0.0)
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = Field(10, ge=1)
    save_steps: Optional[int] = None
    save_strategy: str = "epoch"
    evaluation_strategy: str = "no"
    load_best_model_at_end: bool = False
    seed: int = 42
    gradient_checkpointing: bool = False

    @model_validator(mode="after")
    def validate_mixed_precision(self):
        if self.fp16 and self.bf16:
            raise IncompatibleConfigError("Cannot enable both FP16 and BF16", ["fp16", "bf16"])
        return self

    def to_config(self) -> TrainingConfig:
        return TrainingConfig(
            method=self.method, output_dir=Path(self.output_dir),
            num_epochs=self.num_epochs, batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate, weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio, lr_scheduler_type=self.lr_scheduler_type,
            max_grad_norm=self.max_grad_norm, fp16=self.fp16, bf16=self.bf16,
            logging_steps=self.logging_steps, save_steps=self.save_steps,
            save_strategy=self.save_strategy,  # type: ignore
            evaluation_strategy=self.evaluation_strategy,  # type: ignore
            load_best_model_at_end=self.load_best_model_at_end,
            seed=self.seed, gradient_checkpointing=self.gradient_checkpointing,
        )


class EvaluationConfigModel(BaseModel):
    metrics: List[EvaluationMetric]
    batch_size: int = Field(8, ge=1)
    num_samples: Optional[int] = None
    generation_max_length: int = Field(100, ge=1)
    generation_temperature: float = Field(0.7, ge=0.0, le=2.0)
    generation_top_p: float = Field(0.9, ge=0.0, le=1.0)
    generation_do_sample: bool = True

    def to_config(self) -> EvaluationConfig:
        return EvaluationConfig(
            metrics=self.metrics, batch_size=self.batch_size,
            num_samples=self.num_samples, generation_max_length=self.generation_max_length,
            generation_temperature=self.generation_temperature,
            generation_top_p=self.generation_top_p,
            generation_do_sample=self.generation_do_sample,
        )


# ============================================================================
# PIPELINE CONFIG
# ============================================================================


class PipelineConfig(BaseModel):
    model: ModelConfigModel
    dataset: DatasetConfigModel
    tokenization: TokenizationConfigModel
    training: TrainingConfigModel
    evaluation: Optional[EvaluationConfigModel] = None
    lora: Optional[LoRAConfigModel] = None

    @model_validator(mode="after")
    def validate_method_config(self):
        if self.training.method == TrainingMethod.LORA and self.lora is None:
            raise MissingConfigError("lora", "PipelineConfig")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        return cls(**d)

    @classmethod
    def from_json(cls, path: Path) -> "PipelineConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        # model_dump with mode='json' converts enums to strings
        data = json.loads(self.model_dump_json())
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


# ============================================================================
# CONFIG BUILDER
# ============================================================================


class ConfigBuilder:
    def __init__(self):
        self._config: Dict[str, Any] = {}

    def with_model(self, name: str, **kwargs) -> "ConfigBuilder":
        self._config["model"] = {"name": name, **kwargs}
        return self

    def with_dataset(self, path: str, source: DatasetSource = DatasetSource.LOCAL_FILE,
                     **kwargs) -> "ConfigBuilder":
        self._config["dataset"] = {"source": source, "path": path, **kwargs}
        return self

    def with_tokenization(self, **kwargs) -> "ConfigBuilder":
        self._config["tokenization"] = kwargs
        return self

    def with_training(self, method: TrainingMethod, output_dir: str,
                      **kwargs) -> "ConfigBuilder":
        self._config["training"] = {"method": method, "output_dir": output_dir, **kwargs}
        return self

    def with_lora(self, **kwargs) -> "ConfigBuilder":
        self._config["lora"] = kwargs
        return self

    def with_evaluation(self, metrics: List[EvaluationMetric], **kwargs) -> "ConfigBuilder":
        self._config["evaluation"] = {"metrics": metrics, **kwargs}
        return self

    def build(self) -> PipelineConfig:
        return PipelineConfig.from_dict(self._config)