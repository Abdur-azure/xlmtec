"""
xlmtec.core.config_builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~
ConfigBuilder — fluent API for constructing a PipelineConfig from code.

This is the *only* correct way to build a PipelineConfig programmatically.
Use PipelineConfig.from_yaml() / from_json() for file-based loading.

Example:
    config = (
        ConfigBuilder()
        .with_model("gpt2")
        .with_dataset("./data.jsonl", source=DatasetSource.LOCAL_FILE)
        .with_tokenization(max_length=256)
        .with_training(TrainingMethod.LORA, "./output", num_epochs=5)
        .with_lora(r=16, lora_alpha=32)
        .build()
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .types import DatasetSource, EvaluationMetric, TrainingMethod


class ConfigBuilder:
    """Fluent builder for PipelineConfig."""

    def __init__(self) -> None:
        self._cfg: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def with_model(self, name: str, **kwargs) -> "ConfigBuilder":
        self._cfg["model"] = {"name": name, **kwargs}
        return self

    def with_dataset(
        self,
        path: str,
        source: DatasetSource = DatasetSource.LOCAL_FILE,
        **kwargs,
    ) -> "ConfigBuilder":
        self._cfg["dataset"] = {"source": source, "path": path, **kwargs}
        return self

    def with_tokenization(self, **kwargs) -> "ConfigBuilder":
        self._cfg["tokenization"] = kwargs
        return self

    def with_training(
        self,
        method: TrainingMethod,
        output_dir: str,
        **kwargs,
    ) -> "ConfigBuilder":
        self._cfg["training"] = {"method": method, "output_dir": output_dir, **kwargs}
        return self

    def with_lora(self, **kwargs) -> "ConfigBuilder":
        self._cfg["lora"] = kwargs
        return self

    def with_evaluation(
        self,
        metrics: List[EvaluationMetric],
        **kwargs,
    ) -> "ConfigBuilder":
        self._cfg["evaluation"] = {"metrics": metrics, **kwargs}
        return self

    def build(self):
        """Validate and return a PipelineConfig.

        Raises:
            MissingConfigError: If required sections are absent.
            InvalidConfigError: If values fail validation.
        """
        # Late import to avoid circular deps at module load time
        from .config import PipelineConfig
        return PipelineConfig.from_dict(self._cfg)