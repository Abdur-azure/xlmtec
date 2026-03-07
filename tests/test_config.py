"""Tests for core configuration — ConfigBuilder and Pydantic validation."""

from pathlib import Path

import pytest

from lmtool.core.config import ConfigBuilder, PipelineConfig
from lmtool.core.exceptions import (
    IncompatibleConfigError,
    InvalidConfigError,
    MissingConfigError,
)
from lmtool.core.types import DatasetSource, EvaluationMetric, TrainingMethod


class TestConfigBuilder:
    """ConfigBuilder produces valid PipelineConfig objects."""

    def _base_builder(self, tmp_path):
        # Create a dummy local file so validation passes
        f = tmp_path / "data.jsonl"
        f.write_text('{"text": "hello"}\n')
        return (
            ConfigBuilder()
            .with_model("gpt2")
            .with_dataset(str(f), source=DatasetSource.LOCAL_FILE)
            .with_tokenization(max_length=128)
            .with_training(
                method=TrainingMethod.LORA,
                output_dir=str(tmp_path / "output"),
            )
            .with_lora(r=8, lora_alpha=16)
        )

    def test_build_returns_pipeline_config(self, tmp_path):
        config = self._base_builder(tmp_path).build()
        assert isinstance(config, PipelineConfig)

    def test_model_name_set_correctly(self, tmp_path):
        config = self._base_builder(tmp_path).build()
        assert config.model.name == "gpt2"

    def test_tokenization_max_length(self, tmp_path):
        config = self._base_builder(tmp_path).build()
        assert config.tokenization.max_length == 128

    def test_lora_config_present(self, tmp_path):
        config = self._base_builder(tmp_path).build()
        assert config.lora is not None
        assert config.lora.r == 8
        assert config.lora.lora_alpha == 16

    def test_lora_required_when_method_is_lora(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"text": "hello"}\n')
        with pytest.raises(Exception):
            ConfigBuilder() \
                .with_model("gpt2") \
                .with_dataset(str(f), source=DatasetSource.LOCAL_FILE) \
                .with_tokenization() \
                .with_training(TrainingMethod.LORA, str(tmp_path / "out")) \
                .build()  # No .with_lora() → should raise MissingConfigError

    def test_invalid_dtype_raises(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"text": "hello"}\n')
        with pytest.raises(Exception):
            ConfigBuilder() \
                .with_model("gpt2", torch_dtype="float99") \
                .with_dataset(str(f), source=DatasetSource.LOCAL_FILE) \
                .with_tokenization() \
                .with_training(TrainingMethod.LORA, str(tmp_path / "out")) \
                .with_lora() \
                .build()

    def test_incompatible_quantization_raises(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"text": "hello"}\n')
        with pytest.raises(Exception):
            ConfigBuilder() \
                .with_model("gpt2", load_in_4bit=True, load_in_8bit=True) \
                .with_dataset(str(f), source=DatasetSource.LOCAL_FILE) \
                .with_tokenization() \
                .with_training(TrainingMethod.LORA, str(tmp_path / "out")) \
                .with_lora() \
                .build()

    def test_fp16_bf16_exclusive(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"text": "hello"}\n')
        with pytest.raises(Exception):
            ConfigBuilder() \
                .with_model("gpt2") \
                .with_dataset(str(f), source=DatasetSource.LOCAL_FILE) \
                .with_tokenization() \
                .with_training(
                    TrainingMethod.LORA,
                    str(tmp_path / "out"),
                    fp16=True,
                    bf16=True,
                ) \
                .with_lora() \
                .build()


class TestPipelineConfigSerialization:
    """PipelineConfig round-trips through JSON and YAML."""

    def _make_config(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"text": "hello"}\n')
        return (
            ConfigBuilder()
            .with_model("gpt2")
            .with_dataset(str(f), source=DatasetSource.LOCAL_FILE)
            .with_tokenization(max_length=256)
            .with_training(TrainingMethod.LORA, str(tmp_path / "output"))
            .with_lora(r=4)
            .build()
        )

    def test_json_round_trip(self, tmp_path):
        config = self._make_config(tmp_path)
        json_path = tmp_path / "config.json"
        config.to_json(json_path)
        loaded = PipelineConfig.from_json(json_path)
        assert loaded.model.name == config.model.name
        assert loaded.tokenization.max_length == 256

    def test_yaml_round_trip(self, tmp_path):
        config = self._make_config(tmp_path)
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)
        loaded = PipelineConfig.from_yaml(yaml_path)
        assert loaded.model.name == config.model.name
        assert loaded.lora.r == 4
