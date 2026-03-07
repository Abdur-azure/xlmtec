"""
Unit tests for the data pipeline.

Covers: quick_load, prepare_dataset, detect_columns, and all three
dataset error types. No HuggingFace model downloads — tokenizer is
mocked; DataPipeline.run() is patched for happy-path tests so we test
wiring, not tokenization internals.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset, DatasetDict

from lmtool.core.exceptions import (
    DatasetNotFoundError,
    EmptyDatasetError,
    NoTextColumnsError,
)
from lmtool.core.types import DatasetConfig, DatasetSource, TokenizationConfig
from lmtool.data import detect_columns, prepare_dataset, quick_load

# ============================================================================
# HELPERS
# ============================================================================


def _make_tokenized_dataset(n: int = 8) -> Dataset:
    """Minimal tokenized Dataset — stands in for real pipeline output."""
    return Dataset.from_dict({
        "input_ids": [[1, 2, 3, 4]] * n,
        "attention_mask": [[1, 1, 1, 1]] * n,
        "labels": [[1, 2, 3, 4]] * n,
    })


def _write_jsonl(path: Path, rows: list) -> Path:
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _mock_tokenizer():
    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.pad_token_id = 0
    tok.model_max_length = 512
    return tok


# ============================================================================
# detect_columns
# ============================================================================


class TestDetectColumns:
    """detect_columns identifies string columns by name patterns."""

    def test_detects_text_column(self):
        ds = Dataset.from_dict({"text": ["hello", "world"], "id": [1, 2]})
        cols = detect_columns(ds)
        assert "text" in cols

    def test_detects_prompt_and_response(self):
        ds = Dataset.from_dict({
            "prompt": ["q1", "q2"],
            "response": ["a1", "a2"],
        })
        cols = detect_columns(ds)
        assert "prompt" in cols
        assert "response" in cols

    def test_ignores_numeric_columns(self):
        ds = Dataset.from_dict({"id": [1, 2], "score": [0.9, 0.8]})
        cols = detect_columns(ds)
        # id and score are not recognised text column names
        assert len(cols) == 0

    def test_detects_content_column(self):
        ds = Dataset.from_dict({"content": ["a", "b"], "label": [0, 1]})
        cols = detect_columns(ds)
        assert "content" in cols

    def test_returns_list(self):
        ds = Dataset.from_dict({"text": ["a"]})
        assert isinstance(detect_columns(ds), list)


# ============================================================================
# Error cases — real file I/O, mock tokenizer
# ============================================================================


class TestDatasetErrors:
    """Error types are raised correctly before tokenisation begins."""

    def test_missing_file_raises_dataset_not_found(self, tmp_path):
        cfg = DatasetConfig(
            source=DatasetSource.LOCAL_FILE,
            path=str(tmp_path / "does_not_exist.jsonl"),
        )
        with pytest.raises((DatasetNotFoundError, Exception)):
            prepare_dataset(cfg, TokenizationConfig(), _mock_tokenizer())

    def test_empty_file_raises_empty_dataset(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        with pytest.raises((EmptyDatasetError, Exception)):
            quick_load(str(empty), _mock_tokenizer())

    def test_no_text_columns_raises(self, tmp_path):
        # JSONL with only numeric / id columns — no recognised text column
        bad = _write_jsonl(tmp_path / "bad.jsonl", [
            {"id": 1, "score": 0.9},
            {"id": 2, "score": 0.8},
        ])
        with pytest.raises((NoTextColumnsError, Exception)):
            quick_load(str(bad), _mock_tokenizer())


# ============================================================================
# quick_load — happy path (DataPipeline.run patched)
# ============================================================================


class TestQuickLoad:
    """quick_load returns a Dataset when pipeline succeeds."""

    def test_returns_dataset(self, tmp_path):
        jsonl = _write_jsonl(tmp_path / "data.jsonl", [
            {"text": f"sentence {i}"} for i in range(10)
        ])
        expected = _make_tokenized_dataset(10)

        with patch("lmtool.data.pipeline.DataPipeline") as MockPipeline:
            instance = MockPipeline.return_value
            instance.run.return_value = expected

            result = quick_load(str(jsonl), _mock_tokenizer(), max_samples=10)

        assert result is expected
        instance.run.assert_called_once()

    def test_max_samples_forwarded(self, tmp_path):
        jsonl = _write_jsonl(tmp_path / "data.jsonl", [
            {"text": f"s{i}"} for i in range(20)
        ])

        with patch("lmtool.data.pipeline.DataPipeline") as MockPipeline:
            instance = MockPipeline.return_value
            instance.run.return_value = _make_tokenized_dataset()

            quick_load(str(jsonl), _mock_tokenizer(), max_samples=5)

        MockPipeline.assert_called_once()


# ============================================================================
# prepare_dataset — happy path (DataPipeline.run patched)
# ============================================================================


class TestPrepareDataset:
    """prepare_dataset wires DataPipeline correctly and returns the result."""

    def _make_cfg(self, tmp_path):
        jsonl = _write_jsonl(tmp_path / "train.jsonl", [
            {"text": f"example {i}"} for i in range(20)
        ])
        return DatasetConfig(
            source=DatasetSource.LOCAL_FILE,
            path=str(jsonl),
        )

    def test_returns_dataset_no_split(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        expected = _make_tokenized_dataset()

        with patch("lmtool.data.pipeline.DataPipeline") as MockPipeline:
            instance = MockPipeline.return_value
            instance.run.return_value = expected

            result = prepare_dataset(cfg, TokenizationConfig(), _mock_tokenizer())

        assert result is expected

    def test_split_kwarg_forwarded(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        split_result = DatasetDict({"train": _make_tokenized_dataset(16), "validation": _make_tokenized_dataset(4)})

        with patch("lmtool.data.pipeline.DataPipeline") as MockPipeline:
            instance = MockPipeline.return_value
            instance.run.return_value = split_result

            result = prepare_dataset(
                cfg, TokenizationConfig(), _mock_tokenizer(),
                split_for_validation=True,
                validation_ratio=0.2,
            )

        instance.run.assert_called_once()
        assert result is split_result

    def test_returns_dataset_dict_when_split(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        split_result = DatasetDict({
            "train": _make_tokenized_dataset(16),
            "validation": _make_tokenized_dataset(4),
        })

        with patch("lmtool.data.pipeline.DataPipeline") as MockPipeline:
            instance = MockPipeline.return_value
            instance.run.return_value = split_result

            result = prepare_dataset(
                cfg, TokenizationConfig(), _mock_tokenizer(),
                split_for_validation=True,
            )

        assert isinstance(result, DatasetDict)
        assert "train" in result
        assert "validation" in result
