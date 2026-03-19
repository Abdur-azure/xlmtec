"""
Shared pytest fixtures for all unit tests.

No torch import at module level — tests must be collectable without torch installed.
Heavy deps (torch, transformers) are only imported inside test functions that need them,
or in test_integration.py which guards with pytest.importorskip.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure repo root is on sys.path for absolute imports in tests/
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_model():
    """
    Lightweight mock model — no real tensors, no torch required.

    param.numel() returns a fixed int so parameter-count logic works.
    param.requires_grad is True so trainable-param checks pass.
    """
    model = MagicMock()
    param = MagicMock()
    param.numel.return_value = 1_000_000
    param.requires_grad = True
    # parameters() is called multiple times; use a factory so each call
    # returns a fresh iterator rather than an exhausted one.
    model.parameters.side_effect = lambda: iter([param])
    model.named_modules.return_value = []
    return model


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.pad_token = "[PAD]"
    tok.pad_token_id = 0
    tok.eos_token = "</s>"
    tok.eos_token_id = 2
    tok.model_max_length = 512
    return tok


@pytest.fixture
def tiny_dataset():
    Dataset = pytest.importorskip("datasets").Dataset
    texts = ["The quick brown fox jumps over the lazy dog."] * 10
    return Dataset.from_dict({"text": texts})


@pytest.fixture
def tmp_output_dir(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    return d
