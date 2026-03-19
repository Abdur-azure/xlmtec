"""
tests/test_hub.py
~~~~~~~~~~~~~~~~~~
Tests for xlmtec hub — search, info, trending, formatter, client.
All HuggingFace API calls are mocked — no network required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("xlmtec.hub", reason="xlmtec.hub not found — run: pip install -e '.[dev]'")

from xlmtec.hub.client import ModelDetail, ModelSummary


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_hf_model(
    model_id="bert-base-uncased",
    downloads=100_000,
    likes=500,
    pipeline_tag="text-classification",
    tags=None,
    last_modified="2024-01-01",
):
    m = MagicMock()
    m.modelId = model_id
    m.downloads = downloads
    m.likes = likes
    m.pipeline_tag = pipeline_tag
    m.tags = tags or ["pytorch", "bert"]
    m.lastModified = last_modified
    m.author = model_id.split("/")[0] if "/" in model_id else ""
    m.sha = "abc123"
    m.private = False
    m.library_name = "transformers"
    m.siblings = []
    return m


@pytest.fixture
def mock_api():
    return MagicMock()


@pytest.fixture
def sample_summary():
    return ModelSummary(
        model_id="bert-base-uncased",
        downloads=100_000,
        likes=500,
        task="text-classification",
        tags=["pytorch", "bert"],
    )


@pytest.fixture
def sample_detail():
    return ModelDetail(
        model_id="google/bert-base-uncased",
        downloads=10_000_000,
        likes=2000,
        task="fill-mask",
        tags=["pytorch", "bert", "en"],
        author="google",
        library="transformers",
        languages=["en"],
        size_mb=420.0,
    )


# ---------------------------------------------------------------------------
# HubClient.search
# ---------------------------------------------------------------------------

class TestHubClientSearch:
    def test_returns_model_summaries(self, mock_api):
        from xlmtec.hub.client import HubClient
        mock_api.list_models.return_value = [_make_hf_model()]
        with patch("xlmtec.hub.client.HfApi", return_value=mock_api):
            client = HubClient()
            results = client.search("bert")
        assert len(results) == 1
        assert isinstance(results[0], ModelSummary)
        assert results[0].model_id == "bert-base-uncased"

    def test_returns_empty_list_when_no_results(self, mock_api):
        from xlmtec.hub.client import HubClient
        mock_api.list_models.return_value = []
        with patch("xlmtec.hub.client.HfApi", return_value=mock_api):
            client = HubClient()
            results = client.search("nonexistent_model_xyz")
        assert results == []

    def test_limit_respected(self, mock_api):
        from xlmtec.hub.client import HubClient
        mock_api.list_models.return_value = [_make_hf_model(f"model/{i}") for i in range(20)]
        with patch("xlmtec.hub.client.HfApi", return_value=mock_api):
            client = HubClient()
            results = client.search("bert", limit=5)
        assert len(results) <= 5


# ---------------------------------------------------------------------------
# HubClient.model_info
# ---------------------------------------------------------------------------

class TestHubClientModelInfo:
    def test_returns_model_detail(self, mock_api):
        from xlmtec.hub.client import HubClient
        mock_api.model_info.return_value = _make_hf_model("bert-base-uncased")
        with patch("xlmtec.hub.client.HfApi", return_value=mock_api):
            client = HubClient()
            detail = client.model_info("bert-base-uncased")
        assert isinstance(detail, ModelDetail)
        assert detail.model_id == "bert-base-uncased"


# ---------------------------------------------------------------------------
# HubClient.trending
# ---------------------------------------------------------------------------

class TestHubClientTrending:
    def test_returns_summaries(self, mock_api):
        from xlmtec.hub.client import HubClient
        mock_api.list_models.return_value = [_make_hf_model()]
        with patch("xlmtec.hub.client.HfApi", return_value=mock_api):
            client = HubClient()
            results = client.trending()
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_summary_defaults(self):
        s = ModelSummary(model_id="test/model")
        assert s.downloads == 0
        assert s.tags == []

    def test_detail_inherits_summary(self):
        d = ModelDetail(model_id="test/model", author="test")
        assert isinstance(d, ModelSummary)
        assert d.author == "test"


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class TestFormatter:
    def test_print_search_results_no_crash(self, sample_summary):
        from xlmtec.hub.formatter import print_search_results
        print_search_results([sample_summary], "bert")

    def test_print_search_empty_no_crash(self):
        from xlmtec.hub.formatter import print_search_results
        print_search_results([], "bert")

    def test_print_trending_no_crash(self, sample_summary):
        from xlmtec.hub.formatter import print_trending
        print_trending([sample_summary])

    def test_print_model_info_no_crash(self, sample_detail):
        from xlmtec.hub.formatter import print_model_info
        print_model_info(sample_detail)

    def test_fmt_number_millions(self):
        from xlmtec.hub.formatter import _fmt_number
        assert _fmt_number(1_500_000) == "1.5M"

    def test_fmt_number_thousands(self):
        from xlmtec.hub.formatter import _fmt_number
        assert _fmt_number(2_500) == "2.5K"

    def test_fmt_number_small(self):
        from xlmtec.hub.formatter import _fmt_number
        assert _fmt_number(42) == "42"


# ---------------------------------------------------------------------------
# HubClient missing SDK
# ---------------------------------------------------------------------------

class TestHubClientMissingSDK:
    def test_raises_import_error_when_hfapi_none(self):
        from xlmtec.hub.client import HubClient
        with patch("xlmtec.hub.client.HfApi", None):
            with pytest.raises(ImportError, match="huggingface-hub"):
                HubClient()