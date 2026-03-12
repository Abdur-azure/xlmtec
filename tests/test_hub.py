"""
tests/test_hub.py
~~~~~~~~~~~~~~~~~~
Tests for xlmtec hub — search, info, trending, formatter, client.
All HuggingFace API calls are mocked — no network required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
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

    def test_passes_task_filter(self, mock_api):
        from xlmtec.hub.client import HubClient

        mock_api.list_models.return_value = []
        with patch("xlmtec.hub.client.HfApi", return_value=mock_api):
            client = HubClient()
            client.search("bert", task="text-classification")
        call_kwargs = mock_api.list_models.call_args.kwargs
        assert call_kwargs["filter"] == "text-classification"

    def test_limit_clamped_to_100(self, mock_api):
        from xlmtec.hub.client import HubClient

        mock_api.list_models.return_value = []
        with patch("xlmtec.hub.client.HfApi", return_value=mock_api):
            client = HubClient()
            client.search("bert", limit=999)
        assert mock_api.list_models.call_args.kwargs["limit"] == 100

    def test_limit_clamped_to_1(self, mock_api):
        from xlmtec.hub.client import HubClient

        mock_api.list_models.return_value = []
        with patch("xlmtec.hub.client.HfApi", return_value=mock_api):
            client = HubClient()
            client.search("bert", limit=0)
        assert mock_api.list_models.call_args.kwargs["limit"] == 1

    def test_empty_results(self, mock_api):
        from xlmtec.hub.client import HubClient

        mock_api.list_models.return_value = []
        with patch("xlmtec.hub.client.HfApi", return_value=mock_api):
            client = HubClient()
            assert client.search("zzznomatch") == []


# ---------------------------------------------------------------------------
# HubClient.trending
# ---------------------------------------------------------------------------


class TestHubClientTrending:
    def test_returns_summaries(self, mock_api):
        from xlmtec.hub.client import HubClient

        mock_api.list_models.return_value = [_make_hf_model(), _make_hf_model("gpt2")]
        with patch("xlmtec.hub.client.HfApi", return_value=mock_api):
            results = HubClient().trending(limit=2)
        assert len(results) == 2
        assert all(isinstance(r, ModelSummary) for r in results)

    def test_sorts_by_downloads(self, mock_api):
        from xlmtec.hub.client import HubClient

        mock_api.list_models.return_value = []
        with patch("xlmtec.hub.client.HfApi", return_value=mock_api):
            HubClient().trending()
        assert mock_api.list_models.call_args.kwargs["sort"] == "downloads"


# ---------------------------------------------------------------------------
# HubClient.info
# ---------------------------------------------------------------------------


class TestHubClientInfo:
    def test_returns_model_detail(self, mock_api):
        from xlmtec.hub.client import HubClient

        mock_api.model_info.return_value = _make_hf_model("google/bert-base-uncased")
        with patch("xlmtec.hub.client.HfApi", return_value=mock_api):
            detail = HubClient().info("google/bert-base-uncased")
        assert isinstance(detail, ModelDetail)
        assert detail.model_id == "google/bert-base-uncased"

    def test_not_found_raises_value_error(self, mock_api):
        from xlmtec.hub.client import HubClient, RepositoryNotFoundError

        mock_api.model_info.side_effect = RepositoryNotFoundError("not found")
        with patch("xlmtec.hub.client.HfApi", return_value=mock_api):
            with pytest.raises(ValueError, match="Model not found"):
                HubClient().info("bad/model")

    def test_api_error_raises_runtime_error(self, mock_api):
        from xlmtec.hub.client import HubClient

        mock_api.model_info.side_effect = Exception("timeout")
        with patch("xlmtec.hub.client.HfApi", return_value=mock_api):
            with pytest.raises(RuntimeError, match="HuggingFace API error"):
                HubClient().info("some/model")

    def test_size_calculated_from_siblings(self, mock_api):
        from xlmtec.hub.client import HubClient

        m = _make_hf_model()
        sibling = MagicMock()
        sibling.size = 1024 * 1024 * 100  # 100 MB
        m.siblings = [sibling]
        mock_api.model_info.return_value = m
        with patch("xlmtec.hub.client.HfApi", return_value=mock_api):
            detail = HubClient().info("bert-base-uncased")
        assert detail.size_mb == 100.0


# ---------------------------------------------------------------------------
# ModelSummary / ModelDetail dataclasses
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
