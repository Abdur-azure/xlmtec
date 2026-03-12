"""
tests/test_integrations.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for xlmtec AI integrations.
All API calls are mocked — no real keys needed, no network required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from xlmtec.integrations.base import SuggestResult
from xlmtec.integrations.response_parser import parse_response

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_JSON = {
    "method": "lora",
    "yaml_config": (
        "model:\n  name: gpt2\n"
        "dataset:\n  source: local_file\n  path: data/train.jsonl\n"
        "lora:\n  r: 16\n  alpha: 32\n"
        "training:\n  output_dir: output/run1\n  num_epochs: 3\n"
    ),
    "explanation": "LoRA is ideal for this task — low VRAM, fast convergence.",
}
SAMPLE_RAW = json.dumps(SAMPLE_JSON)


@pytest.fixture
def sample_result() -> SuggestResult:
    return SuggestResult(
        method="lora",
        yaml_config=SAMPLE_JSON["yaml_config"],
        explanation=SAMPLE_JSON["explanation"],
        command="xlmtec train --method lora --config config.yaml --output-dir output/run1",
        raw=SAMPLE_RAW,
    )


# ---------------------------------------------------------------------------
# response_parser
# ---------------------------------------------------------------------------


class TestResponseParser:
    def test_clean_json(self):
        result = parse_response(SAMPLE_RAW)
        assert result.method == "lora"
        assert "gpt2" in result.yaml_config
        assert "LoRA" in result.explanation

    def test_json_with_markdown_fences(self):
        assert parse_response(f"```json\n{SAMPLE_RAW}\n```").method == "lora"

    def test_json_with_plain_fences(self):
        assert parse_response(f"```\n{SAMPLE_RAW}\n```").method == "lora"

    def test_json_embedded_in_prose(self):
        assert (
            parse_response(f"Here is my suggestion:\n{SAMPLE_RAW}\nHope that helps!").method
            == "lora"
        )

    def test_command_built_from_output_dir(self):
        result = parse_response(SAMPLE_RAW)
        assert "output/run1" in result.command
        assert result.command.startswith("xlmtec train")

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            parse_response("this is not json at all")

    def test_raw_preserved(self):
        assert parse_response(SAMPLE_RAW).raw == SAMPLE_RAW


# ---------------------------------------------------------------------------
# get_provider
# ---------------------------------------------------------------------------


class TestGetProvider:
    def test_returns_claude(self):
        from xlmtec.integrations import get_provider
        from xlmtec.integrations.claude import ClaudeIntegration

        assert isinstance(get_provider("claude", api_key="sk-test"), ClaudeIntegration)

    def test_returns_gemini(self):
        from xlmtec.integrations import get_provider
        from xlmtec.integrations.gemini import GeminiIntegration

        assert isinstance(get_provider("gemini", api_key="test"), GeminiIntegration)

    def test_returns_codex(self):
        from xlmtec.integrations import get_provider
        from xlmtec.integrations.codex import CodexIntegration

        assert isinstance(get_provider("codex", api_key="sk-test"), CodexIntegration)

    def test_case_insensitive(self):
        from xlmtec.integrations import get_provider

        assert get_provider("CLAUDE", api_key="sk-test").PROVIDER_NAME == "claude"

    def test_unknown_provider_raises(self):
        from xlmtec.integrations import get_provider

        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("grok")


# ---------------------------------------------------------------------------
# ClaudeIntegration
# ---------------------------------------------------------------------------


class TestClaudeIntegration:
    def test_suggest_returns_result(self):
        from xlmtec.integrations.claude import ClaudeIntegration

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
            content=[MagicMock(text=SAMPLE_RAW)]
        )

        with patch("xlmtec.integrations.claude.anthropic", mock_anthropic):
            result = ClaudeIntegration(api_key="sk-test").suggest("fine-tune GPT-2")

        assert isinstance(result, SuggestResult)
        assert result.method == "lora"

    def test_no_api_key_raises(self):
        from xlmtec.integrations.claude import ClaudeIntegration

        mock_anthropic = MagicMock()
        # Instantiate INSIDE the env-clear so __init__ sees no key
        with patch("xlmtec.integrations.claude.anthropic", mock_anthropic):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False):
                integration = ClaudeIntegration(api_key=None)
                with pytest.raises(RuntimeError, match="No API key"):
                    integration.suggest("test")

    def test_missing_sdk_raises_import_error(self):
        from xlmtec.integrations.claude import ClaudeIntegration

        with patch("xlmtec.integrations.claude.anthropic", None):
            with pytest.raises(ImportError, match="anthropic"):
                ClaudeIntegration(api_key="sk-test").suggest("test")

    def test_api_failure_raises_runtime_error(self):
        from xlmtec.integrations.claude import ClaudeIntegration

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value.messages.create.side_effect = Exception(
            "rate limited"
        )

        with patch("xlmtec.integrations.claude.anthropic", mock_anthropic):
            with pytest.raises(RuntimeError, match="Claude API call failed"):
                ClaudeIntegration(api_key="sk-test").suggest("test")


# ---------------------------------------------------------------------------
# GeminiIntegration
# ---------------------------------------------------------------------------


class TestGeminiIntegration:
    def test_suggest_returns_result(self):
        from xlmtec.integrations.gemini import GeminiIntegration

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value.generate_content.return_value = MagicMock(
            text=SAMPLE_RAW
        )

        with patch("xlmtec.integrations.gemini.genai", mock_genai):
            result = GeminiIntegration(api_key="test-key").suggest("fine-tune for classification")

        assert isinstance(result, SuggestResult)
        assert result.method == "lora"

    def test_missing_sdk_raises_import_error(self):
        from xlmtec.integrations.gemini import GeminiIntegration

        with patch("xlmtec.integrations.gemini.genai", None):
            with pytest.raises(ImportError, match="google-genai"):
                GeminiIntegration(api_key="test").suggest("test")

    def test_api_failure_raises_runtime_error(self):
        from xlmtec.integrations.gemini import GeminiIntegration

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = Exception(
            "quota exceeded"
        )

        with patch("xlmtec.integrations.gemini.genai", mock_genai):
            with pytest.raises(RuntimeError, match="Gemini API call failed"):
                GeminiIntegration(api_key="test-key").suggest("test")


# ---------------------------------------------------------------------------
# CodexIntegration
# ---------------------------------------------------------------------------


class TestCodexIntegration:
    def test_suggest_returns_result(self):
        from xlmtec.integrations.codex import CodexIntegration

        mock_openai = MagicMock()
        mock_openai.return_value.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=SAMPLE_RAW))]
        )

        with patch("xlmtec.integrations.codex.OpenAI", mock_openai):
            result = CodexIntegration(api_key="sk-test").suggest("fine-tune for code")

        assert isinstance(result, SuggestResult)
        assert result.method == "lora"

    def test_missing_sdk_raises_import_error(self):
        from xlmtec.integrations.codex import CodexIntegration

        with patch("xlmtec.integrations.codex.OpenAI", None):
            with pytest.raises(ImportError, match="openai"):
                CodexIntegration(api_key="sk-test").suggest("test")

    def test_api_failure_raises_runtime_error(self):
        from xlmtec.integrations.codex import CodexIntegration

        mock_openai = MagicMock()
        mock_openai.return_value.chat.completions.create.side_effect = Exception("invalid key")

        with patch("xlmtec.integrations.codex.OpenAI", mock_openai):
            with pytest.raises(RuntimeError, match="OpenAI API call failed"):
                CodexIntegration(api_key="sk-test").suggest("test")


# ---------------------------------------------------------------------------
# SuggestResult dataclass
# ---------------------------------------------------------------------------


class TestSuggestResult:
    def test_fields(self, sample_result):
        assert sample_result.method == "lora"
        assert "gpt2" in sample_result.yaml_config
        assert sample_result.command.startswith("xlmtec train")

    def test_raw_not_in_repr(self, sample_result):
        assert "raw" not in repr(sample_result)
