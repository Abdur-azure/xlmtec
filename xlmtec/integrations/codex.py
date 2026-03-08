"""
xlmtec.integrations.codex
~~~~~~~~~~~~~~~~~~~~~~~~~~
OpenAI GPT integration for xlmtec ai-suggest.

Install:  pip install xlmtec[codex]
Env var:  OPENAI_API_KEY
"""

from __future__ import annotations

from xlmtec.integrations.base import AIIntegration, SuggestResult
from xlmtec.integrations.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from xlmtec.integrations.response_parser import parse_response

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]


class CodexIntegration(AIIntegration):
    """AI suggestion provider using OpenAI GPT."""

    PROVIDER_NAME = "codex"
    ENV_KEY = "OPENAI_API_KEY"
    DEFAULT_MODEL = "gpt-4o-mini"

    def suggest(self, prompt: str) -> SuggestResult:
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for the Codex provider.\n"
                "Install it with:  pip install xlmtec[codex]"
            )

        api_key = self._require_api_key()

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(prompt)},
                ],
                max_tokens=1024,
                temperature=0.3,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

        return parse_response(raw, provider=self.PROVIDER_NAME)