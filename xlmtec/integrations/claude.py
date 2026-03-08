"""
xlmtec.integrations.claude
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Anthropic Claude integration for xlmtec ai-suggest.

Install:  pip install xlmtec[claude]
Env var:  ANTHROPIC_API_KEY
"""

from __future__ import annotations

from xlmtec.integrations.base import AIIntegration, SuggestResult
from xlmtec.integrations.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from xlmtec.integrations.response_parser import parse_response

# Module-level import so tests can patch `xlmtec.integrations.claude.anthropic`
try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]


class ClaudeIntegration(AIIntegration):
    """AI suggestion provider using Anthropic Claude."""

    PROVIDER_NAME = "claude"
    ENV_KEY = "ANTHROPIC_API_KEY"
    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def suggest(self, prompt: str) -> SuggestResult:
        if anthropic is None:
            raise ImportError(
                "The 'anthropic' package is required for the Claude provider.\n"
                "Install it with:  pip install xlmtec[claude]"
            )

        api_key = self._require_api_key()
        client = anthropic.Anthropic(api_key=api_key)

        try:
            message = client.messages.create(
                model=self.DEFAULT_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_prompt(prompt)}],
            )
            raw = message.content[0].text
        except Exception as exc:
            raise RuntimeError(f"Claude API call failed: {exc}") from exc

        return parse_response(raw, provider=self.PROVIDER_NAME)