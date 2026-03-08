"""
xlmtec.integrations
~~~~~~~~~~~~~~~~~~~~
AI provider integrations for the ai-suggest command.

Usage:
    from xlmtec.integrations import get_provider
    provider = get_provider("claude", api_key="sk-...")
    result = provider.suggest("fine-tune GPT-2 for sentiment analysis")
"""

from xlmtec.integrations.base import AIIntegration, SuggestResult
from xlmtec.integrations.claude import ClaudeIntegration
from xlmtec.integrations.codex import CodexIntegration
from xlmtec.integrations.gemini import GeminiIntegration

PROVIDERS = {
    "claude": ClaudeIntegration,
    "gemini": GeminiIntegration,
    "codex": CodexIntegration,
}


def get_provider(name: str, api_key: str | None = None) -> "AIIntegration":
    """Instantiate a provider by name.

    Args:
        name:    One of 'claude', 'gemini', 'codex'.
        api_key: API key. If None, falls back to the provider's env variable.

    Returns:
        An AIIntegration instance ready to call .suggest().
    """
    name = name.lower().strip()
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {name!r}. Choose from: {', '.join(PROVIDERS)}"
        )
    return PROVIDERS[name](api_key=api_key)


__all__ = [
    "AIIntegration",
    "SuggestResult",
    "ClaudeIntegration",
    "GeminiIntegration",
    "CodexIntegration",
    "get_provider",
    "PROVIDERS",
]