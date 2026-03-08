"""
xlmtec.integrations.base
~~~~~~~~~~~~~~~~~~~~~~~~~
Shared protocol and result types for all AI provider integrations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SuggestResult:
    """Structured output from an ai-suggest call.

    Attributes:
        method:       Recommended fine-tuning method (e.g. 'lora', 'qlora').
        yaml_config:  Ready-to-run YAML string for xlmtec train.
        explanation:  Human-readable summary of why this config was chosen.
        command:      Ready-to-run CLI command string.
        raw:          Raw text response from the provider (for debugging).
    """

    method: str
    yaml_config: str
    explanation: str
    command: str
    raw: str = field(default="", repr=False)


class AIIntegration(ABC):
    """Abstract base for all AI provider integrations.

    Subclasses must implement :meth:`suggest` and set :attr:`PROVIDER_NAME`
    and :attr:`ENV_KEY`.
    """

    PROVIDER_NAME: str = ""
    ENV_KEY: str = ""          # Environment variable name for the API key
    DEFAULT_MODEL: str = ""    # Model used by this provider

    def __init__(self, api_key: str | None = None) -> None:
        import os
        self.api_key = api_key or os.environ.get(self.ENV_KEY, "")

    @abstractmethod
    def suggest(self, prompt: str) -> SuggestResult:
        """Generate a fine-tuning suggestion from a plain-English prompt.

        Args:
            prompt: Natural language description of the fine-tuning task.

        Returns:
            A :class:`SuggestResult` with YAML config + explanation.

        Raises:
            ImportError:  If the provider SDK is not installed.
            RuntimeError: If the API call fails.
        """

    def _require_api_key(self) -> str:
        """Return the API key or raise a helpful error."""
        if not self.api_key:
            raise RuntimeError(
                f"No API key found for {self.PROVIDER_NAME}. "
                f"Set the {self.ENV_KEY} environment variable or pass api_key=."
            )
        return self.api_key