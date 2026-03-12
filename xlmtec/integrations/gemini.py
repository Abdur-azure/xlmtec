"""
xlmtec.integrations.gemini
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Google Gemini integration for xlmtec ai-suggest.

Install:  pip install xlmtec[gemini]
Env var:  GEMINI_API_KEY
"""

from __future__ import annotations

from xlmtec.integrations.base import AIIntegration, SuggestResult
from xlmtec.integrations.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from xlmtec.integrations.response_parser import parse_response

# google.generativeai is deprecated — use google.genai instead
try:
    import google.genai as genai

    _GENAI_NEW = True
except ImportError:
    try:
        import google.generativeai as genai  # type: ignore[no-redef]

        _GENAI_NEW = False
    except ImportError:
        genai = None  # type: ignore[assignment]
        _GENAI_NEW = False


class GeminiIntegration(AIIntegration):
    """AI suggestion provider using Google Gemini."""

    PROVIDER_NAME = "gemini"
    ENV_KEY = "GEMINI_API_KEY"
    DEFAULT_MODEL = "gemini-1.5-flash"

    def suggest(self, prompt: str) -> SuggestResult:
        if genai is None:
            raise ImportError(
                "The 'google-genai' package is required for the Gemini provider.\n"
                "Install it with:  pip install xlmtec[gemini]"
            )

        api_key = self._require_api_key()

        try:
            if _GENAI_NEW:
                # New google.genai API
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=self.DEFAULT_MODEL,
                    contents=build_user_prompt(prompt),
                    config={"system_instruction": SYSTEM_PROMPT},
                )
                raw = response.text
            else:
                # Legacy google.generativeai fallback
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    model_name=self.DEFAULT_MODEL,
                    system_instruction=SYSTEM_PROMPT,
                )
                raw = model.generate_content(build_user_prompt(prompt)).text
        except Exception as exc:
            raise RuntimeError(f"Gemini API call failed: {exc}") from exc

        return parse_response(raw, provider=self.PROVIDER_NAME)
