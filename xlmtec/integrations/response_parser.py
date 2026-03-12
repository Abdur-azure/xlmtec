"""
xlmtec.integrations.response_parser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Parses raw text responses from AI providers into typed SuggestResult objects.
"""

from __future__ import annotations

import json
import re

from xlmtec.integrations.base import SuggestResult


def parse_response(raw: str, provider: str = "") -> SuggestResult:
    """Parse a raw provider response into a SuggestResult.

    Handles:
    - Clean JSON responses
    - JSON wrapped in markdown code fences
    - Partial / malformed JSON (best-effort fallback)

    Args:
        raw:      Raw text from the provider.
        provider: Provider name for error messages.

    Returns:
        A populated :class:`SuggestResult`.

    Raises:
        ValueError: If JSON cannot be parsed at all.
    """
    text = raw.strip()

    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting the first JSON object via regex
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(
                f"Could not parse JSON from {provider or 'provider'} response.\n"
                f"Raw response:\n{raw[:500]}"
            )
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON from {provider or 'provider'}: {exc}\nRaw response:\n{raw[:500]}"
            ) from exc

    method = data.get("method", "lora").strip()
    yaml_config = data.get("yaml_config", "").strip()
    explanation = data.get("explanation", "").strip()

    command = _build_command(method, yaml_config)

    return SuggestResult(
        method=method,
        yaml_config=yaml_config,
        explanation=explanation,
        command=command,
        raw=raw,
    )


def _build_command(method: str, yaml_config: str) -> str:
    """Derive a ready-to-run CLI command from the parsed result."""
    # Try to extract output_dir from yaml for a nicer command
    match = re.search(r"output_dir:\s*(\S+)", yaml_config)
    output_hint = f" --output-dir {match.group(1)}" if match else ""
    return f"xlmtec train --method {method} --config config.yaml{output_hint}"
