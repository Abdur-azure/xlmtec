# xlmtec/integrations — Context

AI provider integrations powering the `xlmtec ai-suggest` command.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | `get_provider(name, api_key)` factory + re-exports |
| `base.py` | `AIIntegration` abstract base + `SuggestResult` dataclass |
| `claude.py` | Anthropic Claude provider |
| `gemini.py` | Google Gemini provider |
| `codex.py` | OpenAI GPT provider |
| `prompt_builder.py` | Shared system prompt + user prompt template |
| `response_parser.py` | Parse raw JSON → `SuggestResult` (handles fences, prose wrappers) |
| `CONTEXT.md` | This file |

## Adding a new provider

1. Create `xlmtec/integrations/yourprovider.py` subclassing `AIIntegration`
2. Set `PROVIDER_NAME`, `ENV_KEY`, `DEFAULT_MODEL`
3. Implement `suggest(prompt) -> SuggestResult` using `parse_response()`
4. Register in `__init__.py` `PROVIDERS` dict
5. Add `yourprovider = ["sdk>=version"]` extra to `pyproject.toml`
6. Add tests to `tests/test_integrations.py`

## Rules

- **All SDK imports are lazy** (inside `suggest()`) — never at module level.
  This keeps the package importable without any AI SDK installed.
- **`_require_api_key()`** must be called before any API call.
- **`parse_response()`** handles markdown fences and embedded JSON —
  providers don't always return clean JSON even when asked.
- **Tests must mock at the SDK level** — never make real API calls in tests.
  Patch `xlmtec.integrations.claude.anthropic`, not `anthropic`.

## Install extras

```bash
pip install xlmtec[claude]   # Anthropic only
pip install xlmtec[gemini]   # Google only
pip install xlmtec[codex]    # OpenAI only
pip install xlmtec[ai]       # All three
```

## Environment variables

| Provider | Variable |
|----------|----------|
| claude | `ANTHROPIC_API_KEY` |
| gemini | `GEMINI_API_KEY` |
| codex | `OPENAI_API_KEY` |