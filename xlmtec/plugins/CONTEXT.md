# xlmtec/plugins — Context

User plugin registry — extend templates and AI providers without modifying the package.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Docstring only |
| `store.py` | `PluginStore`, `register_template()`, `register_provider()`, `remove_plugin()` — persists to `~/.xlmtec/plugins.json` |
| `loader.py` | `PluginLoader` — reads store and injects plugins into live registries at startup |
| `CONTEXT.md` | This file |

## Commands

```bash
xlmtec plugin add-template my_task templates/my_task.yaml
xlmtec plugin add-provider my_llm providers/my_llm.py --class MyLLMIntegration
xlmtec plugin list
xlmtec plugin remove my_task
```

## Plugin Types

| Type | Injected into | Visible via |
|------|--------------|-------------|
| Template | `xlmtec.templates.registry._TEMPLATES` | `xlmtec template list` |
| Provider | `xlmtec.integrations._PROVIDERS` | `xlmtec ai-suggest --provider` |

## Store Schema (`~/.xlmtec/plugins.json`)

```json
{
  "templates": {
    "my_task": { "name": "my_task", "source": "/abs/path/my_task.yaml", "registered_at": "..." }
  },
  "providers": {
    "my_llm": { "name": "my_llm", "source": "/abs/path/my_llm.py", "class_name": "MyLLM", "registered_at": "..." }
  }
}
```

## Rules

- Built-in template names (`sentiment`, `classification`, `qa`, `summarisation`, `code`, `chat`, `dpo`) are reserved — registration raises `ValueError`.
- Built-in provider names (`claude`, `gemini`, `codex`) are reserved — case-insensitive check.
- `PluginLoader.load()` collects errors without raising — a broken plugin never crashes the CLI.
- `PluginLoader().load()` must be called once in `cli/main.py` `@app.callback()` so plugins are active for every command.
- Tests always pass an explicit `plugin_file=tmp_path/plugins.json` — never touch `~/.xlmtec`.

## Startup wiring (main.py)

```python
@app.callback()
def main(...):
    PluginLoader().load()   # ← first line
```