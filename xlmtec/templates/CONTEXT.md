# xlmtec/templates — Context

Built-in config template registry powering `xlmtec template` commands.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `registry.py` | `_TEMPLATES` dict, `Template` dataclass, `get_template()`, `list_templates()` |
| `CONTEXT.md` | This file |

## Commands

```bash
xlmtec template list
xlmtec template show sentiment
xlmtec template use sentiment --output config.yaml
xlmtec template use summarisation --model facebook/bart-base --output config.yaml
```

## Built-in Templates

| Name | Method | Base Model |
|------|--------|-----------|
| `sentiment` | lora | distilbert-base-uncased |
| `classification` | lora | bert-base-uncased |
| `qa` | lora | deepset/roberta-base-squad2 |
| `summarisation` | lora | facebook/bart-base |
| `code` | qlora | Salesforce/codegen-350M-mono |
| `chat` | instruction | microsoft/DialoGPT-small |
| `dpo` | dpo | gpt2 |

## Rules

- `_TEMPLATES` is a plain `dict[str, Template]` — the plugin loader extends it at startup.
- Built-in names are reserved — `register_template()` raises `ValueError` if a plugin tries to claim one.
- `Template.config` is the raw dict used to render YAML — override fields with `Template.as_dict(**overrides)`.
- No network calls in this module — all templates are hardcoded.

## Extension pattern

To add a new built-in template:
1. Add an entry to `_TEMPLATES` in `registry.py`
2. Add a spot-check test in `tests/test_templates.py`
3. Add the template name to the table above

To add a **user** template: `xlmtec plugin add-template <name> <path.yaml>`