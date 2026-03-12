"""
xlmtec.plugins.store
~~~~~~~~~~~~~~~~~~~~~
Persistent plugin registry stored at ~/.xlmtec/plugins.json

Schema:
{
    "templates": {
        "my_task": {
            "name": "my_task",
            "source": "/path/to/my_task.yaml",
            "registered_at": "2026-03-09T12:00:00"
        }
    },
    "providers": {
        "my_provider": {
            "name": "my_provider",
            "source": "/path/to/my_provider.py",
            "class_name": "MyProvider",
            "registered_at": "2026-03-09T12:00:00"
        }
    }
}
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PLUGIN_DIR = Path.home() / ".xlmtec"
PLUGIN_FILE = PLUGIN_DIR / "plugins.json"


@dataclass
class TemplatePlugin:
    name: str
    source: str  # absolute path to YAML file
    registered_at: str


@dataclass
class ProviderPlugin:
    name: str
    source: str  # absolute path to .py file
    class_name: str
    registered_at: str


@dataclass
class PluginStore:
    templates: dict[str, TemplatePlugin]
    providers: dict[str, ProviderPlugin]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_store(plugin_file: Path = PLUGIN_FILE) -> PluginStore:
    """Load the plugin store from disk. Returns empty store if file missing."""
    if not plugin_file.exists():
        return PluginStore(templates={}, providers={})
    try:
        raw = json.loads(plugin_file.read_text(encoding="utf-8"))
        templates = {k: TemplatePlugin(**v) for k, v in raw.get("templates", {}).items()}
        providers = {k: ProviderPlugin(**v) for k, v in raw.get("providers", {}).items()}
        return PluginStore(templates=templates, providers=providers)
    except (json.JSONDecodeError, TypeError, KeyError):
        return PluginStore(templates={}, providers={})


def save_store(store: PluginStore, plugin_file: Path = PLUGIN_FILE) -> None:
    """Persist the plugin store to disk."""
    plugin_file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "templates": {k: asdict(v) for k, v in store.templates.items()},
        "providers": {k: asdict(v) for k, v in store.providers.items()},
    }
    plugin_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def register_template(
    name: str,
    source: Path,
    plugin_file: Path = PLUGIN_FILE,
) -> TemplatePlugin:
    """Add or update a template plugin entry.

    Raises:
        FileNotFoundError: If source YAML does not exist.
        ValueError: If name conflicts with a built-in template.
    """
    from xlmtec.templates.registry import _TEMPLATES

    if name in _TEMPLATES:
        raise ValueError(
            f"Cannot register plugin: {name!r} is already a built-in template.\n"
            "Choose a different name."
        )
    if not source.exists():
        raise FileNotFoundError(f"Template file not found: {source}")

    store = load_store(plugin_file)
    plugin = TemplatePlugin(
        name=name,
        source=str(source.resolve()),
        registered_at=_now(),
    )
    store.templates[name] = plugin
    save_store(store, plugin_file)
    return plugin


def register_provider(
    name: str,
    source: Path,
    class_name: str,
    plugin_file: Path = PLUGIN_FILE,
) -> ProviderPlugin:
    """Add or update a provider plugin entry.

    Raises:
        FileNotFoundError: If source .py file does not exist.
        ValueError: If name conflicts with a built-in provider.
    """
    BUILTIN_PROVIDERS = {"claude", "gemini", "codex"}
    if name.lower() in BUILTIN_PROVIDERS:
        raise ValueError(
            f"Cannot register plugin: {name!r} is a built-in provider.\nChoose a different name."
        )
    if not source.exists():
        raise FileNotFoundError(f"Provider file not found: {source}")

    store = load_store(plugin_file)
    plugin = ProviderPlugin(
        name=name,
        source=str(source.resolve()),
        class_name=class_name,
        registered_at=_now(),
    )
    store.providers[name] = plugin
    save_store(store, plugin_file)
    return plugin


def remove_plugin(
    name: str,
    plugin_file: Path = PLUGIN_FILE,
) -> bool:
    """Remove a plugin by name (template or provider).

    Returns:
        True if removed, False if not found.
    """
    store = load_store(plugin_file)
    removed = False
    if name in store.templates:
        del store.templates[name]
        removed = True
    if name in store.providers:
        del store.providers[name]
        removed = True
    if removed:
        save_store(store, plugin_file)
    return removed
