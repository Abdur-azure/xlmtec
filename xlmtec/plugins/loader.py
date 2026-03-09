"""
xlmtec.plugins.loader
~~~~~~~~~~~~~~~~~~~~~~
PluginLoader — reads the plugin store and injects custom templates/providers
into the live registries at startup.

Designed to be called once at CLI startup:
    from xlmtec.plugins.loader import PluginLoader
    PluginLoader().load()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LoadResult:
    """Summary of what was loaded."""
    templates_loaded: list[str] = field(default_factory=list)
    providers_loaded: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


class PluginLoader:
    """Load user plugins into the xlmtec registries."""

    def __init__(self, plugin_file: Path | None = None) -> None:
        from xlmtec.plugins.store import PLUGIN_FILE
        self.plugin_file = plugin_file or PLUGIN_FILE

    def load(self) -> LoadResult:
        """Load all registered plugins. Errors are collected, not raised."""
        from xlmtec.plugins.store import load_store
        store = load_store(self.plugin_file)
        result = LoadResult()

        for name, plugin in store.templates.items():
            err = self._load_template(name, plugin)
            if err:
                result.errors.append(err)
            else:
                result.templates_loaded.append(name)

        for name, plugin in store.providers.items():
            err = self._load_provider(name, plugin)
            if err:
                result.errors.append(err)
            else:
                result.providers_loaded.append(name)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_template(self, name: str, plugin) -> str | None:
        """Load a template YAML into the template registry. Returns error string or None."""
        try:
            import yaml
            from xlmtec.templates.registry import Template, _TEMPLATES

            source = Path(plugin.source)
            if not source.exists():
                return f"Template {name!r}: file not found at {source}"

            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return f"Template {name!r}: YAML must be a mapping, got {type(raw).__name__}"

            # Required fields with defaults
            template = Template(
                name=name,
                description=raw.get("description", f"Custom template: {name}"),
                task=raw.get("task", "text-generation"),
                method=raw.get("method", "lora"),
                base_model=raw.get(
                    "base_model",
                    raw.get("model", {}).get("name", "gpt2") if isinstance(raw.get("model"), dict) else "gpt2"
                ),
                config=raw,
                tags=raw.get("tags", ["custom"]),
            )
            _TEMPLATES[name] = template
            return None

        except Exception as exc:
            return f"Template {name!r}: {exc}"

    def _load_provider(self, name: str, plugin) -> str | None:
        """Dynamically import a provider class and register it. Returns error or None."""
        try:
            import importlib.util
            from xlmtec.integrations import _PROVIDERS

            source = Path(plugin.source)
            if not source.exists():
                return f"Provider {name!r}: file not found at {source}"

            spec = importlib.util.spec_from_file_location(f"xlmtec_plugin_{name}", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            cls = getattr(module, plugin.class_name, None)
            if cls is None:
                return f"Provider {name!r}: class {plugin.class_name!r} not found in {source}"

            _PROVIDERS[name] = cls
            return None

        except Exception as exc:
            return f"Provider {name!r}: {exc}"