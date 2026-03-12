"""
tests/test_plugins.py
~~~~~~~~~~~~~~~~~~~~~~
Tests for the plugin system — store, loader, and CLI logic.
All filesystem operations use tmp_path — no ~/.xlmtec written.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from xlmtec.plugins.store import (
    PluginStore,
    ProviderPlugin,
    TemplatePlugin,
    load_store,
    register_provider,
    register_template,
    remove_plugin,
    save_store,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plugin_file(tmp_path: Path) -> Path:
    return tmp_path / "plugins.json"


def _make_yaml_template(tmp_path: Path, name: str = "my_task") -> Path:
    f = tmp_path / f"{name}.yaml"
    f.write_text(
        yaml.dump(
            {
                "method": "lora",
                "description": f"Custom {name} template",
                "task": "text-classification",
                "base_model": "gpt2",
                "model": {"name": "gpt2"},
                "training": {"output_dir": f"output/{name}", "num_epochs": 3},
            }
        ),
        encoding="utf-8",
    )
    return f


def _make_provider_py(tmp_path: Path, class_name: str = "MyProvider") -> Path:
    f = tmp_path / "my_provider.py"
    f.write_text(
        f"""
class {class_name}:
    name = "my_provider"
    def suggest(self, task): return None
""",
        encoding="utf-8",
    )
    return f


# ---------------------------------------------------------------------------
# load_store / save_store
# ---------------------------------------------------------------------------


class TestStoreIO:
    def test_load_missing_returns_empty(self, tmp_path):
        store = load_store(_plugin_file(tmp_path))
        assert store.templates == {}
        assert store.providers == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        pf = _plugin_file(tmp_path)
        store = PluginStore(
            templates={"t1": TemplatePlugin("t1", "/a/b.yaml", "2026-01-01T00:00:00")},
            providers={"p1": ProviderPlugin("p1", "/a/p.py", "MyClass", "2026-01-01T00:00:00")},
        )
        save_store(store, pf)
        loaded = load_store(pf)
        assert "t1" in loaded.templates
        assert "p1" in loaded.providers
        assert loaded.templates["t1"].source == "/a/b.yaml"
        assert loaded.providers["p1"].class_name == "MyClass"

    def test_corrupt_file_returns_empty(self, tmp_path):
        pf = _plugin_file(tmp_path)
        pf.write_text("not json", encoding="utf-8")
        store = load_store(pf)
        assert store.templates == {}

    def test_creates_parent_dir(self, tmp_path):
        pf = tmp_path / "deep" / "dir" / "plugins.json"
        save_store(PluginStore({}, {}), pf)
        assert pf.exists()


# ---------------------------------------------------------------------------
# register_template
# ---------------------------------------------------------------------------


class TestRegisterTemplate:
    def test_registers_new_template(self, tmp_path):
        pf = _plugin_file(tmp_path)
        src = _make_yaml_template(tmp_path, "my_task")
        plugin = register_template("my_task", src, pf)
        assert plugin.name == "my_task"
        assert Path(plugin.source).exists()

    def test_persists_to_file(self, tmp_path):
        pf = _plugin_file(tmp_path)
        src = _make_yaml_template(tmp_path)
        register_template("my_task", src, pf)
        store = load_store(pf)
        assert "my_task" in store.templates

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            register_template("x", tmp_path / "missing.yaml", _plugin_file(tmp_path))

    def test_builtin_name_raises(self, tmp_path):
        src = _make_yaml_template(tmp_path, "sentiment")
        with pytest.raises(ValueError, match="built-in"):
            register_template("sentiment", src, _plugin_file(tmp_path))

    def test_overwrite_existing_plugin(self, tmp_path):
        pf = _plugin_file(tmp_path)
        src = _make_yaml_template(tmp_path)
        register_template("my_task", src, pf)
        src2 = tmp_path / "other.yaml"
        src2.write_text(yaml.dump({"method": "full"}))
        register_template("my_task", src2, pf)
        store = load_store(pf)
        assert store.templates["my_task"].source.endswith("other.yaml")

    def test_registered_at_is_set(self, tmp_path):
        pf = _plugin_file(tmp_path)
        src = _make_yaml_template(tmp_path)
        plugin = register_template("my_task", src, pf)
        assert "2026" in plugin.registered_at or "202" in plugin.registered_at


# ---------------------------------------------------------------------------
# register_provider
# ---------------------------------------------------------------------------


class TestRegisterProvider:
    def test_registers_new_provider(self, tmp_path):
        pf = _plugin_file(tmp_path)
        src = _make_provider_py(tmp_path)
        plugin = register_provider("my_provider", src, "MyProvider", pf)
        assert plugin.name == "my_provider"
        assert plugin.class_name == "MyProvider"

    def test_persists_to_file(self, tmp_path):
        pf = _plugin_file(tmp_path)
        src = _make_provider_py(tmp_path)
        register_provider("my_provider", src, "MyProvider", pf)
        store = load_store(pf)
        assert "my_provider" in store.providers

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            register_provider("x", tmp_path / "missing.py", "X", _plugin_file(tmp_path))

    def test_builtin_name_raises(self, tmp_path):
        src = _make_provider_py(tmp_path)
        with pytest.raises(ValueError, match="built-in"):
            register_provider("claude", src, "MyProvider", _plugin_file(tmp_path))

    def test_builtin_name_case_insensitive(self, tmp_path):
        src = _make_provider_py(tmp_path)
        with pytest.raises(ValueError, match="built-in"):
            register_provider("GEMINI", src, "MyProvider", _plugin_file(tmp_path))


# ---------------------------------------------------------------------------
# remove_plugin
# ---------------------------------------------------------------------------


class TestRemovePlugin:
    def test_removes_template(self, tmp_path):
        pf = _plugin_file(tmp_path)
        src = _make_yaml_template(tmp_path)
        register_template("my_task", src, pf)
        assert remove_plugin("my_task", pf) is True
        assert "my_task" not in load_store(pf).templates

    def test_removes_provider(self, tmp_path):
        pf = _plugin_file(tmp_path)
        src = _make_provider_py(tmp_path)
        register_provider("my_provider", src, "MyProvider", pf)
        assert remove_plugin("my_provider", pf) is True
        assert "my_provider" not in load_store(pf).providers

    def test_missing_returns_false(self, tmp_path):
        pf = _plugin_file(tmp_path)
        assert remove_plugin("nonexistent", pf) is False


# ---------------------------------------------------------------------------
# PluginLoader — template loading
# ---------------------------------------------------------------------------


class TestPluginLoader:
    def test_loads_template_into_registry(self, tmp_path):
        from xlmtec.plugins.loader import PluginLoader
        from xlmtec.templates.registry import _TEMPLATES

        pf = _plugin_file(tmp_path)
        src = _make_yaml_template(tmp_path, "custom_nlp")
        register_template("custom_nlp", src, pf)

        result = PluginLoader(pf).load()
        assert "custom_nlp" in result.templates_loaded
        assert "custom_nlp" in _TEMPLATES

        # Cleanup registry so other tests aren't affected
        _TEMPLATES.pop("custom_nlp", None)

    def test_missing_yaml_file_recorded_as_error(self, tmp_path):
        from xlmtec.plugins.loader import PluginLoader
        from xlmtec.plugins.store import PluginStore, TemplatePlugin

        pf = _plugin_file(tmp_path)
        store = PluginStore(
            templates={
                "ghost": TemplatePlugin("ghost", "/nonexistent/path.yaml", "2026-01-01T00:00:00")
            },
            providers={},
        )
        save_store(store, pf)

        result = PluginLoader(pf).load()
        assert len(result.errors) == 1
        assert "ghost" in result.errors[0]

    def test_empty_store_loads_cleanly(self, tmp_path):
        from xlmtec.plugins.loader import PluginLoader

        result = PluginLoader(_plugin_file(tmp_path)).load()
        assert result.templates_loaded == []
        assert result.providers_loaded == []
        assert result.errors == []
        assert result.ok is True
