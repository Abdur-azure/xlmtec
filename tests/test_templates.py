"""
tests/test_templates.py
~~~~~~~~~~~~~~~~~~~~~~~~
Tests for built-in config templates.
No ML dependencies — pure data and YAML validation.
"""

from __future__ import annotations

import pytest
import yaml

from xlmtec.templates.registry import Template, get_template, list_templates

# ---------------------------------------------------------------------------
# list_templates
# ---------------------------------------------------------------------------


class TestListTemplates:
    def test_returns_all_templates(self):
        templates = list_templates()
        assert len(templates) == 7

    def test_returns_template_objects(self):
        assert all(isinstance(t, Template) for t in list_templates())

    def test_sorted_by_name(self):
        names = [t.name for t in list_templates()]
        assert names == sorted(names)

    def test_all_have_required_fields(self):
        for t in list_templates():
            assert t.name
            assert t.description
            assert t.method
            assert t.base_model
            assert isinstance(t.config, dict)


# ---------------------------------------------------------------------------
# get_template
# ---------------------------------------------------------------------------


class TestGetTemplate:
    def test_returns_correct_template(self):
        t = get_template("sentiment")
        assert t.name == "sentiment"
        assert t.method == "lora"

    def test_case_insensitive(self):
        assert get_template("SENTIMENT").name == "sentiment"
        assert get_template("Summarisation").name == "summarisation"

    def test_all_templates_retrievable(self):
        names = ["sentiment", "classification", "qa", "summarisation", "code", "chat", "dpo"]
        for name in names:
            t = get_template(name)
            assert t.name == name

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="not found"):
            get_template("nonexistent")

    def test_error_lists_available(self):
        with pytest.raises(ValueError, match="sentiment"):
            get_template("bad")


# ---------------------------------------------------------------------------
# Template.as_dict
# ---------------------------------------------------------------------------


class TestAsDict:
    def test_returns_deep_copy(self):
        t = get_template("sentiment")
        d1 = t.as_dict()
        d2 = t.as_dict()
        d1["model"]["name"] = "modified"
        assert d2["model"]["name"] != "modified"

    def test_no_overrides(self):
        t = get_template("sentiment")
        d = t.as_dict()
        assert d["method"] == "lora"
        assert d["model"]["name"] == "distilbert-base-uncased"

    def test_model_override(self):
        t = get_template("sentiment")
        d = t.as_dict(overrides={"model": {"name": "bert-base-uncased"}})
        assert d["model"]["name"] == "bert-base-uncased"

    def test_nested_training_override(self):
        t = get_template("sentiment")
        d = t.as_dict(overrides={"training": {"num_epochs": 10}})
        assert d["training"]["num_epochs"] == 10
        # Other training fields preserved
        assert "learning_rate" in d["training"]

    def test_top_level_override(self):
        t = get_template("sentiment")
        d = t.as_dict(overrides={"method": "full"})
        assert d["method"] == "full"


# ---------------------------------------------------------------------------
# Template.to_yaml
# ---------------------------------------------------------------------------


class TestToYaml:
    def test_produces_valid_yaml(self):
        for t in list_templates():
            yaml_str = t.to_yaml()
            parsed = yaml.safe_load(yaml_str)
            assert isinstance(parsed, dict)

    def test_yaml_contains_model_name(self):
        t = get_template("sentiment")
        assert "distilbert-base-uncased" in t.to_yaml()

    def test_yaml_with_override(self):
        t = get_template("sentiment")
        yaml_str = t.to_yaml(overrides={"model": {"name": "gpt2"}})
        parsed = yaml.safe_load(yaml_str)
        assert parsed["model"]["name"] == "gpt2"


# ---------------------------------------------------------------------------
# Template configs — spot checks per template
# ---------------------------------------------------------------------------


class TestTemplateConfigs:
    def test_sentiment_uses_lora(self):
        assert get_template("sentiment").method == "lora"

    def test_code_uses_qlora(self):
        assert get_template("code").method == "qlora"

    def test_chat_uses_instruction(self):
        assert get_template("chat").method == "instruction"

    def test_dpo_uses_dpo(self):
        assert get_template("dpo").method == "dpo"

    def test_all_configs_have_training_section(self):
        for t in list_templates():
            assert "training" in t.config, f"{t.name} missing training section"

    def test_all_configs_have_model_section(self):
        for t in list_templates():
            assert "model" in t.config, f"{t.name} missing model section"

    def test_all_configs_have_output_dir(self):
        for t in list_templates():
            assert "output_dir" in t.config["training"], f"{t.name} missing output_dir"


# ---------------------------------------------------------------------------
# template use (direct logic — no CLI runner)
# ---------------------------------------------------------------------------


class TestTemplateUse:
    def test_saves_yaml_file(self, tmp_path):
        t = get_template("sentiment")
        out = tmp_path / "config.yaml"
        out.write_text(t.to_yaml(), encoding="utf-8")
        assert out.exists()
        parsed = yaml.safe_load(out.read_text())
        assert parsed["method"] == "lora"

    def test_override_applied_in_saved_file(self, tmp_path):
        t = get_template("classification")
        out = tmp_path / "config.yaml"
        out.write_text(t.to_yaml(overrides={"model": {"name": "gpt2"}}), encoding="utf-8")
        parsed = yaml.safe_load(out.read_text())
        assert parsed["model"]["name"] == "gpt2"
