"""
xlmtec.templates
~~~~~~~~~~~~~~~~~
Built-in starter configs for common fine-tuning tasks.

Usage:
    from xlmtec.templates import get_template, list_templates
    t = get_template("sentiment")
    print(t.to_yaml())
"""

from xlmtec.templates.registry import Template, get_template, list_templates

__all__ = ["Template", "get_template", "list_templates"]
