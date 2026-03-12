"""
xlmtec.templates.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~
Built-in starter configs for common fine-tuning tasks.

Each template is a complete, ready-to-run PipelineConfig dict.
Users can use them as-is or override specific fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Template:
    """A named starter config for a common fine-tuning task."""

    name: str
    description: str
    task: str
    method: str
    base_model: str
    config: dict[str, Any]
    tags: list[str] = field(default_factory=list)

    def as_dict(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return config dict with optional field overrides applied.

        Overrides are applied at the top level and nested levels.
        E.g. overrides={"model": {"name": "gpt2"}} replaces just the model name.
        """
        import copy

        result = copy.deepcopy(self.config)
        if overrides:
            for key, value in overrides.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key].update(value)
                else:
                    result[key] = value
        return result

    def to_yaml(self, overrides: dict[str, Any] | None = None) -> str:
        """Return config as a YAML string."""
        import yaml

        return yaml.dump(self.as_dict(overrides), default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, Template] = {
    "sentiment": Template(
        name="sentiment",
        description="Sentiment analysis (positive/negative/neutral) on text.",
        task="text-classification",
        method="lora",
        base_model="distilbert-base-uncased",
        tags=["classification", "nlp", "beginner"],
        config={
            "method": "lora",
            "model": {
                "name": "distilbert-base-uncased",
                "torch_dtype": "float16",
            },
            "dataset": {
                "source": "local_file",
                "path": "data/train.jsonl",
                "text_column": "text",
                "label_column": "label",
            },
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.1,
                "target_modules": ["q_lin", "v_lin"],
            },
            "training": {
                "output_dir": "output/sentiment",
                "num_epochs": 3,
                "batch_size": 16,
                "learning_rate": 2e-4,
                "warmup_steps": 100,
                "save_steps": 500,
                "eval_steps": 500,
            },
        },
    ),
    "classification": Template(
        name="classification",
        description="Multi-class text classification for any number of categories.",
        task="text-classification",
        method="lora",
        base_model="bert-base-uncased",
        tags=["classification", "nlp"],
        config={
            "method": "lora",
            "model": {
                "name": "bert-base-uncased",
                "torch_dtype": "float16",
            },
            "dataset": {
                "source": "local_file",
                "path": "data/train.jsonl",
                "text_column": "text",
                "label_column": "label",
            },
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["query", "value"],
            },
            "training": {
                "output_dir": "output/classification",
                "num_epochs": 5,
                "batch_size": 16,
                "learning_rate": 2e-4,
                "warmup_steps": 200,
                "save_steps": 500,
                "eval_steps": 500,
            },
        },
    ),
    "qa": Template(
        name="qa",
        description="Question answering — extract answers from a context passage.",
        task="question-answering",
        method="lora",
        base_model="deepset/roberta-base-squad2",
        tags=["qa", "extraction", "nlp"],
        config={
            "method": "lora",
            "model": {
                "name": "deepset/roberta-base-squad2",
                "torch_dtype": "float16",
            },
            "dataset": {
                "source": "local_file",
                "path": "data/train.jsonl",
                "text_column": "context",
            },
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["query", "value"],
            },
            "training": {
                "output_dir": "output/qa",
                "num_epochs": 3,
                "batch_size": 8,
                "learning_rate": 1e-4,
                "warmup_steps": 300,
                "save_steps": 500,
                "eval_steps": 500,
            },
        },
    ),
    "summarisation": Template(
        name="summarisation",
        description="Abstractive text summarisation — condense long documents.",
        task="summarization",
        method="lora",
        base_model="facebook/bart-base",
        tags=["summarisation", "generation", "nlp"],
        config={
            "method": "lora",
            "model": {
                "name": "facebook/bart-base",
                "torch_dtype": "float16",
            },
            "dataset": {
                "source": "local_file",
                "path": "data/train.jsonl",
                "text_column": "document",
                "label_column": "summary",
            },
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
            },
            "training": {
                "output_dir": "output/summarisation",
                "num_epochs": 5,
                "batch_size": 4,
                "learning_rate": 5e-5,
                "warmup_steps": 500,
                "save_steps": 1000,
                "eval_steps": 1000,
            },
        },
    ),
    "code": Template(
        name="code",
        description="Code generation and completion on a custom codebase.",
        task="text-generation",
        method="qlora",
        base_model="Salesforce/codegen-350M-mono",
        tags=["code", "generation", "qlora"],
        config={
            "method": "qlora",
            "model": {
                "name": "Salesforce/codegen-350M-mono",
                "torch_dtype": "float16",
                "load_in_4bit": True,
            },
            "dataset": {
                "source": "local_file",
                "path": "data/train.jsonl",
                "text_column": "code",
            },
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["qkv_proj"],
            },
            "training": {
                "output_dir": "output/code",
                "num_epochs": 3,
                "batch_size": 4,
                "learning_rate": 2e-4,
                "warmup_steps": 100,
                "save_steps": 500,
                "eval_steps": 500,
            },
        },
    ),
    "chat": Template(
        name="chat",
        description="Instruction-tuned conversational assistant.",
        task="text-generation",
        method="instruction",
        base_model="microsoft/DialoGPT-small",
        tags=["chat", "instruction", "conversational"],
        config={
            "method": "instruction",
            "model": {
                "name": "microsoft/DialoGPT-small",
                "torch_dtype": "float16",
            },
            "dataset": {
                "source": "local_file",
                "path": "data/train.jsonl",
                "text_column": "instruction",
                "label_column": "response",
            },
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["c_attn"],
            },
            "training": {
                "output_dir": "output/chat",
                "num_epochs": 3,
                "batch_size": 4,
                "learning_rate": 2e-4,
                "warmup_steps": 100,
                "save_steps": 500,
                "eval_steps": 500,
            },
        },
    ),
    "dpo": Template(
        name="dpo",
        description="Direct Preference Optimisation — train from human preference pairs.",
        task="text-generation",
        method="dpo",
        base_model="gpt2",
        tags=["dpo", "rlhf", "preference"],
        config={
            "method": "dpo",
            "model": {
                "name": "gpt2",
                "torch_dtype": "float16",
            },
            "dataset": {
                "source": "local_file",
                "path": "data/preferences.jsonl",
                "prompt_column": "prompt",
                "chosen_column": "chosen",
                "rejected_column": "rejected",
            },
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["c_attn"],
            },
            "training": {
                "output_dir": "output/dpo",
                "num_epochs": 1,
                "batch_size": 2,
                "learning_rate": 5e-7,
                "warmup_steps": 50,
                "save_steps": 200,
                "eval_steps": 200,
            },
        },
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_templates() -> list[Template]:
    """Return all available templates sorted by name."""
    return sorted(_TEMPLATES.values(), key=lambda t: t.name)


def get_template(name: str) -> Template:
    """Return a template by name.

    Raises:
        ValueError: If the template is not found.
    """
    name = name.lower().strip()
    if name not in _TEMPLATES:
        available = ", ".join(sorted(_TEMPLATES))
        raise ValueError(f"Template {name!r} not found. Available: {available}")
    return _TEMPLATES[name]
