# finetune-cli

**Production-grade LLM fine-tuning, distillation, and pruning from the command line.**

[![CI](https://github.com/Abdur-azure/finetune_cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdur-azure/finetune_cli/actions)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What it does

`finetune-cli` is a modular Python framework for fine-tuning, distilling, and pruning large language models. It wraps HuggingFace Transformers + PEFT in a clean CLI, a validated config system, a composable trainer stack, an interactive TUI, and a full test suite — all CPU-runnable for unit tests.

---

## Install

```bash
git clone https://github.com/Abdur-azure/finetune_cli.git
cd finetune_cli
pip install -e .
```

---

## 5-minute quickstart

```bash
# 1. Generate sample training data (no network required)
python examples/generate_sample_data.py

# 2. Not sure which method to use? Ask
finetune-cli recommend gpt2 --output my_config.yaml

# 3. Train with the generated config
finetune-cli train --config my_config.yaml

# 4. Or use a ready-made config
finetune-cli train --config examples/configs/lora_gpt2.yaml

# 5. Launch the interactive TUI
finetune-cli tui
```

---

## CLI commands

| Command | What it does |
|---------|-------------|
| `finetune-cli train` | Fine-tune using a YAML config or inline flags (LoRA / QLoRA / Full / Instruction / DPO / Distillation) |
| `finetune-cli evaluate` | Score a saved checkpoint (ROUGE, BLEU, Perplexity) |
| `finetune-cli benchmark` | Before/after comparison: base vs fine-tuned |
| `finetune-cli merge` | Merge LoRA adapter into base model → standalone model |
| `finetune-cli upload` | Push adapter or merged model to HuggingFace Hub |
| `finetune-cli recommend` | Inspect model size + VRAM, output optimal YAML config |
| `finetune-cli prune` | Structured pruning — zero lowest-magnitude attention heads |
| `finetune-cli wanda` | WANDA unstructured pruning — zero weights by \|W\|×activation score |
| `finetune-cli tui` | Interactive Textual TUI — all commands via a terminal UI |

---

## Training methods

| Method | Flag | Notes |
|--------|------|-------|
| LoRA | `--method lora` | Default. Adapter-based, memory-efficient |
| QLoRA | `--method qlora` | 4-bit quantised LoRA — large models on limited VRAM |
| Full Fine-Tuning | `--method full_finetuning` | All parameters — small models only |
| Instruction Tuning | `--method instruction_tuning` | Alpaca-style `{instruction, input, response}` data |
| DPO | `--method dpo` | Direct Preference Optimization — requires `pip install trl` |
| Response Distillation | `--method vanilla_distillation` | Student mimics teacher logits (KL + CE loss) |
| Feature Distillation | `--method feature_distillation` | Student mimics teacher hidden states (MSE + KL + CE) |

---

## Pruning commands

```bash
# Structured pruning — zero lowest-magnitude attention heads
finetune-cli prune ./outputs/gpt2_lora \
    --output ./outputs/gpt2_pruned \
    --sparsity 0.3 \
    --method heads

# WANDA unstructured pruning — weight × activation scoring, zero-shot
finetune-cli wanda ./outputs/gpt2_lora \
    --output ./outputs/gpt2_wanda \
    --sparsity 0.5 \
    --dataset ./data/sample.jsonl
```

---

## Example configs

| Config | Method | Model | Data |
|--------|--------|-------|------|
| `lora_gpt2.yaml` | LoRA | GPT-2 | `data/sample.jsonl` |
| `qlora_llama.yaml` | QLoRA | LLaMA-3.2-1B | HF Hub (needs token) |
| `instruction_tuning.yaml` | Instruction | GPT-2 | `data/instructions.jsonl` |
| `full_finetuning.yaml` | Full | GPT-2 | `data/sample.jsonl` |
| `dpo.yaml` | DPO | GPT-2 | `data/dpo_sample.jsonl` |
| `response_distillation.yaml` | Response Distillation | GPT-2 (student) ← GPT-2-medium | `data/sample.jsonl` |
| `feature_distillation.yaml` | Feature Distillation | GPT-2 (student) ← GPT-2-medium | `data/sample.jsonl` |
| `structured_pruning.yaml` | Structured Pruning | GPT-2 | — |
| `wanda.yaml` | WANDA Pruning | GPT-2 | `data/sample.jsonl` (calibration) |

---

## Python API

```python
from finetune_cli.core.config import ConfigBuilder
from finetune_cli.core.types import TrainingMethod, DatasetSource
from finetune_cli.models.loader import load_model_and_tokenizer
from finetune_cli.data import prepare_dataset
from finetune_cli.trainers import TrainerFactory

config = (
    ConfigBuilder()
    .with_model("gpt2")
    .with_dataset("./data/sample.jsonl", source=DatasetSource.LOCAL_FILE)
    .with_tokenization(max_length=256)
    .with_training(TrainingMethod.LORA, "./output", num_epochs=3)
    .with_lora(r=8, lora_alpha=16)
    .build()
)

model, tokenizer = load_model_and_tokenizer(config.model.to_config())
dataset = prepare_dataset(config.dataset.to_config(), config.tokenization.to_config(), tokenizer)
result = TrainerFactory.train(
    model, tokenizer, dataset,
    config.training.to_config(),
    config.lora.to_config(),
)
print(f"Done. Loss: {result.train_loss:.4f}  →  {result.output_dir}")
```

---

## Docs

- [Usage Guide](docs/usage.md) — all 9 commands with examples
- [Configuration Reference](docs/configuration.md) — YAML config fields for all methods
- [API Reference](docs/api.md) — Python API for all trainers and pruners
- [TUI Guide](docs/tui.md) — interactive terminal interface
- [Architecture](docs/ARCHITECTURE.md) — module design
- [Contributing](CONTRIBUTING.md) — how to add trainers or commands

---

## Tests

```bash
# Unit tests (no GPU needed)
pytest tests/ -v --ignore=tests/test_integration.py

# Integration tests (CPU ok, ~30s — downloads GPT-2 once)
pytest tests/test_integration.py -v -s

# Full suite
pytest tests/ -v
```

---

## Project status

| Aspect | Status |
|--------|--------|
| Version | 3.13.0 |
| Tests | 200+ unit + integration, all green |
| CI | pytest on Python 3.10 / 3.11 / 3.12 |
| Platform | Windows / macOS / Linux |
| License | MIT |