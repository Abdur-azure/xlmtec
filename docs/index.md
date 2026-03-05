# finetune-cli

**Production-grade LLM fine-tuning, distillation, and pruning from the command line.**

`finetune-cli` is a modular Python framework that wraps HuggingFace Transformers + PEFT in a clean CLI, a validated config system, a composable trainer stack, an interactive TUI, and a full test suite — all CPU-runnable for unit tests.

---

## Quick start

```bash
# Install
git clone https://github.com/Abdur-azure/finetune_cli.git
cd finetune_cli
pip install -e .

# Generate sample data (no network required)
python examples/generate_sample_data.py

# Fine-tune GPT-2 with LoRA
finetune-cli train --config examples/configs/lora_gpt2.yaml

# Or use the interactive TUI
finetune-cli tui
```

---

## What's included

### CLI commands

| Command | Description |
|---------|-------------|
| `finetune-cli train` | Fine-tune with LoRA / QLoRA / Full / Instruction / DPO / Distillation |
| `finetune-cli evaluate` | Score a checkpoint — ROUGE, BLEU, Perplexity |
| `finetune-cli benchmark` | Before/after comparison report |
| `finetune-cli merge` | Merge LoRA adapter into base model |
| `finetune-cli upload` | Push model or adapter to HuggingFace Hub |
| `finetune-cli recommend` | Inspect model + VRAM, output optimal YAML config |
| `finetune-cli prune` | Structured pruning — zero lowest-magnitude attention heads |
| `finetune-cli wanda` | WANDA unstructured pruning — weight × activation scoring |
| `finetune-cli tui` | Interactive Textual TUI — all commands via terminal UI |

### Training methods

| Method | Class | Sprint |
|--------|-------|--------|
| LoRA | `LoRATrainer` | 2 |
| QLoRA | `QLoRATrainer` | 2 |
| Full Fine-Tuning | `FullFineTuner` | 2 |
| Instruction Tuning | `InstructionTrainer` | 2 |
| DPO | `DPOTrainer` | 8 |
| Response Distillation | `ResponseDistillationTrainer` | 23 |
| Feature Distillation | `FeatureDistillationTrainer` | 24 |

### Pruning

| Method | Class | Algorithm |
|--------|-------|-----------|
| Structured Pruning | `StructuredPruner` | Zero lowest-magnitude attention head rows per layer |
| WANDA | `WandaPruner` | Zero weights by \|W_ij\| × \|\|X_j\|\|₂ score |

### Core components

| Component | Description |
|-----------|-------------|
| `ConfigBuilder` | Fluent Python API for building validated `PipelineConfig` |
| `DataPipeline` | Loads JSON/JSONL/CSV/Parquet/HF datasets, tokenizes, splits |
| `TrainerFactory` | Single entry point — selects trainer from `TrainingMethod` enum |
| `BenchmarkRunner` | Model-agnostic evaluation with comparison reports |

---

## Navigation

- **[Installation](installation.md)** — requirements, GPU setup, HuggingFace login
- **[Usage Guide](usage.md)** — all 9 CLI commands with examples
- **[Configuration](configuration.md)** — YAML config reference for all methods
- **[API Reference](api.md)** — Python API for trainers, pruners, and data pipeline
- **[TUI Guide](tui.md)** — interactive terminal interface walkthrough
- **[Troubleshooting](troubleshooting.md)** — OOM, NaN loss, dataset errors

---

## Project status

| Aspect | Status |
|--------|--------|
| **Version** | 3.13.0 |
| **Tests** | 200+ unit + integration (all green, no GPU required for unit suite) |
| **CI** | pytest matrix — Python 3.10 / 3.11 / 3.12 |
| **Platform** | Windows / macOS / Linux |
| **License** | MIT |