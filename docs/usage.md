# Usage Guide

Complete guide to all `finetune-cli` commands with real examples.

---

## 5-minute quickstart

```bash
# 1. Install
git clone https://github.com/Abdur-azure/finetune_cli.git
cd finetune_cli
pip install -e .

# 2. Generate sample data (no network required)
python examples/generate_sample_data.py
# Creates:
#   data/sample.jsonl          (500 rows, causal LM)
#   data/instructions.jsonl    (300 rows, alpaca format)
#   data/dpo_sample.jsonl      (200 rows, prompt/chosen/rejected)

# 3. Not sure which method to use? Ask
finetune-cli recommend gpt2 --output my_config.yaml
finetune-cli train --config my_config.yaml
```

---

## Command reference

| Command | What it does |
|---------|-------------|
| `finetune-cli train` | Fine-tune using a YAML/JSON config or inline flags |
| `finetune-cli evaluate` | Score a saved checkpoint (ROUGE, BLEU, Perplexity) |
| `finetune-cli benchmark` | Before/after comparison: base vs fine-tuned |
| `finetune-cli merge` | Merge LoRA adapter into base model — standalone model |
| `finetune-cli upload` | Push adapter or merged model to HuggingFace Hub |
| `finetune-cli recommend` | Inspect model size + VRAM, output optimal YAML config |
| `finetune-cli prune` | Structured attention-head pruning — no retraining required |
| `finetune-cli wanda` | WANDA unstructured pruning — weight × activation scoring |
| `finetune-cli tui` | Interactive Textual TUI — all commands via terminal UI |

---

## `train` — Fine-tune a model

### Using a config file (recommended)

```bash
finetune-cli train --config examples/configs/lora_gpt2.yaml
```

### Using flags (quick experiments)

```bash
finetune-cli train \
  --model gpt2 \
  --dataset ./data/sample.jsonl \
  --method lora \
  --epochs 3 \
  --output ./output
```

### Training methods

#### LoRA — general purpose adapter fine-tuning
```bash
finetune-cli train --model gpt2 --dataset ./data/sample.jsonl --method lora --epochs 3
# or: finetune-cli train --config examples/configs/lora_gpt2.yaml
```

#### QLoRA — 4-bit quantised, large models on limited VRAM
```bash
finetune-cli train \
  --model meta-llama/Llama-3.2-1B \
  --dataset ./data/sample.jsonl \
  --method qlora --4bit --fp16 --epochs 2
# or: finetune-cli train --config examples/configs/qlora_llama.yaml
```

#### Instruction tuning — alpaca-style `{instruction, input, response}` data
```bash
finetune-cli train \
  --model gpt2 \
  --dataset ./data/instructions.jsonl \
  --method instruction_tuning --epochs 3
# or: finetune-cli train --config examples/configs/instruction_tuning.yaml
```

#### Full fine-tuning — all parameters, small models only
```bash
finetune-cli train \
  --model gpt2 \
  --dataset ./data/sample.jsonl \
  --method full_finetuning --lr 1e-5 --epochs 3
# or: finetune-cli train --config examples/configs/full_finetuning.yaml
```

> **Warning:** Full fine-tuning trains every parameter. Only safe on models ≤300M params unless you have 24GB+ VRAM.

#### DPO — Direct Preference Optimization
Requires `pip install trl>=0.7.0`. Dataset must have `prompt`, `chosen`, `rejected` columns.

```bash
finetune-cli train \
  --model gpt2 \
  --dataset ./data/dpo_sample.jsonl \
  --method dpo --epochs 1
# or: finetune-cli train --config examples/configs/dpo.yaml
```

#### Response Distillation — student mimics teacher logits
Student (smaller model) is trained to match the output distribution of a larger teacher via KL divergence + cross-entropy loss. No labelled data required beyond the training corpus.

```bash
finetune-cli train --config examples/configs/response_distillation.yaml
```

Config key fields:
```yaml
training:
  method: vanilla_distillation
distillation:
  teacher_model_name: gpt2-medium
  temperature: 2.0      # higher = softer teacher distribution
  alpha: 0.5            # blend: 0.5 KL + 0.5 CE
```

#### Feature Distillation — student mimics teacher hidden states
Extends response distillation with an MSE loss on intermediate hidden states, giving stronger layer-level supervision.

```bash
finetune-cli train --config examples/configs/feature_distillation.yaml
```

Config key fields:
```yaml
training:
  method: feature_distillation
feature_distillation:
  teacher_model_name: gpt2-medium
  temperature: 2.0
  alpha: 0.5            # KL weight
  beta: 0.3             # MSE hidden-state weight
  feature_layers: null  # null = auto-select 4 evenly-spaced layers
```

---

## `evaluate` — Score a checkpoint

```bash
finetune-cli evaluate ./outputs/gpt2_lora \
  --dataset ./data/sample.jsonl \
  --metrics rougeL,bleu \
  --num-samples 200
```

Available metrics: `rouge1`, `rouge2`, `rougeL`, `bleu`, `perplexity`.

---

## `benchmark` — Before/after comparison

```bash
finetune-cli benchmark gpt2 ./outputs/gpt2_lora \
  --dataset ./data/sample.jsonl \
  --metrics rougeL,bleu
```

Outputs a side-by-side table: base model scores vs fine-tuned scores with delta indicators.

---

## `merge` — Merge adapter into base model

Produces a standalone model with no PEFT dependency — ready for direct inference.

```bash
finetune-cli merge ./outputs/gpt2_lora ./outputs/gpt2_merged \
  --base-model gpt2 \
  --dtype float16
```

---

## `upload` — Push to HuggingFace Hub

```bash
# Adapter only
finetune-cli upload ./outputs/gpt2_lora \
  --repo username/gpt2-lora-finetuned \
  --token $HF_TOKEN

# Merge adapter before upload
finetune-cli upload ./outputs/gpt2_lora \
  --repo username/gpt2-merged \
  --token $HF_TOKEN \
  --merge-adapter \
  --base-model gpt2

# Private repository
finetune-cli upload ./outputs/gpt2_lora \
  --repo username/gpt2-private \
  --private
```

`HF_TOKEN` can also be set as an environment variable — the `--token` flag is then optional.

---

## `recommend` — Get an optimal config

Inspects the model's parameter count and your available VRAM, then writes a ready-to-use YAML config.

```bash
finetune-cli recommend gpt2 --output my_config.yaml
```

---

## `prune` — Structured attention-head pruning

Soft structured pruning: scores each attention head by mean absolute weight magnitude, then zeros the lowest-scoring fraction. No retraining required. The model shape is unchanged — it runs with any standard HuggingFace inference stack.

```bash
# Prune 30% of attention heads (default)
finetune-cli prune ./outputs/gpt2_lora \
  --output ./outputs/gpt2_pruned \
  --sparsity 0.3

# Prune FFN neurons instead of attention heads
finetune-cli prune ./outputs/gpt2_lora \
  --output ./outputs/gpt2_pruned \
  --sparsity 0.3 \
  --method ffn

# Keep at least 2 heads per layer (prevent collapse)
finetune-cli prune ./outputs/gpt2_lora \
  --output ./outputs/gpt2_pruned \
  --sparsity 0.5 \
  --min-heads 2
```

| Option | Default | Description |
|--------|---------|-------------|
| `--sparsity` | `0.3` | Fraction of heads to zero (0.0–1.0) |
| `--method` | `heads` | `heads` or `ffn` |
| `--min-heads` | `1` | Minimum heads kept per layer |

Sparsity guidance: `0.1` = light, `0.3` = moderate (recommended), `0.5` = aggressive.

After pruning, evaluate with:
```bash
finetune-cli benchmark gpt2 ./outputs/gpt2_pruned --dataset ./data/sample.jsonl
```

---

## `wanda` — WANDA unstructured pruning

WANDA (Weight AND Activation) scores each weight by `|W_ij| × ‖X_j‖₂` where `X` is the input activation norm collected on a small calibration dataset. Weights with the lowest scores are zeroed. No gradient pass or retraining is required.

```bash
# With calibration dataset (recommended — uses activation norms)
finetune-cli wanda ./outputs/gpt2_lora \
  --output ./outputs/gpt2_wanda \
  --sparsity 0.5 \
  --dataset ./data/sample.jsonl \
  --n-samples 128

# Without calibration data (falls back to magnitude-only scoring)
finetune-cli wanda ./outputs/gpt2_lora \
  --output ./outputs/gpt2_wanda \
  --sparsity 0.5
```

| Option | Default | Description |
|--------|---------|-------------|
| `--sparsity` | `0.5` | Fraction of weights to zero per layer (0.0–1.0) |
| `--dataset` | none | Calibration `.jsonl`/`.txt` file |
| `--n-samples` | `128` | Number of calibration forward passes |
| `--seq-len` | `128` | Token sequence length for calibration |
| `--row-wise/--global` | row-wise | Per-row threshold vs global threshold |

Sparsity guidance: `0.3` = light, `0.5` = standard (matches original WANDA paper on LLaMA/OPT), `0.6+` = aggressive.

**WANDA vs Structured Pruning:**

| | WANDA | Structured Pruning |
|--|-------|-------------------|
| Granularity | Unstructured (individual weights) | Structured (whole heads/neurons) |
| Calibration | Recommended (uses activations) | Not needed |
| Accuracy | Higher (more surgical) | Simpler, no activation tracking |
| Score | \|W\| × ‖activation‖ | Mean \|W\| per head |

---

## `tui` — Interactive terminal UI

Launches a full Textual TUI with 8 screens covering all commands. No flags needed.

```bash
finetune-cli tui
```

Navigation: arrow keys or mouse, `Tab` between fields, `Enter` to submit, `Esc`/`h` to go back, `q` to quit.

See [TUI Guide](tui.md) for full documentation with screenshots.