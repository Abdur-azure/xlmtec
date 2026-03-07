# Configuration Reference

All YAML config fields for every training method, distillation mode, and pruning command.

---

## Common sections

Every config file that uses `lmtool train` shares these top-level sections.

### `model`

```yaml
model:
  name: "gpt2"            # HuggingFace model id or local path
  device: "auto"          # "auto" | "cpu" | "cuda"
  torch_dtype: "float32"  # "float32" | "float16" | "bfloat16"
  load_in_4bit: false     # true for QLoRA
  load_in_8bit: false
```

### `dataset`

```yaml
dataset:
  source: local_file      # local_file | huggingface_hub | csv | parquet
  path: ./data/sample.jsonl
  max_samples: 1000       # null = use all
  text_columns: null      # null = auto-detect
  shuffle: true
```

### `tokenization`

```yaml
tokenization:
  max_length: 256
  truncation: true
  padding: max_length     # "max_length" | "longest" | false
```

### `training`

```yaml
training:
  method: lora            # see methods below
  output_dir: ./outputs/run
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  warmup_ratio: 0.1
  lr_scheduler_type: cosine
  fp16: false
  gradient_checkpointing: false
  save_strategy: epoch    # "epoch" | "steps" | "no"
  logging_steps: 10
```

### `lora`

Used by: `lora`, `qlora`, `instruction_tuning`, `dpo`.

```yaml
lora:
  r: 8                    # rank — higher = more expressive, more memory
  lora_alpha: 16          # scale factor (rule of thumb: 2× r)
  lora_dropout: 0.05
  target_modules: null    # null = auto-detect; or ["c_attn", "c_proj"]
  bias: none
```

**Target module guidance:**

| Model family | Recommended `target_modules` |
|-------------|------------------------------|
| GPT-2 | `["c_attn", "c_proj"]` |
| LLaMA / Mistral | `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| BERT | `["query", "value"]` |
| Auto | `null` |

---

## Method-specific config

### LoRA

```yaml
training:
  method: lora
lora:
  r: 8
  lora_alpha: 16
  target_modules: null
```

Full example: `examples/configs/lora_gpt2.yaml`

### QLoRA

Requires `model.load_in_4bit: true`.

```yaml
model:
  name: meta-llama/Llama-3.2-1B
  load_in_4bit: true
  torch_dtype: float16
training:
  method: qlora
  fp16: true
  gradient_checkpointing: true
lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

Full example: `examples/configs/qlora_llama.yaml`

### Full fine-tuning

No `lora` section required. Use only on small models (≤300M params) or with ≥24GB VRAM.

```yaml
training:
  method: full_finetuning
  learning_rate: 1.0e-5
  num_epochs: 3
```

Full example: `examples/configs/full_finetuning.yaml`

### Instruction tuning

Dataset must have `instruction`, `input` (optional), and `response` columns (alpaca format).

```yaml
dataset:
  path: ./data/instructions.jsonl
training:
  method: instruction_tuning
lora:
  r: 8
  lora_alpha: 16
```

Full example: `examples/configs/instruction_tuning.yaml`

### DPO

Requires `pip install trl>=0.7.0`. Dataset must have `prompt`, `chosen`, `rejected` columns.

```yaml
dataset:
  path: ./data/dpo_sample.jsonl
training:
  method: dpo
  learning_rate: 5.0e-5
lora:
  r: 8
  lora_alpha: 16
```

`beta` (preference shaping strength, default `0.1`) is set via the Python API:
```python
DPOTrainer(model, tokenizer, training_config, lora_config, beta=0.2)
```

Full example: `examples/configs/dpo.yaml`

---

## Distillation config

### Response Distillation

Student model is trained to match the teacher's output distribution. Uses KL divergence blended with cross-entropy loss. No labelled data required beyond the training corpus.

```yaml
training:
  method: vanilla_distillation
  output_dir: ./outputs/response_distillation
  num_epochs: 3
  batch_size: 4
  learning_rate: 1.0e-4

distillation:
  teacher_model_name: gpt2-medium   # any HuggingFace model id
  temperature: 2.0                  # higher = softer probability distribution
  alpha: 0.5                        # loss = alpha×KL + (1-alpha)×CE
```

**Tuning guide:**

| Parameter | Effect |
|-----------|--------|
| `temperature` | Higher (2–4) softens teacher logits, shares more relative probability across tokens |
| `alpha` | 1.0 = pure KL distillation; 0.0 = pure CE (standard training); 0.5 = balanced |

Full example: `examples/configs/response_distillation.yaml`

### Feature Distillation

Extends response distillation with an additional MSE loss on intermediate hidden states. Provides stronger layer-level supervision, especially useful when the teacher and student share similar architecture.

```yaml
training:
  method: feature_distillation
  output_dir: ./outputs/feature_distillation
  num_epochs: 3
  batch_size: 4
  learning_rate: 1.0e-4

feature_distillation:
  teacher_model_name: gpt2-medium
  temperature: 2.0
  alpha: 0.5          # KL divergence weight
  beta: 0.3           # hidden-state MSE weight
  feature_layers: null  # null = auto-select 4 evenly-spaced student layers
```

**Loss breakdown:**
```
total_loss = alpha × KL(student || teacher)
           + (1 - alpha - beta) × CE(student, labels)
           + beta × MSE(student_hidden, teacher_hidden)
```

`feature_layers` accepts a list of student layer indices, e.g. `[0, 4, 8, 11]`. Each is mapped to the proportionally corresponding teacher layer. `null` auto-selects 4 evenly-spaced layers.

**Tuning guide:**

| Parameter | Effect |
|-----------|--------|
| `beta` | Higher = more hidden-state alignment; start at 0.1–0.3 |
| `feature_layers` | Fewer layers = faster training; more = stronger supervision |

Full example: `examples/configs/feature_distillation.yaml`

---

## Pruning config

Pruning commands (`prune`, `wanda`) use their own config objects, not `PipelineConfig`. They are standalone operations on a saved model — no `training` or `lora` sections.

### Structured Pruning (`lmtool prune`)

```yaml
# examples/configs/structured_pruning.yaml

model_path: "./outputs/gpt2_lora"

pruning:
  output_dir: "./outputs/gpt2_pruned"
  sparsity: 0.3             # fraction of heads to zero (0.0–1.0)
  method: "heads"           # "heads" (attention) or "ffn" (feed-forward)
  importance_metric: "magnitude"
  min_heads_per_layer: 1    # safety floor — never collapse a layer entirely
```

**Sparsity guidance:**

| `sparsity` | Effect |
|-----------|--------|
| 0.1 | Light — minimal accuracy impact, small size reduction |
| 0.3 | Moderate — good balance (recommended starting point) |
| 0.5 | Aggressive — noticeable accuracy drop, significant speedup |
| 0.7+ | Extreme — only for heavily over-parameterised models |

Python API:
```python
from lmtool.core.types import PruningConfig
from lmtool.trainers import StructuredPruner

config = PruningConfig(
    output_dir=Path("./outputs/pruned"),
    sparsity=0.3,
    method="heads",
    min_heads_per_layer=1,
)
result = StructuredPruner(model, tokenizer, config).prune()
```

Full example: `examples/configs/structured_pruning.yaml`

### WANDA Pruning (`lmtool wanda`)

```yaml
# examples/configs/wanda.yaml

model_path: "./outputs/gpt2_lora"

wanda:
  output_dir: "./outputs/gpt2_wanda"
  sparsity: 0.5                    # fraction of weights to zero (0.0–1.0)
  n_calibration_samples: 128       # number of calibration forward passes
  calibration_seq_len: 128         # token sequence length for calibration
  use_row_wise: true               # true = per-row threshold (recommended)
  layer_types: null                # null = auto (Linear + Conv1D)
```

**Sparsity guidance:**

| `sparsity` | Effect |
|-----------|--------|
| 0.3 | Light — minimal accuracy drop |
| 0.5 | Standard — matches original paper results on LLaMA/OPT |
| 0.6 | Aggressive — calibration data recommended |
| 0.7+ | Research-grade — significant accuracy loss expected |

**`use_row_wise`:** When `true`, the sparsity threshold is computed per output neuron row — each neuron retains the same fraction of its incoming weights. When `false`, a single global threshold is applied across the entire weight matrix.

Python API:
```python
from lmtool.core.types import WandaConfig
from lmtool.trainers import WandaPruner

config = WandaConfig(
    output_dir=Path("./outputs/wanda"),
    sparsity=0.5,
    n_calibration_samples=128,
    use_row_wise=True,
)
result = WandaPruner(model, tokenizer, config).prune(calibration_input_ids=calib_ids)
```

Full example: `examples/configs/wanda.yaml`

---

## Configuration recipes

### Quick CPU experiment

```yaml
model:
  name: gpt2
  torch_dtype: float32
dataset:
  max_samples: 500
tokenization:
  max_length: 128
training:
  num_epochs: 1
  batch_size: 2
  gradient_accumulation_steps: 2
lora:
  r: 4
  lora_alpha: 8
```

### Balanced quality — 8GB GPU

```yaml
model:
  name: gpt2-medium
  torch_dtype: float16
training:
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  fp16: true
lora:
  r: 8
  lora_alpha: 32
```

### Large model — limited VRAM (QLoRA)

```yaml
model:
  name: meta-llama/Llama-3.2-1B
  load_in_4bit: true
  torch_dtype: float16
training:
  method: qlora
  batch_size: 2
  gradient_accumulation_steps: 8
  fp16: true
  gradient_checkpointing: true
lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### Distillation — student ← teacher

```yaml
model:
  name: gpt2             # student
training:
  method: vanilla_distillation
  num_epochs: 3
distillation:
  teacher_model_name: gpt2-medium
  temperature: 2.0
  alpha: 0.5
```