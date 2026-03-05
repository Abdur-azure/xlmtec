# API Reference

Python API for programmatic use. All public classes are importable from their respective subpackages.

---

## Configuration — `finetune_cli.core.config`

### `ConfigBuilder`

Fluent builder for constructing a validated `PipelineConfig`.

```python
from finetune_cli.core.config import ConfigBuilder
from finetune_cli.core.types import TrainingMethod, DatasetSource

config = (
    ConfigBuilder()
    .with_model("gpt2", torch_dtype="float32")
    .with_dataset("./data.jsonl", source=DatasetSource.LOCAL_FILE, max_samples=1000)
    .with_tokenization(max_length=512)
    .with_training(TrainingMethod.LORA, "./output", num_epochs=3, batch_size=4)
    .with_lora(r=8, lora_alpha=32, lora_dropout=0.1)
    .build()
)
```

| Method | Key kwargs | Description |
|--------|-----------|-------------|
| `.with_model(name, **kwargs)` | `torch_dtype`, `load_in_4bit`, `load_in_8bit` | Set model config |
| `.with_dataset(path, source, **kwargs)` | `max_samples`, `text_columns`, `shuffle` | Set dataset config |
| `.with_tokenization(**kwargs)` | `max_length`, `truncation`, `padding` | Set tokenization config |
| `.with_training(method, output_dir, **kwargs)` | `num_epochs`, `batch_size`, `learning_rate`, `fp16` | Set training config |
| `.with_lora(**kwargs)` | `r`, `lora_alpha`, `lora_dropout`, `target_modules` | Set LoRA config |
| `.with_evaluation(metrics, **kwargs)` | `batch_size`, `num_samples` | Set evaluation config |
| `.build()` | — | Validate and return `PipelineConfig` |

### `PipelineConfig`

Pydantic model holding the full pipeline config. Supports JSON and YAML I/O.

```python
config = PipelineConfig.from_yaml(Path("config.yaml"))
config = PipelineConfig.from_json(Path("config.json"))
config.to_yaml(Path("config.yaml"))
```

---

## Data Pipeline — `finetune_cli.data`

### `quick_load`

One-liner for loading and tokenizing a dataset.

```python
from finetune_cli.data import quick_load
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = quick_load("./data.jsonl", tokenizer, max_samples=500, max_length=512)
# Returns: datasets.Dataset with input_ids, attention_mask, labels
```

### `prepare_dataset`

Full pipeline with optional train/validation split.

```python
from finetune_cli.data import prepare_dataset

result = prepare_dataset(
    dataset_config=config.dataset.to_config(),
    tokenization_config=config.tokenization.to_config(),
    tokenizer=tokenizer,
    split_for_validation=True,
    validation_ratio=0.1,
)
# result["train"], result["validation"]
```

---

## Model Loading — `finetune_cli.models.loader`

```python
from finetune_cli.models.loader import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer(config.model.to_config())
```

Handles device mapping, 4-bit/8-bit quantization, and `pad_token` setup automatically.

---

## Trainers — `finetune_cli.trainers`

### `TrainerFactory.train` (recommended)

Single entry point — selects the right trainer based on `TrainingMethod`.

```python
from finetune_cli.trainers import TrainerFactory

result = TrainerFactory.train(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    training_config=config.training.to_config(),
    lora_config=config.lora.to_config(),          # required for lora / qlora / instruction
    distillation_config=distillation_config,       # required for vanilla_distillation
    feature_distillation_config=fd_config,         # required for feature_distillation
)
```

### `LoRATrainer` / `QLoRATrainer` / `FullFineTuner` / `InstructionTrainer`

```python
from finetune_cli.trainers import LoRATrainer

trainer = LoRATrainer(model, tokenizer, training_config, lora_config)
result = trainer.train(dataset)
```

### `DPOTrainer`

Requires `pip install trl>=0.7.0`. Dataset must have `prompt`, `chosen`, `rejected` columns.

```python
from finetune_cli.trainers import DPOTrainer, validate_dpo_dataset

validate_dpo_dataset(dataset)   # raises ValueError if columns are missing
trainer = DPOTrainer(model, tokenizer, training_config, lora_config, beta=0.1)
result = trainer.train(dataset)
```

`beta` controls preference shaping strength: lower (0.05–0.1) stays close to the reference model; higher (0.3–0.5) applies stronger shaping.

### `ResponseDistillationTrainer`

Student learns to match the output distribution of a larger teacher model (KL divergence + cross-entropy loss).

```python
from finetune_cli.core.types import DistillationConfig
from finetune_cli.trainers import ResponseDistillationTrainer

distillation_config = DistillationConfig(
    teacher_model_name="gpt2-medium",
    temperature=2.0,     # higher = softer teacher distribution
    alpha=0.5,           # blend: alpha×KL + (1-alpha)×CE
)
trainer = ResponseDistillationTrainer(
    model, tokenizer, training_config, distillation_config
)
result = trainer.train(dataset)
```

### `FeatureDistillationTrainer`

Extends response distillation with MSE loss on intermediate hidden states for stronger layer-level supervision.

```python
from finetune_cli.core.types import FeatureDistillationConfig
from finetune_cli.trainers import FeatureDistillationTrainer

fd_config = FeatureDistillationConfig(
    teacher_model_name="gpt2-medium",
    temperature=2.0,
    alpha=0.5,           # KL divergence weight
    beta=0.3,            # hidden-state MSE weight
    feature_layers=None, # None = auto-select 4 evenly-spaced layers
)
trainer = FeatureDistillationTrainer(
    model, tokenizer, training_config, fd_config
)
result = trainer.train(dataset)
```

`feature_layers` accepts a list of student layer indices to supervise, e.g. `[0, 4, 8, 11]`. Each student layer is mapped to the proportionally corresponding teacher layer. Pass `None` for automatic selection.

### `TrainingResult`

Frozen dataclass returned by all `BaseTrainer` subclasses.

```python
result.output_dir             # Path — where model/adapter was saved
result.train_loss             # float
result.eval_loss              # float | None
result.epochs_completed       # int
result.steps_completed        # int
result.training_time_seconds  # float
result.trainer_logs           # Dict[str, Any] — raw HF Trainer log history
```

---

## Pruning — `finetune_cli.trainers`

Pruners are **not** `BaseTrainer` subclasses — they transform a model in-place rather than training it.

### `StructuredPruner`

Soft structured pruning. Scores each attention head by mean absolute weight magnitude, then zeros the bottom `sparsity` fraction per layer. The model shape is unchanged.

```python
from pathlib import Path
from finetune_cli.core.types import PruningConfig
from finetune_cli.trainers import StructuredPruner

pruning_config = PruningConfig(
    output_dir=Path("./outputs/pruned"),
    sparsity=0.3,           # fraction of heads to zero
    method="heads",         # "heads" (default) or "ffn"
    min_heads_per_layer=1,  # safety floor — never collapse a layer entirely
)
pruner = StructuredPruner(model, tokenizer, pruning_config)
result = pruner.prune()

result.output_dir              # Path
result.original_param_count    # int
result.zeroed_param_count      # int
result.sparsity_achieved       # float
result.heads_pruned_per_layer  # Dict[str, int] — layer name → heads zeroed
result.pruning_time_seconds    # float
```

`method="heads"` targets the query-projection rows of each attention layer. `method="ffn"` targets the gate/fc1 neuron rows of each FFN layer.

### `WandaPruner`

WANDA (Weight AND Activation) unstructured pruning. Scores each weight by `|W_ij| × ‖X_j‖₂` where `X` is the input activation norm, then zeros the bottom `sparsity` fraction. Requires a calibration dataset for best results; falls back to magnitude-only scoring without one.

```python
from finetune_cli.core.types import WandaConfig
from finetune_cli.trainers import WandaPruner
import torch

wanda_config = WandaConfig(
    output_dir=Path("./outputs/wanda"),
    sparsity=0.5,
    n_calibration_samples=128,
    calibration_seq_len=128,
    use_row_wise=True,       # per-output-row threshold (recommended)
    layer_types=None,        # None = auto (Linear + Conv1D)
)

# With calibration data (recommended)
calib_ids = torch.load("calib_ids.pt")   # (N, seq_len) token id tensor
pruner = WandaPruner(model, tokenizer, wanda_config)
result = pruner.prune(calibration_input_ids=calib_ids)

# Without calibration data (magnitude-only fallback)
result = pruner.prune()

result.output_dir           # Path
result.original_param_count # int (total weights across all target layers)
result.zeroed_param_count   # int
result.sparsity_achieved    # float
result.layers_pruned        # int — number of linear layers processed
result.pruning_time_seconds # float
```

---

## Evaluation — `finetune_cli.evaluation`

### `BenchmarkRunner`

```python
from finetune_cli.evaluation.benchmarker import BenchmarkRunner
from finetune_cli.core.types import EvaluationConfig, EvaluationMetric

eval_config = EvaluationConfig(
    metrics=[EvaluationMetric.ROUGE_L, EvaluationMetric.BLEU],
    num_samples=200,
)
runner = BenchmarkRunner(base_model, finetuned_model, tokenizer, eval_config)
report = runner.run(dataset)

report.summary()              # formatted string
report.base_scores            # Dict[str, float]
report.finetuned_scores       # Dict[str, float]
report.delta                  # Dict[str, float] — improvement per metric
```

### Individual metrics

```python
from finetune_cli.evaluation.metrics import RougeMetric, BleuMetric
from finetune_cli.core.types import EvaluationMetric

metric = RougeMetric(EvaluationMetric.ROUGE_L)
score = metric.compute(
    predictions=["the quick brown fox"],
    references=["the quick brown fox"],
)
# score = 1.0
```