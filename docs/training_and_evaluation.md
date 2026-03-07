# Training and Evaluation Guide

**Complete guide to training and evaluating LLMs with the fine-tuning framework**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Training Methods](#training-methods)
3. [Training Configuration](#training-configuration)
4. [Training Examples](#training-examples)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Benchmarking](#benchmarking)
7. [Complete Workflows](#complete-workflows)
8. [Best Practices](#best-practices)

---

## 🎯 Overview

The framework provides three main training methods and comprehensive evaluation capabilities:

### **Training Methods**
- **LoRA** - Parameter-efficient fine-tuning (0.1-1% parameters)
- **QLoRA** - Quantized LoRA for memory efficiency
- **Full Fine-tuning** - Train all parameters

### **Evaluation Features**
- **7 Metrics** - ROUGE, BLEU, Perplexity, F1, Exact Match, Accuracy
- **Benchmarking** - Compare base vs fine-tuned models
- **Reports** - Generate Markdown, JSON, or HTML reports

---

## 🚀 Training Methods

### **1. LoRA (Low-Rank Adaptation)**

**Overview:**
- Trains only 0.1-1% of parameters
- Adds lightweight adapter layers
- ~50% memory savings vs full fine-tuning
- Can merge adapters back into base model

**When to Use:**
- Most general-purpose fine-tuning tasks
- Medium to large models (1B+ parameters)
- When you need multiple task-specific adapters
- Balance between quality and efficiency

**Configuration:**
```python
from lmtool.core.config import ConfigBuilder
from lmtool.core.types import TrainingMethod

config = ConfigBuilder() \
    .with_model("gpt2") \
    .with_training(TrainingMethod.LORA, "./output", num_epochs=3) \
    .with_lora(
        r=8,                    # Rank (4, 8, 16, 32)
        lora_alpha=32,          # Scaling (typically 2-4x rank)
        lora_dropout=0.1        # Dropout (0.05-0.2)
    ) \
    .build()
```

**LoRA Parameters:**

| Parameter | Range | Recommended | Effect |
|-----------|-------|-------------|--------|
| **r (rank)** | 1-256 | 8-16 | Adapter capacity; higher = more parameters |
| **alpha** | 1-256 | 2-4× rank | Scaling factor; affects learning strength |
| **dropout** | 0.0-0.5 | 0.1 | Regularization; prevents overfitting |

**Example:**
```python
from lmtool.trainers import train_with_lora

result = train_with_lora(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    training_config=training_config,
    lora_config=lora_config,
    eval_dataset=val_data
)

print(f"Final Loss: {result.final_loss:.4f}")
print(f"Training Time: {result.training_time_seconds:.2f}s")
```

**Advanced: Merge Adapters**
```python
# After training
trainer = LoRATrainer(model, tokenizer, training_config, lora_config)
trainer.train(dataset)

# Merge adapters into base model
trainer.merge_and_save("./merged_model")
# Now you have a standalone model without PEFT dependency
```

---

### **2. QLoRA (Quantized LoRA)**

**Overview:**
- LoRA on quantized models (4-bit or 8-bit)
- ~75-88% memory savings vs full fine-tuning
- Enables 7B+ models on consumer GPUs
- Uses paged_adamw optimizer

**When to Use:**
- Large models (7B+ parameters)
- Consumer GPUs with limited VRAM
- Memory-constrained environments
- When 4-bit quantization is acceptable

**Configuration:**
```python
config = ConfigBuilder() \
    .with_model(
        "meta-llama/Llama-2-7b-hf",
        load_in_4bit=True,           # Enable 4-bit quantization
        device=DeviceType.AUTO
    ) \
    .with_training(
        TrainingMethod.QLORA,
        "./output",
        gradient_checkpointing=True   # Required for QLoRA
    ) \
    .with_lora(
        r=16,                         # Higher rank for quantized models
        lora_alpha=64
    ) \
    .build()
```

**Memory Requirements (7B Model):**

| Method | VRAM Required | Memory Savings |
|--------|---------------|----------------|
| Full FT | ~28GB | - |
| LoRA | ~14GB | 50% |
| QLoRA (8-bit) | ~7GB | 75% |
| QLoRA (4-bit) | ~4GB | 86% |

**Example:**
```python
from lmtool.trainers import train_with_qlora

result = train_with_qlora(
    model=quantized_model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    training_config=training_config,
    lora_config=lora_config,
    model_config=model_config  # Contains quantization info
)
```

**Best Practices:**
```python
from lmtool.trainers import get_qlora_best_practices

practices = get_qlora_best_practices()
# Returns comprehensive recommendations for QLoRA training
```

---

### **3. Full Fine-tuning**

**Overview:**
- Trains all model parameters
- Maximum adaptation capability
- Highest memory requirements
- Best for small models or unlimited GPU

**When to Use:**
- Small models (<1B parameters)
- Tasks requiring substantial model changes
- When GPU memory is abundant
- Maximum quality is priority

**Configuration:**
```python
config = ConfigBuilder() \
    .with_model("gpt2") \
    .with_training(
        TrainingMethod.FULL_FINETUNING,
        "./output",
        num_epochs=3,
        batch_size=2,                # Smaller batch for memory
        gradient_checkpointing=True  # Recommended
    ) \
    .build()
```

**Memory Estimation:**
```python
from lmtool.trainers import FullFineTuner

trainer = FullFineTuner(model, tokenizer, config)
memory = trainer.estimate_memory_usage()

print(f"Parameters: {memory['parameters_gb']:.2f} GB")
print(f"Gradients: {memory['gradients_gb']:.2f} GB")
print(f"Optimizer: {memory['optimizer_gb']:.2f} GB")
print(f"Total: {memory['total_estimated_gb']:.2f} GB")
```

**Example:**
```python
from lmtool.trainers import train_full_finetuning

result = train_full_finetuning(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    training_config=training_config,
    eval_dataset=val_data
)
```

---

## ⚙️ Training Configuration

### **Core Training Parameters**

```python
training_config = TrainingConfig(
    method=TrainingMethod.LORA,
    output_dir=Path("./outputs/model"),
    
    # Training schedule
    num_epochs=3,                    # Number of passes through data
    batch_size=4,                    # Samples per GPU
    gradient_accumulation_steps=4,   # Effective batch = 4 × 4 = 16
    
    # Optimization
    learning_rate=2e-4,              # Step size (1e-5 to 5e-4)
    weight_decay=0.01,               # Regularization
    warmup_ratio=0.1,                # 10% warmup
    lr_scheduler_type="cosine",      # Learning rate schedule
    max_grad_norm=1.0,               # Gradient clipping
    
    # Mixed precision
    fp16=False,                      # FP16 training (NVIDIA GPUs)
    bf16=False,                      # BF16 training (newer GPUs)
    
    # Checkpointing
    save_strategy="epoch",           # When to save checkpoints
    evaluation_strategy="epoch",     # When to evaluate
    load_best_model_at_end=True,    # Load best checkpoint after training
    
    # Optimization
    gradient_checkpointing=False,    # Trade compute for memory
    
    # Reproducibility
    seed=42
)
```

### **Learning Rate Selection**

| Model Size | Recommended LR | Range |
|------------|----------------|-------|
| Small (<500M) | 2e-4 | 1e-4 to 5e-4 |
| Medium (500M-3B) | 1e-4 | 5e-5 to 2e-4 |
| Large (3B+) | 5e-5 | 1e-5 to 1e-4 |

**Learning Rate Schedulers:**
- `linear` - Linear decay to 0
- `cosine` - Cosine annealing (recommended)
- `constant` - Fixed learning rate
- `polynomial` - Polynomial decay

### **Batch Size Guidelines**

**Effective Batch Size** = `batch_size × gradient_accumulation_steps × num_gpus`

| GPU VRAM | Batch Size | Accumulation | Effective |
|----------|------------|--------------|-----------|
| 8GB | 2 | 4 | 8 |
| 12GB | 4 | 4 | 16 |
| 16GB | 8 | 4 | 32 |
| 24GB | 16 | 4 | 64 |

**Tips:**
- Use gradient accumulation to simulate larger batches
- Larger effective batches = more stable training
- Smaller batches = faster iterations but noisier

---

## 📝 Training Examples

### **Example 1: Quick LoRA Training**

```python
from lmtool.core.config import ConfigBuilder
from lmtool.core.types import TrainingMethod, DatasetSource
from lmtool.models.loader import load_model_and_tokenizer
from lmtool.data import prepare_dataset
from lmtool.trainers import train_model

# 1. Configuration
config = ConfigBuilder() \
    .with_model("gpt2") \
    .with_dataset("./data.jsonl", source=DatasetSource.LOCAL_FILE) \
    .with_tokenization(max_length=512) \
    .with_training(TrainingMethod.LORA, "./output") \
    .with_lora(r=8, lora_alpha=32) \
    .build()

# 2. Load model
model, tokenizer = load_model_and_tokenizer(config.model.to_config())

# 3. Prepare data
dataset = prepare_dataset(
    config.dataset.to_config(),
    config.tokenization.to_config(),
    tokenizer
)

# 4. Train
result = train_model(
    model, tokenizer, dataset,
    config.training.to_config(),
    config.lora.to_config()
)

print(f"Training complete! Loss: {result.final_loss:.4f}")
```

### **Example 2: QLoRA on Large Model**

```python
# Configure for 7B model on 12GB GPU
config = ConfigBuilder() \
    .with_model(
        "meta-llama/Llama-2-7b-hf",
        load_in_4bit=True,
        device=DeviceType.CUDA
    ) \
    .with_dataset("HuggingFaceH4/ultrachat_200k", max_samples=10000) \
    .with_training(
        TrainingMethod.QLORA,
        "./outputs/llama-qlora",
        num_epochs=2,
        batch_size=2,
        gradient_accumulation_steps=8
    ) \
    .with_lora(r=16, lora_alpha=64, lora_dropout=0.1) \
    .build()

# Train
model, tokenizer = load_model_and_tokenizer(config.model.to_config())
dataset = prepare_dataset(...)
result = train_model(model, tokenizer, dataset, ...)
```

### **Example 3: Resume from Checkpoint**

```python
from pathlib import Path

# Original training
result = train_model(
    model, tokenizer, dataset,
    training_config,
    lora_config
)

# Resume from checkpoint
checkpoint_path = Path("./outputs/model/checkpoint-100")
result = train_model(
    model, tokenizer, dataset,
    training_config,
    lora_config,
    resume_from_checkpoint=checkpoint_path
)
```

### **Example 4: Training with Validation**

```python
# Prepare data with splits
splits = prepare_dataset(
    dataset_config,
    tokenization_config,
    tokenizer,
    split_for_validation=True,
    validation_ratio=0.1
)

# Train with validation
result = train_model(
    model, tokenizer,
    train_dataset=splits['train'],
    eval_dataset=splits['validation'],
    training_config=training_config,
    lora_config=lora_config
)
```

---

## 📊 Evaluation Metrics

### **Available Metrics**

| Metric | Description | Range | Higher is Better |
|--------|-------------|-------|------------------|
| **ROUGE-1** | Unigram overlap | 0-1 | ✓ |
| **ROUGE-2** | Bigram overlap | 0-1 | ✓ |
| **ROUGE-L** | Longest common subsequence | 0-1 | ✓ |
| **BLEU** | N-gram precision | 0-1 | ✓ |
| **Perplexity** | Prediction quality | 1-∞ | ✗ (lower is better) |
| **F1** | Token-level F1 score | 0-1 | ✓ |
| **Exact Match** | Exact string match | 0-1 | ✓ |

### **ROUGE Metrics**

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**

Measures overlap between generated and reference text.

```python
from lmtool.evaluation import evaluate_model
from lmtool.core.types import EvaluationMetric

config = EvaluationConfig(
    metrics=[
        EvaluationMetric.ROUGE_1,  # Unigram overlap
        EvaluationMetric.ROUGE_2,  # Bigram overlap
        EvaluationMetric.ROUGE_L   # Longest common subsequence
    ],
    batch_size=8
)

result = evaluate_model(model, tokenizer, test_dataset, config)
print(f"ROUGE-1: {result.metrics['rouge1']:.4f}")
print(f"ROUGE-2: {result.metrics['rouge2']:.4f}")
print(f"ROUGE-L: {result.metrics['rougeL']:.4f}")
```

**When to Use:**
- Summarization tasks
- Text generation quality
- Content preservation

### **BLEU Score**

**BLEU (Bilingual Evaluation Understudy)**

Measures n-gram precision between generated and reference.

```python
config = EvaluationConfig(
    metrics=[EvaluationMetric.BLEU],
    batch_size=8
)

result = evaluate_model(model, tokenizer, test_dataset, config)
print(f"BLEU: {result.metrics['bleu']:.4f}")
```

**When to Use:**
- Translation tasks
- Paraphrase generation
- When precision matters more than recall

### **Perplexity**

Measures how well the model predicts text. Lower is better.

```python
config = EvaluationConfig(
    metrics=[EvaluationMetric.PERPLEXITY],
    batch_size=8
)

result = evaluate_model(model, tokenizer, test_dataset, config)
print(f"Perplexity: {result.metrics['perplexity']:.2f}")
```

**When to Use:**
- Language modeling quality
- Model comparison
- Measuring fluency

**Interpretation:**
- `1.0` - Perfect prediction
- `10-50` - Excellent
- `50-100` - Good
- `100+` - Poor

### **F1 Score**

Token-level precision and recall balance.

```python
config = EvaluationConfig(
    metrics=[EvaluationMetric.F1],
    batch_size=8
)

result = evaluate_model(model, tokenizer, test_dataset, config)
print(f"F1: {result.metrics['f1']:.4f}")
```

**When to Use:**
- Information extraction
- Question answering
- When both precision and recall matter

### **Exact Match**

Percentage of predictions that exactly match reference.

```python
from lmtool.evaluation.metrics import ExactMatchMetric

# With normalization
metric = ExactMatchMetric(
    ignore_case=True,
    ignore_punctuation=True
)

# Without normalization
metric_strict = ExactMatchMetric(
    ignore_case=False,
    ignore_punctuation=False
)
```

**When to Use:**
- Classification tasks
- Structured output generation
- Strict correctness requirements

---

## 🔬 Benchmarking

### **Basic Benchmarking**

Compare base model vs fine-tuned model:

```python
from lmtool.evaluation import benchmark_models

result = benchmark_models(
    base_model=base_model,
    finetuned_model=finetuned_model,
    tokenizer=tokenizer,
    dataset=test_dataset,
    config=eval_config
)

print(f"Average Improvement: {result.get_average_improvement():.2f}%")
```

**Output:**
```
BENCHMARK RESULTS
═══════════════════════════════════════════════════════════════════════
Metric          Base         Fine-tuned   Improvement    
───────────────────────────────────────────────────────────────────────
rouge1          0.2345       0.3421       +45.86%
rouge2          0.1234       0.2156       +74.71%
rougeL          0.2123       0.3089       +45.50%
═══════════════════════════════════════════════════════════════════════
Average Improvement: +55.36%
═══════════════════════════════════════════════════════════════════════
```

### **Generate Reports**

```python
from lmtool.evaluation import ReportGenerator
from pathlib import Path

# Markdown report
ReportGenerator.save_report(
    result,
    Path("./reports/benchmark.md"),
    format="markdown",
    title="GPT-2 LoRA Fine-tuning Results"
)

# HTML report
ReportGenerator.save_report(
    result,
    Path("./reports/benchmark.html"),
    format="html"
)

# JSON report
ReportGenerator.save_report(
    result,
    Path("./reports/benchmark.json"),
    format="json"
)
```

### **Compare Pre-computed Metrics**

```python
from lmtool.evaluation import compare_metrics

base_metrics = {
    'rouge1': 0.25,
    'rouge2': 0.15,
    'rougeL': 0.22
}

finetuned_metrics = {
    'rouge1': 0.35,
    'rouge2': 0.23,
    'rougeL': 0.31
}

result = compare_metrics(base_metrics, finetuned_metrics)
print(result.improvements)
# Output: {'rouge1': 40.0, 'rouge2': 53.33, 'rougeL': 40.91}
```

---

## 🔄 Complete Workflows

### **Workflow 1: Train and Evaluate**

```python
from lmtool.core.config import ConfigBuilder
from lmtool.core.types import TrainingMethod, EvaluationMetric
from lmtool.models.loader import load_model_and_tokenizer
from lmtool.data import prepare_dataset
from lmtool.trainers import train_model
from lmtool.evaluation import evaluate_model

# 1. Configuration
config = ConfigBuilder() \
    .with_model("gpt2") \
    .with_dataset("./data.jsonl", max_samples=5000) \
    .with_tokenization(max_length=512) \
    .with_training(TrainingMethod.LORA, "./output") \
    .with_lora(r=8, lora_alpha=32) \
    .with_evaluation(
        metrics=[
            EvaluationMetric.ROUGE_1,
            EvaluationMetric.ROUGE_L,
            EvaluationMetric.PERPLEXITY
        ]
    ) \
    .build()

# 2. Load model
model, tokenizer = load_model_and_tokenizer(config.model.to_config())

# 3. Prepare data with splits
splits = prepare_dataset(
    config.dataset.to_config(),
    config.tokenization.to_config(),
    tokenizer,
    split_for_validation=True,
    validation_ratio=0.2
)

# 4. Train
train_result = train_model(
    model, tokenizer,
    train_dataset=splits['train'],
    eval_dataset=splits['validation'],
    training_config=config.training.to_config(),
    lora_config=config.lora.to_config()
)

# 5. Evaluate
eval_result = evaluate_model(
    model, tokenizer,
    splits['validation'],
    config.evaluation.to_config()
)

print(f"Training Loss: {train_result.final_loss:.4f}")
print(f"ROUGE-1: {eval_result.metrics['rouge1']:.4f}")
```

### **Workflow 2: Before/After Comparison**

```python
from lmtool.evaluation import benchmark_models
from pathlib import Path

# Load base model
base_model, tokenizer = load_model_and_tokenizer(model_config)

# Train
finetuned_model = base_model  # Will be modified by training
train_result = train_model(...)

# Reload base model for fair comparison
base_model_fresh, _ = load_model_and_tokenizer(model_config)

# Benchmark
benchmark_result = benchmark_models(
    base_model=base_model_fresh,
    finetuned_model=finetuned_model,
    tokenizer=tokenizer,
    dataset=test_dataset,
    config=eval_config,
    save_report=Path("./benchmark_report.md")
)

print(f"Average Improvement: {benchmark_result.get_average_improvement():.2f}%")
```

### **Workflow 3: Quick Evaluation**

```python
from lmtool.evaluation import quick_evaluate

# Fast evaluation without configuration
test_inputs = [
    "What is machine learning?",
    "Explain neural networks.",
    "Define artificial intelligence."
]

test_references = [
    "Machine learning is a subset of AI...",
    "Neural networks are computing systems...",
    "Artificial intelligence is the simulation..."
]

scores = quick_evaluate(model, tokenizer, test_inputs, test_references)
print(f"ROUGE-1: {scores['rouge1']:.4f}")
print(f"ROUGE-L: {scores['rougeL']:.4f}")
```

---

## 💡 Best Practices

### **Training Best Practices**

**1. Start Small, Scale Up**
```python
# Development
config = ConfigBuilder() \
    .with_dataset("./data.jsonl", max_samples=1000) \
    .with_training(num_epochs=1, batch_size=2) \
    .with_lora(r=4) \
    .build()

# Production
config = ConfigBuilder() \
    .with_dataset("./data.jsonl", max_samples=None) \
    .with_training(num_epochs=3, batch_size=8) \
    .with_lora(r=16) \
    .build()
```

**2. Monitor Training**
```python
# Enable logging
import logging
logging.basicConfig(level=logging.INFO)

# Save training state
from lmtool.trainers import LoRATrainer

trainer = LoRATrainer(...)
result = trainer.train(dataset)

# Access training state
print(f"Best loss: {trainer.state.best_loss:.4f}")
print(f"Best epoch: {trainer.state.best_epoch}")
print(f"Loss history: {trainer.state.loss_history}")
```

**3. Use Validation Sets**
```python
# Always split data
splits = prepare_dataset(..., split_for_validation=True)

# Train with validation
result = train_model(
    train_dataset=splits['train'],
    eval_dataset=splits['validation'],  # Important!
    ...
)
```

**4. Handle OOM Errors**
```python
try:
    result = train_model(...)
except OutOfMemoryError as e:
    print(f"OOM: {e}")
    print("Suggestions:")
    print("- Reduce batch_size")
    print("- Reduce max_length")
    print("- Enable gradient_checkpointing")
    print("- Use lower LoRA rank")
```

### **Evaluation Best Practices**

**1. Use Multiple Metrics**
```python
config = EvaluationConfig(
    metrics=[
        EvaluationMetric.ROUGE_1,
        EvaluationMetric.ROUGE_2,
        EvaluationMetric.ROUGE_L,
        EvaluationMetric.BLEU,
        EvaluationMetric.F1
    ]
)
```

**2. Separate Test Set**
```python
# Never evaluate on training data!
splits = prepare_dataset(..., validation_ratio=0.2)

# Use held-out validation set
eval_result = evaluate_model(
    model, tokenizer,
    splits['validation'],  # Not training set!
    eval_config
)
```

**3. Consistent Evaluation**
```python
# Use same config for fair comparison
eval_config = EvaluationConfig(
    metrics=[...],
    batch_size=8,
    generation_max_length=100,
    generation_temperature=0.7
)

base_result = evaluate_model(base_model, ..., eval_config)
ft_result = evaluate_model(finetuned_model, ..., eval_config)
```

### **Method Selection**

**Use the Method Recommender:**
```python
from lmtool.trainers import MethodRecommender

recommendation = MethodRecommender.recommend(
    model_size_params=124e6,   # GPT-2 small
    available_vram_gb=8.0,
    task_complexity="medium"
)

print(f"Recommended: {recommendation['recommendation'].value}")
print(f"Reason: {recommendation['reason']}")
```

**Decision Tree:**
```
┌─ Memory abundant (24GB+) ?
│  └─ Yes → Full Fine-tuning
│  └─ No → Continue
│
├─ Model size < 1B ?
│  └─ Yes → LoRA or Full Fine-tuning
│  └─ No → Continue
│
├─ Model size < 7B ?
│  └─ Yes → LoRA
│  └─ No → QLoRA
│
└─ Model size 7B+ ?
   └─ Yes → QLoRA (4-bit)
```

---

## 📚 Additional Resources

### **Configuration Examples**

See `examples/complete_training_pipeline.py` for:
- LoRA training
- Full fine-tuning
- Method recommendation
- Checkpoint resumption
- Config file usage

### **API Reference**

- [Trainers API](api.md#trainers)
- [Evaluation API](api.md#evaluation)
- [Configuration API](api.md#configuration)

### **Troubleshooting**

- [Training Issues](troubleshooting.md#training)
- [Memory Problems](troubleshooting.md#memory)
- [Evaluation Issues](troubleshooting.md#evaluation)

---

## 🎓 Summary

### **Quick Reference**

| Task | Command |
|------|---------|
| Train with LoRA | `train_model(..., TrainingMethod.LORA, lora_config)` |
| Train with QLoRA | `train_model(..., TrainingMethod.QLORA, lora_config, model_config)` |
| Full fine-tuning | `train_model(..., TrainingMethod.FULL_FINETUNING)` |
| Evaluate model | `evaluate_model(model, tokenizer, dataset, config)` |
| Benchmark models | `benchmark_models(base, finetuned, tokenizer, dataset, config)` |
| Quick eval | `quick_evaluate(model, tokenizer, inputs, references)` |

### **Key Takeaways**

1. **Start with LoRA** for most use cases
2. **Use QLoRA** for large models on consumer GPUs
3. **Monitor training** with validation sets
4. **Evaluate comprehensively** with multiple metrics
5. **Benchmark** before/after to measure improvement
6. **Save checkpoints** frequently
7. **Use gradient checkpointing** if memory constrained

---

**Last Updated:** 2025-01-29  
**Version:** 2.0.0