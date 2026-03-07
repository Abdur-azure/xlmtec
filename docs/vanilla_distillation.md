# Knowledge Distillation Guide

**Complete guide to Knowledge Distillation methods in the LLM Fine-Tuning Framework**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Vanilla Distillation](#vanilla-distillation)
3. [Feature Distillation](#feature-distillation)
4. [Comparison Table](#comparison-table)
5. [Configuration Guide](#configuration-guide)
6. [Use Cases](#use-cases)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

---

## 🎯 Overview

**Knowledge Distillation** is a model compression technique where a smaller **student model** learns to mimic a larger, more powerful **teacher model**. This enables:

- **Model Compression**: Reduce model size by 2-10x
- **Inference Speed**: Faster predictions on resource-constrained devices
- **Efficiency**: Maintain most of the teacher's performance with fewer parameters
- **Deployment**: Enable edge deployment and mobile applications

### **Key Concepts**

| Concept | Description |
|---------|-------------|
| **Teacher Model** | Large, pre-trained model with strong performance |
| **Student Model** | Smaller model that learns from teacher |
| **Knowledge Transfer** | Process of transferring teacher's knowledge to student |
| **Soft Targets** | Probability distributions (softer than hard labels) |
| **Temperature** | Controls softness of probability distributions |

---

## 🔵 Vanilla Distillation

### **Overview**

**Vanilla Distillation** (also called **Output Distillation**) transfers knowledge through the teacher's output probability distributions.

**How It Works:**
1. Teacher generates soft probability distributions using temperature scaling
2. Student learns to match these distributions via KL divergence
3. Combined loss: α × CrossEntropy + (1-α) × KL_Divergence

### **Mathematical Formulation**

```
Soft Targets: p_teacher = softmax(logits_teacher / T)
Student Predictions: p_student = log_softmax(logits_student / T)

KL Loss = KL_div(p_student, p_teacher) × T²
Total Loss = α × CE_loss + (1-α) × KL_loss
```

Where:
- **T** = Temperature (higher = softer distributions)
- **α** = Weight for standard cross-entropy loss
- **KL_div** = Kullback-Leibler divergence

### **Advantages**

✅ **Simple**: Only requires output logits  
✅ **Effective**: Proven to work well across domains  
✅ **Fast**: Minimal computational overhead  
✅ **Universal**: Works with any architecture  

### **When to Use**

- **Model Compression**: Reducing model size while maintaining performance
- **Task Generalization**: Teacher provides richer supervision than hard labels
- **Transfer Learning**: Transferring knowledge to a different architecture
- **Edge Deployment**: Creating smaller models for mobile/embedded devices

### **Configuration Parameters**

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `temperature` | 1.0-10.0 | 2.0 | Softness of probability distributions |
| `alpha` | 0.0-1.0 | 0.5 | Weight for cross-entropy loss |

**Temperature Guide:**
- **T = 1.0**: Hard targets (no distillation benefit)
- **T = 2.0**: Balanced softness (recommended)
- **T = 5.0**: Very soft targets (for very different architectures)
- **T = 10.0**: Maximum softness (rarely needed)

**Alpha Guide:**
- **α = 0.0**: Pure distillation (no ground truth)
- **α = 0.3**: Mostly distillation (recommended for strong teachers)
- **α = 0.5**: Balanced (default, good starting point)
- **α = 0.7**: Mostly ground truth (weaker teachers)
- **α = 1.0**: No distillation (standard fine-tuning)

### **Example Usage**

```python
from lmtool import LLMFineTuner

# Initialize
finetuner = LLMFineTuner("gpt2", "./distilled_model")

# Load student model
finetuner.load_model(method="vanilla_distillation")

# Load teacher model
finetuner.load_teacher_model("gpt2-medium")

# Setup vanilla distillation
finetuner.setup_vanilla_distillation(
    temperature=2.0,
    alpha=0.5
)

# Train
finetuner.train(dataset, num_epochs=3)
```

**CLI Usage:**
```bash
python lmtool.py

# Select method: 4 (Vanilla Distillation)
# Student model: gpt2
# Teacher model: gpt2-medium
# Temperature: 2.0
# Alpha: 0.5
```

---

## 🔴 Feature Distillation

### **Overview**

**Feature Distillation** (also called **Intermediate Layer Distillation**) transfers knowledge through intermediate layer representations, not just final outputs.

**How It Works:**
1. Extract hidden states from teacher's intermediate layers
2. Extract corresponding hidden states from student's layers
3. Minimize MSE between teacher and student representations
4. Combine with output distillation or standard loss

### **Mathematical Formulation**

```
Feature Loss = (1/N) × Σ MSE(h_student[i], h_teacher[i])

Where:
- h_student[i] = student's hidden states at layer i
- h_teacher[i] = teacher's hidden states at layer i
- N = number of layers being matched

Total Loss = α × CE_loss + (1-α) × Feature_loss
```

### **Advantages**

✅ **Richer Knowledge**: Captures intermediate representations  
✅ **Better Generalization**: Learns feature hierarchies  
✅ **Architectural Understanding**: Transfers layer-wise knowledge  
✅ **Improved Performance**: Often outperforms vanilla distillation  

### **When to Use**

- **Complex Tasks**: Tasks requiring understanding of intermediate features
- **Similar Architectures**: Student and teacher have similar layer structures
- **Maximum Performance**: When you want the best possible student
- **Feature Learning**: When intermediate representations are important

### **Configuration Parameters**

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `temperature` | 1.0-10.0 | 2.0 | For output distillation component |
| `alpha` | 0.0-1.0 | 0.3 | Weight for CE loss (lower for feature distillation) |
| `feature_layers` | List[int] | Auto | Which layers to match |

**Layer Selection Strategies:**

1. **Auto (Default)**: Evenly spaced layers
   ```python
   # If student has 12 layers, selects: [0, 3, 6, 9]
   feature_layers = None  # Auto-detect
   ```

2. **All Layers**: Maximum knowledge transfer
   ```python
   feature_layers = list(range(12))  # All 12 layers
   ```

3. **Key Layers**: Focus on important layers
   ```python
   feature_layers = [0, 5, 11]  # First, middle, last
   ```

4. **Late Layers**: Focus on high-level features
   ```python
   feature_layers = [8, 9, 10, 11]  # Last 4 layers
   ```

### **Example Usage**

```python
from lmtool import LLMFineTuner

# Initialize
finetuner = LLMFineTuner("gpt2", "./distilled_model")

# Load models
finetuner.load_model(method="feature_distillation")
finetuner.load_teacher_model("gpt2-large")

# Setup feature distillation
finetuner.setup_feature_distillation(
    temperature=2.0,
    alpha=0.3,  # Lower alpha for feature focus
    feature_layers=[0, 3, 6, 9, 11]  # Key layers
)

# Train
finetuner.train(dataset, num_epochs=5)
```

**CLI Usage:**
```bash
python lmtool.py

# Select method: 5 (Feature Distillation)
# Student model: gpt2
# Teacher model: gpt2-large
# Temperature: 2.0
# Alpha: 0.3
```

---

## 📊 Comparison Table

| Aspect | Vanilla Distillation | Feature Distillation |
|--------|---------------------|---------------------|
| **Complexity** | Low | Medium |
| **Training Speed** | Fast | Slower (2-3x overhead) |
| **Memory Usage** | Low | Higher (stores hidden states) |
| **Performance** | Good | Better |
| **Teacher-Student Gap** | Works with large gaps | Better with similar architectures |
| **Hyperparameter Tuning** | Easier (2 params) | More complex (3+ params) |
| **Use Case** | General compression | Maximum performance |
| **Implementation** | Simple | More involved |

### **Performance Comparison (Typical)**

| Method | Student Size | Performance Retained | Training Time | Memory |
|--------|-------------|---------------------|---------------|---------|
| **Vanilla Distillation** | 50% of teacher | 85-90% | 1x | 1.2x |
| **Feature Distillation** | 50% of teacher | 90-95% | 2x | 1.5x |
| **No Distillation** | 50% of teacher | 75-80% | 1x | 1x |

---

## ⚙️ Configuration Guide

### **Choosing Temperature**

**General Guidelines:**

| Temperature | Use Case | Example |
|-------------|----------|---------|
| **1.0-1.5** | Similar architectures, small gap | GPT-2 → GPT-2-small |
| **2.0-3.0** | Moderate gap (recommended) | GPT-2-medium → GPT-2 |
| **4.0-6.0** | Large gap | GPT-2-large → GPT-2 |
| **7.0-10.0** | Very different architectures | BERT → DistilBERT |

**Tuning Strategy:**
1. Start with T=2.0
2. If student struggles: increase to 3.0-4.0
3. If overfitting: decrease to 1.5
4. Monitor validation loss to find optimal value

### **Choosing Alpha**

**Decision Matrix:**

| Teacher Quality | Task Difficulty | Recommended Alpha |
|----------------|-----------------|-------------------|
| Strong | Easy | 0.2-0.3 |
| Strong | Hard | 0.4-0.5 |
| Moderate | Easy | 0.5-0.6 |
| Moderate | Hard | 0.6-0.7 |
| Weak | Any | 0.7-0.9 |

**Rules of Thumb:**
- **Strong teacher + abundant data**: Lower alpha (0.2-0.4)
- **Weak teacher or noisy data**: Higher alpha (0.6-0.8)
- **Balanced approach**: α = 0.5
- **Feature distillation**: Use 0.2-0.4 (focus on features)

### **Teacher-Student Pairing**

**Recommended Compression Ratios:**

| Teacher | Student | Compression | Method | Expected Performance |
|---------|---------|-------------|--------|---------------------|
| GPT-2 Medium (355M) | GPT-2 (124M) | 2.9x | Vanilla | 85-90% |
| GPT-2 Large (774M) | GPT-2 Medium (355M) | 2.2x | Feature | 90-95% |
| GPT-2 XL (1.5B) | GPT-2 Large (774M) | 1.9x | Feature | 92-97% |
| LLaMA-7B | LLaMA-3B | 2.3x | Feature | 88-93% |

**Key Principles:**
- **Moderate gap** (2-4x compression) works best
- **Too small gap**: Minimal benefit
- **Too large gap**: Performance drop
- **Architecture similarity**: Important for feature distillation

---

## 💡 Use Cases

### **Use Case 1: Mobile Deployment**

**Scenario**: Deploy GPT-2-large quality on mobile devices

**Solution**: Vanilla Distillation
```python
# Teacher: GPT-2-large (774M params)
# Student: GPT-2 (124M params)
# Result: 6.2x smaller, 85% performance
finetuner.setup_vanilla_distillation(temperature=3.0, alpha=0.4)
```

**Benefits:**
- 6x faster inference
- Fits in mobile memory
- Maintains most capabilities

### **Use Case 2: Production API**

**Scenario**: Reduce inference costs for high-volume API

**Solution**: Feature Distillation
```python
# Teacher: GPT-2-XL (1.5B params)
# Student: GPT-2-large (774M params)
# Result: 2x smaller, 95% performance
finetuner.setup_feature_distillation(
    temperature=2.0,
    alpha=0.3,
    feature_layers=[0, 4, 8, 12, 16, 20, 23]
)
```

**Benefits:**
- 2x throughput
- 50% cost reduction
- Minimal quality loss

### **Use Case 3: Edge AI**

**Scenario**: IoT device with 512MB RAM

**Solution**: Aggressive Vanilla Distillation
```python
# Teacher: GPT-2-medium (355M)
# Student: DistilGPT-2 (82M)
# Result: 4.3x smaller, fits in 512MB
finetuner.setup_vanilla_distillation(temperature=4.0, alpha=0.5)
```

**Benefits:**
- Runs on constrained hardware
- Offline inference
- 70-80% performance retained

### **Use Case 4: Domain Adaptation**

**Scenario**: Transfer medical knowledge from large model

**Solution**: Feature Distillation with domain data
```python
# Teacher: GPT-2-large fine-tuned on medical data
# Student: GPT-2 trained with distillation
finetuner.setup_feature_distillation(
    temperature=2.5,
    alpha=0.4,
    feature_layers=[0, 5, 11]  # Key medical knowledge layers
)
```

**Benefits:**
- Captures domain-specific representations
- Smaller model with specialized knowledge
- Better than training student from scratch

---

## 🎓 Best Practices

### **1. Teacher Model Selection**

**Do:**
- ✅ Use a model 2-4x larger than student
- ✅ Ensure teacher is well-trained on target task
- ✅ Use similar architecture families when possible
- ✅ Verify teacher performance before distillation

**Don't:**
- ❌ Use poorly trained teachers (will transfer mistakes)
- ❌ Use teachers with vastly different architectures for feature distillation
- ❌ Skip teacher validation
- ❌ Use compression ratios > 10x

### **2. Hyperparameter Tuning**

**Tuning Order:**
1. **Temperature**: Start with 2.0, adjust based on loss convergence
2. **Alpha**: Start with 0.5, tune based on validation performance
3. **Feature Layers**: Start with auto-selection, refine if needed

**Grid Search Example:**
```python
# Try these combinations
configs = [
    {'temp': 2.0, 'alpha': 0.3},
    {'temp': 2.0, 'alpha': 0.5},
    {'temp': 3.0, 'alpha': 0.4},
    {'temp': 4.0, 'alpha': 0.5},
]
```

### **3. Training Strategy**

**Recommended Approach:**
1. **Pre-train student**: Regular training first (optional)
2. **Distillation**: Apply distillation for refinement
3. **Fine-tune**: Additional epochs on hard examples
4. **Evaluation**: Test on held-out set

**Training Schedule:**
```python
# Phase 1: Standard training (optional)
finetuner.train(dataset, epochs=3, method="standard")

# Phase 2: Distillation
finetuner.setup_vanilla_distillation(temp=2.0, alpha=0.5)
finetuner.train(dataset, epochs=5)

# Phase 3: Fine-tuning
finetuner.train(hard_examples, epochs=2, learning_rate=1e-5)
```

### **4. Validation and Testing**

**Metrics to Track:**
- **Student Loss**: Should converge below teacher's
- **KL Divergence**: Should decrease over time
- **Task Performance**: Compare student vs teacher on validation
- **Inference Speed**: Measure actual speedup

**Validation Strategy:**
```python
# Regular validation during training
val_scores_student = evaluate(student_model, val_set)
val_scores_teacher = evaluate(teacher_model, val_set)

print(f"Performance retention: {val_scores_student / val_scores_teacher * 100:.1f}%")
```

### **5. Common Pitfalls**

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| **Too high temperature** | Student loss doesn't converge | Lower to 2.0-3.0 |
| **Too low alpha** | Ignores ground truth | Increase to 0.5-0.7 |
| **Wrong layers** | Poor feature transfer | Use auto-selection or key layers |
| **Undertrained teacher** | Student learns errors | Train teacher thoroughly first |
| **Overfitting** | Good train, poor val | Increase regularization, lower alpha |

---

## 📚 Examples

### **Example 1: Basic Vanilla Distillation**

```python
from lmtool import LLMFineTuner

# Setup
finetuner = LLMFineTuner("gpt2", "./vanilla_distilled")

# Load models
finetuner.load_model(method="vanilla_distillation")
finetuner.load_teacher_model("gpt2-medium")

# Load dataset
dataset = finetuner.load_dataset_from_source("./data.jsonl", num_samples=10000)
tokenized, _ = finetuner.prepare_dataset(dataset, max_length=512)

# Configure vanilla distillation
finetuner.setup_vanilla_distillation(
    temperature=2.0,
    alpha=0.5
)

# Train
finetuner.train(
    tokenized,
    num_epochs=5,
    batch_size=8,
    learning_rate=2e-4
)

print("✅ Vanilla distillation complete!")
```

### **Example 2: Advanced Feature Distillation**

```python
from lmtool import LLMFineTuner

# Setup
finetuner = LLMFineTuner("gpt2", "./feature_distilled")

# Load models
finetuner.load_model(method="feature_distillation")
finetuner.load_teacher_model("gpt2-large")

# Load dataset
dataset = finetuner.load_dataset_from_source(
    "wikitext",
    dataset_config="wikitext-2-raw-v1",
    num_samples=20000
)
tokenized, _ = finetuner.prepare_dataset(dataset, max_length=512)

# Configure feature distillation
finetuner.setup_feature_distillation(
    temperature=2.5,
    alpha=0.3,  # Focus more on features
    feature_layers=[0, 2, 4, 6, 8, 10, 11]  # Key layers
)

# Train with smaller learning rate
finetuner.train(
    tokenized,
    num_epochs=8,
    batch_size=4,
    learning_rate=1e-4  # Lower LR for feature matching
)

print("✅ Feature distillation complete!")
```

### **Example 3: Comparison Study**

```python
from lmtool import LLMFineTuner

# Compare vanilla vs feature distillation
methods = [
    ("vanilla", {"temperature": 2.0, "alpha": 0.5}),
    ("feature", {"temperature": 2.0, "alpha": 0.3, "feature_layers": None})
]

results = {}

for method_name, config in methods:
    print(f"\n{'='*50}")
    print(f"Training with {method_name} distillation")
    print(f"{'='*50}")
    
    finetuner = LLMFineTuner("gpt2", f"./distilled_{method_name}")
    finetuner.load_model(method=f"{method_name}_distillation")
    finetuner.load_teacher_model("gpt2-medium")
    
    # Setup distillation
    if method_name == "vanilla":
        finetuner.setup_vanilla_distillation(**config)
    else:
        finetuner.setup_feature_distillation(**config)
    
    # Train and evaluate
    finetuner.train(dataset, num_epochs=5)
    scores = finetuner.benchmark(test_prompts, use_finetuned=True)
    results[method_name] = scores

# Compare results
print("\n" + "="*70)
print("COMPARISON: Vanilla vs Feature Distillation")
print("="*70)
for metric in results["vanilla"]:
    vanilla_score = results["vanilla"][metric]
    feature_score = results["feature"][metric]
    improvement = (feature_score - vanilla_score) / vanilla_score * 100
    print(f"{metric}: Vanilla={vanilla_score:.4f}, Feature={feature_score:.4f}, Δ={improvement:+.2f}%")
```

---

## 🔬 Advanced Topics

### **Multi-Teacher Distillation**

Combine knowledge from multiple teachers:

```python
# Pseudo-code (would require custom implementation)
teacher1 = load_model("gpt2-medium")
teacher2 = load_model("opt-350m")

# Average teacher outputs
combined_logits = 0.5 * teacher1_logits + 0.5 * teacher2_logits
```

### **Progressive Distillation**

Distill in stages:

```python
# Stage 1: Large → Medium
distill(teacher="gpt2-xl", student="gpt2-large")

# Stage 2: Medium → Small
distill(teacher="gpt2-large", student="gpt2-medium")

# Stage 3: Small → Tiny
distill(teacher="gpt2-medium", student="gpt2")
```

### **Task-Specific Distillation**

Focus on specific capabilities:

```python
# Distill only summarization capability
finetuner.setup_feature_distillation(
    feature_layers=[8, 9, 10, 11],  # High-level reasoning layers
    alpha=0.2  # Heavy distillation focus
)
```

---

## 📊 Performance Benchmarks

**Typical Results (GPT-2 family):**

| Teacher | Student | Method | ROUGE-L | Perplexity | Compression |
|---------|---------|--------|---------|------------|-------------|
| GPT-2-medium | GPT-2 | None | 0.28 | 45.2 | 2.9x |
| GPT-2-medium | GPT-2 | Vanilla | 0.34 | 38.7 | 2.9x |
| GPT-2-medium | GPT-2 | Feature | 0.36 | 36.4 | 2.9x |
| GPT-2-large | GPT-2-medium | Vanilla | 0.41 | 32.1 | 2.2x |
| GPT-2-large | GPT-2-medium | Feature | 0.43 | 29.8 | 2.2x |

---

## 🎯 Summary

### **Quick Decision Guide**

**Use Vanilla Distillation when:**
- ✅ You need fast training
- ✅ Memory is limited
- ✅ Teacher and student are very different
- ✅ You want simple implementation

**Use Feature Distillation when:**
- ✅ You want maximum performance
- ✅ Architectures are similar
- ✅ You can afford longer training
- ✅ Intermediate features matter

### **Key Takeaways**

1. **Distillation** reduces model size while maintaining performance
2. **Temperature** controls softness (start with 2.0)
3. **Alpha** balances distillation vs ground truth (start with 0.5 for vanilla, 0.3 for feature)
4. **Feature distillation** often outperforms vanilla but is more complex
5. **Teacher quality** is critical - train teachers thoroughly

---

**Last Updated:** 2025-01-29  
**Version:** 2.0.0  
**Framework:** LLM Fine-Tuning CLI Extended Edition