# 🏗️ Architecture Overview

**Version:** 2.0 (FAANG-Grade Refactor)  
**Last Updated:** 2025-01-29

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [System Architecture](#system-architecture)
4. [Module Breakdown](#module-breakdown)
5. [Data Flow](#data-flow)
6. [Extension Points](#extension-points)
7. [Migration from v1](#migration-from-v1)

---

## 🎯 Overview

This framework provides a **production-grade, modular, and extensible** system for fine-tuning Large Language Models. Built following FAANG engineering standards with:

- **Type Safety:** Full type hints with protocols and type checking
- **Modularity:** Clean separation of concerns across layers
- **Extensibility:** Plugin architecture for new methods and data sources
- **Testability:** Dependency injection and interface-based design
- **Observability:** Comprehensive logging and error handling
- **Configuration:** Pydantic-based config with validation

---

## 🧭 Design Principles

### 1. **Separation of Concerns**
Each module has a single, well-defined responsibility:
- `core/` - Type definitions, configuration, exceptions
- `models/` - Model loading and management
- `data/` - Dataset loading and processing
- `trainers/` - Training implementations
- `evaluation/` - Metrics and benchmarking
- `cli/` - User interface

### 2. **Composition Over Inheritance**
Uses protocols (interfaces) rather than deep inheritance hierarchies:
```python
class ModelLoader(Protocol):
    def load_model(self, config: ModelConfig) -> PreTrainedModel: ...
```

### 3. **Dependency Injection**
Components receive dependencies explicitly:
```python
def __init__(self, tokenizer: PreTrainedTokenizer, config: TokenizationConfig):
    self.tokenizer = tokenizer
    self.config = config
```

### 4. **Factory & Registry Patterns**
Dynamic selection of implementations:
```python
registry = DatasetLoaderRegistry()
loader = registry.get_loader(config)  # Auto-selects based on config
```

### 5. **Immutable Configuration**
Config objects are frozen dataclasses preventing accidental mutation:
```python
@dataclass(frozen=True)
class ModelConfig:
    name: str
    device: DeviceType
```

### 6. **Fail Fast**
Validate at config layer, not execution layer:
```python
class ModelConfigModel(BaseModel):
    @model_validator(mode='after')
    def validate_quantization(self) -> 'ModelConfigModel':
        if self.load_in_8bit and self.load_in_4bit:
            raise IncompatibleConfigError(...)
```

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  train   │  │evaluate  │  │benchmark │  │  upload  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
┌───────▼─────────────▼─────────────▼─────────────▼──────────┐
│                    Pipeline Layer                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Training Pipeline Orchestrator              │   │
│  └──┬────────────────┬────────────────┬─────────────┬──┘   │
└─────┼────────────────┼────────────────┼─────────────┼──────┘
      │                │                │             │
┌─────▼────────┐  ┌────▼─────┐  ┌──────▼──────┐  ┌───▼──────┐
│    Models    │  │   Data   │  │  Trainers   │  │   Eval   │
│              │  │          │  │             │  │          │
│ ┌──────────┐ │  │ ┌──────┐ │  │ ┌─────────┐ │  │ ┌──────┐ │
│ │  Loader  │ │  │ │Loader│ │  │ │  LoRA   │ │  │ │Metric│ │
│ └──────────┘ │  │ └──────┘ │  │ └─────────┘ │  │ └──────┘ │
│ ┌──────────┐ │  │ ┌──────┐ │  │ ┌─────────┐ │  │ ┌──────┐ │
│ │ Detector │ │  │ │Proces│ │  │ │  QLoRA  │ │  │ │Bench │ │
│ └──────────┘ │  │ └──────┘ │  │ └─────────┘ │  │ └──────┘ │
└──────────────┘  └──────────┘  └─────────────┘  └──────────┘
      │                │                │             │
┌─────▼────────────────▼────────────────▼─────────────▼──────┐
│                      Core Layer                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Types   │  │  Config  │  │Exception │  │ Logging  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Module Breakdown

### **Core (`core/`)**

#### `types.py` - Type System
- **Enums:** `TrainingMethod`, `DatasetSource`, `EvaluationMetric`, etc.
- **Dataclasses:** Immutable config objects (`ModelConfig`, `TrainingConfig`, etc.)
- **Protocols:** Interface definitions (`ModelLoader`, `Trainer`, `Evaluator`)
- **Results:** Structured output types (`TrainingResult`, `EvaluationResult`)

**Design:** Centralized type definitions prevent duplication and ensure consistency.

#### `exceptions.py` - Exception Hierarchy
```
FineTuneError (base)
├── ConfigurationError
│   ├── InvalidConfigError
│   ├── MissingConfigError
│   └── IncompatibleConfigError
├── ModelError
│   ├── ModelLoadError
│   ├── ModelNotFoundError
│   └── UnsupportedModelError
├── DatasetError
│   ├── DatasetLoadError
│   ├── NoTextColumnsError
│   └── EmptyDatasetError
├── TrainingError
│   ├── OutOfMemoryError
│   ├── NaNLossError
│   └── CheckpointError
└── EvaluationError
    └── MetricComputationError
```

**Design:** Specific exceptions enable precise error handling and actionable messages.

#### `config.py` - Configuration Management
- **Pydantic Models:** Validation + serialization (`ModelConfigModel`, etc.)
- **Builders:** Programmatic config construction
- **I/O:** JSON/YAML loading and saving
- **Validation:** Cross-field validation and type conversion

**Example:**
```python
config = ConfigBuilder() \
    .with_model("gpt2") \
    .with_dataset("./data.jsonl", source=DatasetSource.LOCAL_FILE) \
    .with_training(TrainingMethod.LORA, "./output") \
    .with_lora(r=8, lora_alpha=32) \
    .build()
```

---

### **Models (`models/`)**

#### `loader.py` - Model Loading
**Components:**
- `ModelLoader`: Load models with quantization, device mapping, Flash Attention
- `TargetModuleDetector`: Auto-detect LoRA target modules

**Features:**
- Device auto-detection (CUDA/MPS/CPU)
- 4-bit/8-bit quantization
- Flash Attention 2 support
- Gradient checkpointing
- Memory-efficient loading

**Example:**
```python
loader = ModelLoader()
model, tokenizer = loader.load(model_config)

detector = TargetModuleDetector(model)
target_modules = detector.detect()  # Auto-detect for LoRA
```

---

### **Data (`data/`)**

#### `base.py` - Abstract Interfaces
- `DatasetLoader` protocol
- `DatasetProcessor` protocol
- `DatasetStatistics`, `DatasetAnalyzer`, `DatasetFilter`

#### `loaders.py` - Loading Implementations
**Loaders:**
- `LocalFileLoader`: JSON, JSONL, CSV, Parquet, TXT
- `HuggingFaceLoader`: Hub datasets with streaming

**Registry:**
```python
registry = DatasetLoaderRegistry()
loader = registry.get_loader(config)  # Auto-selects
dataset = loader.load(config)
```

#### `processors.py` - Processing & Tokenization
**Strategies:**
- `SingleColumnStrategy`: One text column
- `MultiColumnStrategy`: Multiple columns combined
- `InstructionStrategy`: Instruction-response format

**Auto-detection:**
```python
detector = TextColumnDetector()
columns = detector.detect(dataset)  # Finds text columns
```

#### `pipeline.py` - Complete Pipeline
**DataPipeline:**
```python
pipeline = DataPipeline(dataset_config, tokenization_config, tokenizer)
dataset = pipeline.run(split_for_validation=True, validation_ratio=0.1)
# Returns: {'train': Dataset, 'validation': Dataset}
```

**Quick functions:**
```python
# Minimal config
dataset = quick_load("./data.jsonl", tokenizer, max_samples=1000)

# Full config
dataset = prepare_dataset(dataset_config, tokenization_config, tokenizer)
```

---

### **Utils (`utils/`)**

#### `logging.py` - Logging Infrastructure
- Colored console output
- File logging
- Context managers (`LogProgress`, `LogContext`)
- Specialized loggers (`log_model_info`, `log_dataset_info`)

**Example:**
```python
logger = setup_logger("my_module", level=LogLevel.INFO, log_file=Path("log.txt"))

with LogProgress(logger, "Training model"):
    train()  # Automatically logs start/end time
```

---

## 🔄 Data Flow

### **Training Pipeline Flow:**

```
1. Configuration
   ├─ Load config from file/CLI
   ├─ Validate with Pydantic
   └─ Convert to immutable dataclasses

2. Model Loading
   ├─ Load model from HuggingFace
   ├─ Apply quantization (optional)
   ├─ Setup device mapping
   └─ Detect target modules

3. Data Pipeline
   ├─ Load dataset (local/HF)
   ├─ Detect text columns
   ├─ Filter invalid samples
   ├─ Tokenize
   └─ Split train/val

4. Training
   ├─ Initialize trainer (LoRA/QLoRA/Full)
   ├─ Setup LoRA adapters
   ├─ Train with HF Trainer
   └─ Save checkpoints

5. Evaluation
   ├─ Compute metrics (ROUGE/BLEU/etc)
   ├─ Compare base vs fine-tuned
   └─ Generate report

6. Upload (optional)
   └─ Push to HuggingFace Hub
```

---

## 🔌 Extension Points

### **Adding a New Data Source:**

```python
class CustomLoader(DatasetLoader):
    def can_handle(self, config: DatasetConfig) -> bool:
        return config.source == DatasetSource.CUSTOM
    
    def load(self, config: DatasetConfig) -> Dataset:
        # Your loading logic
        return dataset

# Register
from lmtool.data import register_loader
register_loader(CustomLoader())
```

### **Adding a New Training Method:**

```python
class CustomTrainer(Trainer):
    def train(self, model, dataset, config) -> TrainingResult:
        # Your training logic
        return result

# Register in factory
TrainerFactory.register(TrainingMethod.CUSTOM, CustomTrainer)
```

### **Adding a New Metric:**

```python
class CustomMetric(Metric):
    def compute(self, predictions, references) -> float:
        # Your metric logic
        return score

# Register
MetricRegistry.register(EvaluationMetric.CUSTOM, CustomMetric)
```

---

## 🔄 Migration from v1

### **Old (Monolithic):**
```python
# Single 400-line file
finetuner = LLMFineTuner("gpt2", "./output")
finetuner.load_model()
dataset = finetuner.load_dataset_from_source("./data.json")
tokenized = finetuner.prepare_dataset(dataset)
finetuner.setup_lora(r=8)
finetuner.train(tokenized)
```

### **New (Modular):**
```python
# Separate concerns
from lmtool.core.config import ConfigBuilder
from lmtool.core.types import DatasetSource, TrainingMethod
from lmtool.models.loader import load_model_and_tokenizer
from lmtool.data import prepare_dataset
from lmtool.trainers import LoRATrainer

# 1. Configuration (validated)
config = ConfigBuilder() \
    .with_model("gpt2") \
    .with_dataset("./data.json", source=DatasetSource.LOCAL_FILE) \
    .with_training(TrainingMethod.LORA, "./output", num_epochs=3) \
    .with_lora(r=8, lora_alpha=32) \
    .build()

# 2. Load model (with auto-detection)
model, tokenizer = load_model_and_tokenizer(config.model.to_config())

# 3. Prepare data (with pipeline)
dataset = prepare_dataset(
    config.dataset.to_config(),
    config.tokenization.to_config(),
    tokenizer
)

# 4. Train (with specific trainer)
trainer = LoRATrainer(model, tokenizer, config.training.to_config(), config.lora.to_config())
result = trainer.train(dataset)
```

### **Benefits of New Architecture:**
- ✅ **Testable:** Each component can be tested independently
- ✅ **Type-safe:** Full type checking with mypy
- ✅ **Extensible:** Add new methods without modifying core
- ✅ **Maintainable:** Clear responsibilities, no 400-line files
- ✅ **Documented:** Self-documenting with types and protocols
- ✅ **Production-ready:** Error handling, logging, validation

---

## 📊 Comparison: v1 vs v2

| Aspect | v1 (Monolithic) | v2 (Modular) |
|--------|-----------------|--------------|
| **Lines per file** | 400+ | <200 |
| **Type hints** | None | Complete |
| **Testing** | Impossible | Unit + Integration |
| **Extensibility** | Modify core | Plugin system |
| **Error handling** | Generic | Specific exceptions |
| **Configuration** | Hardcoded | Validated configs |
| **Logging** | Print statements | Structured logging |
| **Methods** | 1 (LoRA) | 20+ (planned) |
| **Maintainability** | Poor | Excellent |

---

## 🚀 Next Steps

### **Phase 3: Trainer System** (Next Priority)
- Abstract `Trainer` base class
- `LoRATrainer` implementation
- `QLoRATrainer` implementation
- `FullFineTuner` implementation
- `TrainerFactory` for method selection

### **Phase 4: Evaluation System**
- Metric implementations (ROUGE, BLEU, Perplexity)
- Benchmarking pipeline
- Comparison reports

### **Phase 5: CLI Interface**
- Typer-based CLI
- Subcommands for each operation
- Interactive mode
- Config file support

---

## 📚 Additional Resources

- **Configuration Guide:** See `docs/configuration.md`
- **API Reference:** See `docs/api.md`
- **Examples:** See `examples/` directory
- **Contributing:** See `CONTRIBUTING.md`

---

**Last Updated:** 2025-01-29  
**Architecture Version:** 2.0.0  
**Status:** Phase 2 Complete (Data Pipeline)