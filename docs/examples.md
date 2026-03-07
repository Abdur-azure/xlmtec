# Examples

Practical examples for common fine-tuning scenarios.

## Example 1: Fine-tune GPT-2 on Custom Dialogue Data

### Scenario
You have a JSONL file with conversational data and want to create a chatbot.

### Dataset Format (`dialogue.jsonl`)

```json
{"prompt": "Hello, how are you?", "response": "I'm doing great! How can I help you today?"}
{"prompt": "What's the weather like?", "response": "I don't have access to weather data, but you can check weather.com"}
```

### Configuration

```bash
python lmtool.py

# Interactive prompts:
Model name: gpt2
Output directory: ./dialogue_model
Dataset path: ./dialogue.jsonl
Limit samples: yes
Number of samples: 5000
Max sequence length: 256
LoRA r: 8
LoRA alpha: 32
LoRA dropout: 0.1
Epochs: 5
Batch size: 8
Learning rate: 2e-4
Upload to HuggingFace: no
```

### Expected Results

```
📊 PERFORMANCE COMPARISON
Metric       Base Model      Fine-tuned      Improvement
ROUGE1       0.1823          0.3245          +78.03%
ROUGE2       0.0912          0.2134          +133.99%
ROUGEL       0.1645          0.2987          +81.58%
```

### Using the Fine-tuned Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./dialogue_model")

# Generate response
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Example 2: Domain Adaptation for Medical Text

### Scenario
Adapt a model to medical terminology and clinical notes.

### Dataset
Using HuggingFace medical dataset.

### Configuration

```bash
python lmtool.py

# Interactive prompts:
Model name: facebook/opt-350m
Output directory: ./medical_model
Dataset name: medical_meadow_medical_flashcards
Dataset config: default
Split: train
Limit samples: yes
Number of samples: 20000
Max sequence length: 512
LoRA r: 16
LoRA alpha: 64
LoRA dropout: 0.15
Epochs: 3
Batch size: 4
Learning rate: 1e-4
Upload to HuggingFace: yes
Repo name: myusername/opt-medical-350m
```

### Why These Settings?

- **Higher rank (16)**: Medical domain requires learning specialized terminology
- **Higher alpha (64)**: Stronger adaptation to domain-specific patterns
- **More dropout (0.15)**: Medical text can be noisy, prevent overfitting
- **Lower learning rate (1e-4)**: Conservative to preserve general knowledge

## Example 3: Summarization Task

### Scenario
Fine-tune for news article summarization.

### Dataset Format (`summaries.csv`)

```csv
article,summary
"Long article text here...","Brief summary here..."
"Another article...","Another summary..."
```

### Configuration

```bash
python lmtool.py

# Interactive prompts:
Model name: google/flan-t5-base
Output directory: ./summarization_model
Dataset path: ./summaries.csv
Limit samples: yes
Number of samples: 15000
Max sequence length: 1024
LoRA r: 8
LoRA alpha: 32
LoRA dropout: 0.1
Epochs: 3
Batch size: 4
Learning rate: 2e-4
```

### Data Preparation Tips

The tool auto-detects columns. For best results:

1. Name columns clearly: `article`, `text`, `summary`, `content`
2. Clean your data: Remove HTML tags, special characters
3. Balance length: Keep articles similar length when possible

## Example 4: Code Generation

### Scenario
Fine-tune on code examples for Python code generation.

### Dataset
Using HuggingFace code dataset with specific file selection.

### Configuration

```bash
python lmtool.py

# Interactive prompts:
Model name: Salesforce/codegen-350M-mono
Output directory: ./code_model
Dataset name: codeparrot/github-code
Dataset config: python
Load specific file: yes
File path: train-00000-of-00200.parquet
Number of samples: 10000
Max sequence length: 512
LoRA r: 8
LoRA alpha: 32
LoRA dropout: 0.05
Epochs: 2
Batch size: 8
Learning rate: 2e-4
```

### Why These Settings?

- **Lower dropout (0.05)**: Code has consistent structure, less noise
- **Fewer epochs (2)**: Code datasets are large, less overfitting risk
- **Specific file selection**: Avoids loading entire 200-file dataset

## Example 5: Question Answering

### Scenario
Fine-tune on SQuAD-style Q&A data.

### Configuration

```bash
python lmtool.py

# Interactive prompts:
Model name: distilbert-base-uncased
Output directory: ./qa_model
Dataset name: squad
Dataset config: plain_text
Split: train
Limit samples: yes
Number of samples: 30000
Max sequence length: 384
LoRA r: 8
LoRA alpha: 32
LoRA dropout: 0.1
Epochs: 3
Batch size: 16
Learning rate: 3e-4
```

### Inference Example

```python
from transformers import pipeline
from peft import PeftModel, AutoModelForQuestionAnswering

# Load fine-tuned model
base_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
model = PeftModel.from_pretrained(base_model, "./qa_model")

# Create QA pipeline
qa = pipeline("question-answering", model=model, tokenizer="distilbert-base-uncased")

# Ask question
context = "Paris is the capital of France. It is known for the Eiffel Tower."
question = "What is the capital of France?"
result = qa(question=question, context=context)
print(result['answer'])  # "Paris"
```

## Example 6: Multi-language Fine-tuning

### Scenario
Fine-tune multilingual model for specific languages.

### Configuration

```bash
python lmtool.py

# Interactive prompts:
Model name: xlm-roberta-base
Output directory: ./multilingual_model
Dataset name: Helsinki-NLP/tatoeba
Dataset config: eng-spa
Split: train
Limit samples: yes
Number of samples: 25000
Max sequence length: 128
LoRA r: 16
LoRA alpha: 32
LoRA dropout: 0.1
Epochs: 4
Batch size: 8
Learning rate: 2e-4
```

## Example 7: Sentiment Analysis

### Scenario
Adapt model for sentiment classification.

### Dataset Format (`sentiment.jsonl`)

```json
{"text": "This product is amazing!", "sentiment": "positive"}
{"text": "Terrible experience, would not recommend.", "sentiment": "negative"}
```

### Configuration

```bash
python lmtool.py

# Interactive prompts:
Model name: roberta-base
Output directory: ./sentiment_model
Dataset path: ./sentiment.jsonl
Limit samples: yes
Number of samples: 10000
Max sequence length: 256
LoRA r: 4
LoRA alpha: 16
LoRA dropout: 0.1
Epochs: 5
Batch size: 16
Learning rate: 3e-4
```

### Why These Settings?

- **Lower rank (4)**: Sentiment is relatively simple classification
- **Higher batch size (16)**: Shorter sequences allow larger batches
- **More epochs (5)**: Smaller dataset benefits from more iterations

## Example 8: Instruction Following

### Scenario
Fine-tune on instruction-response pairs.

### Dataset
Using large instruction dataset with selective loading.

### Configuration

```bash
python lmtool.py

# Interactive prompts:
Model name: EleutherAI/pythia-410m
Output directory: ./instruction_model
Dataset name: HuggingFaceH4/ultrachat_200k
Load specific file: yes
File path: data/train_sft-00000-of-00004.parquet
Number of samples: 15000
Max sequence length: 1024
LoRA r: 16
LoRA alpha: 64
LoRA dropout: 0.1
Epochs: 2
Batch size: 4
Learning rate: 1e-4
Upload to HuggingFace: yes
Repo name: myusername/pythia-instruction-410m
Private: no
```

## Best Practices from Examples

### Dataset Size Guidelines

- **Experiments**: 1,000-5,000 samples
- **Development**: 5,000-20,000 samples
- **Production**: 20,000+ samples

### Model Selection Tips

1. **Start small**: Test with GPT-2 or OPT-125M
2. **Match architecture**: Use decoder models (GPT) for generation
3. **Consider license**: Check model licensing for commercial use
4. **Resource awareness**: Larger models need more VRAM

### Common Pitfalls to Avoid

1. ❌ Training on too few samples (< 500)
2. ❌ Using max_length longer than needed (wastes memory)
3. ❌ Setting rank too high for simple tasks (overfitting)
4. ❌ Forgetting to validate results before uploading
5. ❌ Not monitoring GPU memory usage

### Performance Optimization

```python
# For faster iteration during development:
- Use smaller model variants
- Limit samples to 1000-5000
- Reduce max_length
- Use rank 4-8

# For production quality:
- Use full dataset
- Increase rank to 16
- Train for more epochs
- Validate on held-out test set
```

## Recipes Summary

| Use Case | Model | Rank | Samples | Max Length |
|----------|-------|------|---------|------------|
| Chatbot | GPT-2 | 8 | 5k | 256 |
| Medical | OPT-350M | 16 | 20k | 512 |
| Summarization | FLAN-T5 | 8 | 15k | 1024 |
| Code Gen | CodeGen | 8 | 10k | 512 |
| QA | DistilBERT | 8 | 30k | 384 |
| Multilingual | XLM-R | 16 | 25k | 128 |
| Sentiment | RoBERTa | 4 | 10k | 256 |
| Instructions | Pythia | 16 | 15k | 1024 |

## Next Steps

- Learn about [Configuration Options](configuration.md)
- Check [Troubleshooting](troubleshooting.md) for common issues
- Review [API Reference](api.md) for programmatic usage