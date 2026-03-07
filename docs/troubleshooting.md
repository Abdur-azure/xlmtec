# Troubleshooting

Solutions to common issues when using the fine-tuning tool.

## Installation Issues

### Issue: CUDA Not Available

**Symptoms:**
```
CUDA Available: False
Device: cpu
```

**Causes:**

1. No NVIDIA GPU
2. CUDA drivers not installed
3. PyTorch installed without CUDA support

**Solutions:**

**Check GPU:**
```bash
nvidia-smi
```

**Reinstall PyTorch with CUDA:**
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Verify installation:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Module Not Found Error

**Symptoms:**
```
ModuleNotFoundError: No module named 'peft'
```

**Solution:**

```bash
pip install --upgrade -r requirements.txt
```

If issue persists:
```bash
pip install peft transformers datasets --upgrade
```

### Issue: Version Conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver...
```

**Solution:**

Create fresh virtual environment:
```bash
python -m venv fresh_env
source fresh_env/bin/activate  # Windows: fresh_env\Scripts\activate
pip install -r requirements.txt
```

## Memory Issues

### Issue: CUDA Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions (try in order):**

**1. Reduce batch size:**
```
Batch size: 2  # Instead of 4 or 8
```

**2. Reduce max sequence length:**
```
Max length: 256  # Instead of 512 or 1024
```

**3. Reduce LoRA rank:**
```
LoRA r: 4  # Instead of 8 or 16
```

**4. Limit number of samples:**
```
Number of samples: 1000  # For testing
```

**5. Enable gradient checkpointing** (edit code):
```python
# Add to model loading (line 59)
model.gradient_checkpointing_enable()
```

**6. Use smaller model:**
```
Model name: gpt2  # Instead of gpt2-medium or gpt2-large
```

**Memory calculation formula:**
```
Required VRAM ≈ batch_size × max_length² × model_size / 1e9 GB
```

### Issue: CPU Out of Memory

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

**1. Limit dataset size:**
```
Number of samples: 5000
```

**2. Use streaming for large datasets:**

Modify code to add streaming:
```python
dataset = load_dataset(dataset_source, split=split, streaming=True)
```

**3. Increase system swap:**
```bash
# Linux
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Training Issues

### Issue: Loss Not Decreasing

**Symptoms:**
```
Epoch 1: Loss 3.45
Epoch 2: Loss 3.44
Epoch 3: Loss 3.43
```

**Causes & Solutions:**

**1. Learning rate too low:**
```
Learning rate: 2e-4  # Increase from 1e-4
```

**2. Model frozen:**
Check that LoRA is properly applied:
```
🎯 Setting up LoRA configuration...
trainable params: X || all params: Y || trainable%: Z
```

**3. Insufficient training:**
```
Epochs: 5  # Increase from 3
Samples: 10000  # Increase from 1000
```

**4. Data quality issues:**
- Check dataset has meaningful text
- Verify columns are correctly detected
- Ensure no empty or null values

### Issue: Loss Diverging (NaN)

**Symptoms:**
```
Epoch 1: Loss 2.34
Epoch 2: Loss 12.45
Epoch 3: Loss NaN
```

**Causes & Solutions:**

**1. Learning rate too high:**
```
Learning rate: 1e-4  # Reduce from 5e-4 or 1e-3
```

**2. Gradient explosion:**

Add gradient clipping (edit code):
```python
# In TrainingArguments (line 223)
max_grad_norm=1.0
```

**3. Data issues:**
- Remove extreme outliers
- Check for special characters causing issues
- Normalize text inputs

### Issue: Overfitting

**Symptoms:**

- Training loss decreases but validation would increase
- ROUGE scores decrease on new data
- Model outputs repetitive text

**Solutions:**

**1. Increase dropout:**
```
LoRA dropout: 0.2  # Increase from 0.1
```

**2. Reduce epochs:**
```
Epochs: 2  # Reduce from 5
```

**3. Add more training data:**
```
Number of samples: 20000  # Increase from 5000
```

**4. Reduce model capacity:**
```
LoRA r: 4  # Reduce from 8 or 16
```

**5. Early stopping** (edit code):
```python
# Add to TrainingArguments
early_stopping_patience=2
```

## Dataset Issues

### Issue: No Text Columns Detected

**Symptoms:**
```
ValueError: No text columns found in dataset
```

**Solutions:**

**Check dataset structure:**
```python
print(dataset.column_names)
print(dataset[0])
```

**Manual column specification** (edit code around line 120):
```python
text_columns = ["my_text_column", "my_content_column"]
tokenized_dataset, _ = finetuner.prepare_dataset(dataset, text_columns=text_columns)
```

### Issue: Dataset Too Large

**Symptoms:**
- Slow loading
- Memory issues
- Long preprocessing

**Solutions:**

**1. Use selective file loading:**
```
Load specific file: yes
File path: train-00000-of-00100.parquet  # Load only one shard
```

**2. Limit samples aggressively:**
```
Number of samples: 5000
```

**3. Use streaming mode:**

Modify dataset loading:
```python
dataset = load_dataset(dataset_source, streaming=True)
dataset = dataset.take(num_samples)
```

### Issue: Column Names Not Recognized

**Symptoms:**

Tool doesn't detect your text columns properly.

**Common column names recognized:**

- `text`, `content`, `input`, `output`
- `prompt`, `response`, `instruction`
- `question`, `answer`, `summary`

**Solution:**

Rename your columns or modify detection logic (line 103):
```python
common_names = ['text', 'content', 'your_column_name']
```

## Model Issues

### Issue: Model Not Found

**Symptoms:**
```
OSError: Can't find model 'xyz'
```

**Solution:**

**Verify model exists:**
- Check [HuggingFace Models](https://huggingface.co/models)
- Ensure exact name match (case-sensitive)

**Common model names:**
```
✅ gpt2
✅ facebook/opt-125m
✅ EleutherAI/pythia-410m
❌ GPT-2 (wrong case)
❌ opt-125m (missing organization)
```

### Issue: Model Architecture Not Supported

**Symptoms:**
```
Target modules not found
LoRA cannot be applied
```

**Solution:**

**Check supported architectures:**

- ✅ GPT-2, GPT-Neo, GPT-J
- ✅ OPT, BLOOM, LLaMA
- ✅ T5, FLAN-T5
- ❌ BERT (requires different task type)

**Manual target module specification:**

Find module names:
```python
for name, module in model.named_modules():
    print(name)
```

Specify manually in setup_lora call.

### Issue: Tokenizer Warnings

**Symptoms:**
```
Token indices sequence length is longer than specified maximum
```

**Solution:**

This is informational. To suppress:
```
Max sequence length: 512  # Match your typical text length
```

Or truncate more aggressively.

## Upload Issues

### Issue: Authentication Failed

**Symptoms:**
```
HTTPError: 401 Client Error: Unauthorized
```

**Solutions:**

**1. Check token:**
- Get new token: https://huggingface.co/settings/tokens
- Ensure "Write" permission enabled

**2. Login via CLI:**
```bash
huggingface-cli login
```

**3. Set environment variable:**
```bash
export HUGGING_FACE_HUB_TOKEN="hf_xxxxxxxxxxxxx"
```

### Issue: Repository Already Exists

**Symptoms:**
```
HTTPError: 409 Conflict
```

**Solutions:**

**1. Use existing repository:**
```
Create new repository: no
```

**2. Choose different name:**
```
Repo name: username/new-model-name-v2
```

**3. Delete old repository:**
- Go to repository settings on HuggingFace
- Delete repository
- Try again

### Issue: Upload Failed

**Symptoms:**
```
Error uploading model: Connection timeout
```

**Solutions:**

**1. Check internet connection**

**2. Retry upload:**
The tool supports resumable uploads.

**3. Manual upload:**
```bash
huggingface-cli upload username/repo-name ./finetuned_model
```

## Performance Issues

### Issue: Training Too Slow

**Symptoms:**

- < 1 iteration/second
- Hours for small datasets

**Solutions:**

**1. Use GPU:**
Verify CUDA is enabled:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**2. Reduce sequence length:**
```
Max length: 256  # Instead of 512 or 1024
```

**3. Increase batch size:**
```
Batch size: 8  # If memory allows
```

**4. Use mixed precision:**
Automatically enabled on GPU (FP16).

**5. Reduce dataset size for testing:**
```
Number of samples: 1000
```

### Issue: Poor Fine-tuning Results

**Symptoms:**

- ROUGE scores barely improve
- Model outputs generic responses

**Solutions:**

**1. Increase model capacity:**
```
LoRA r: 16  # Increase from 8
LoRA alpha: 64  # Increase from 32
```

**2. Train longer:**
```
Epochs: 5  # Increase from 3
```

**3. Check data quality:**
- Ensure diverse, high-quality examples
- Remove duplicates
- Balance dataset

**4. Use larger base model:**
```
Model name: facebook/opt-1.3b  # Instead of opt-125m
```

**5. Increase training data:**
```
Number of samples: 20000  # Instead of 5000
```

## Debugging Tips

### Enable Verbose Logging

```bash
export TRANSFORMERS_VERBOSITY=debug
export PEFT_VERBOSITY=debug
python lmtool.py
```

### Monitor GPU Usage

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log to file
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 1 > gpu_log.csv
```

### Check Model Size

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
print(f"Params: {model.num_parameters() / 1e6:.1f}M")
```

### Validate Dataset

```python
from datasets import load_dataset

dataset = load_dataset("your_dataset")
print(dataset)
print(dataset[0])
print(dataset.column_names)
```

## Getting Help

If issues persist:

1. **Check logs**: Review error messages carefully
2. **Search issues**: [GitHub Issues](https://github.com/Abdur-azure/lmtool/issues)
3. **Open new issue**: Include:
   - Error message
   - Configuration used
   - System info (GPU, Python version)
   - Steps to reproduce

## Common Error Messages Reference

| Error | Likely Cause | Quick Fix |
|-------|--------------|-----------|
| CUDA OOM | Memory exceeded | Reduce batch size |
| NaN loss | Learning rate too high | Reduce learning rate |
| No text columns | Column names not recognized | Check dataset structure |
| 401 Unauthorized | Invalid HF token | Re-login to HuggingFace |
| Connection timeout | Network issue | Retry upload |
| Module not found | Missing dependency | Reinstall requirements |
| Model not found | Wrong model name | Check spelling |

## Next Steps

- Review [Configuration Guide](configuration.md) for optimization
- Check [Examples](examples.md) for working configurations
- See [API Reference](api.md) for programmatic usage