# Installation Guide

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or higher |
| GPU (optional) | CUDA-capable, 8GB+ VRAM recommended |
| RAM | 16GB minimum, 32GB recommended for larger models |
| Disk | 10GB+ for model downloads and checkpoints |

---

## Quick Install

### 1. Clone the repo

```bash
git clone https://github.com/Abdur-azure/lmtool.git
cd lmtool
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install

The package uses **optional extras** so you only pull what you need.

```bash
# Lightweight core only (CLI, config, evaluation — no GPU libraries)
pip install -e .

# Full ML stack (training, fine-tuning, pruning)
pip install -e ".[ml]"

# Interactive TUI
pip install -e ".[tui]"

# DPO training (requires trl)
pip install -e ".[dpo]"

# Everything at once (recommended for development)
pip install -e ".[full]"
```

### 4. Verify

```bash
lmtool --help
python -c "from lmtool.core.config import ConfigBuilder; print('OK')"
```

---

## Install from PyPI (when published)

```bash
# Core only
pip install lmtool

# With ML stack
pip install "lmtool[ml]"

# Everything
pip install "lmtool[full]"
```

---

## GPU Setup

### NVIDIA (CUDA)

Install PyTorch with the correct CUDA version for your driver:

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU only (slower but works everywhere)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the rest of the ML stack:

```bash
pip install -e ".[ml]"
```

Verify:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## HuggingFace Setup (for upload / private models)

```bash
huggingface-cli login
# or set the env variable:
export HF_TOKEN="hf_..."
```

---

## Development Install

```bash
pip install -e ".[full]"

# Run unit tests (no GPU needed)
pytest tests/ -v --ignore=tests/test_integration.py

# Run all tests including integration (CPU fine, downloads GPT-2 once)
pytest tests/ -v
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: torch` | Run `pip install -e ".[ml]"` |
| `ModuleNotFoundError: textual` | Run `pip install -e ".[tui]"` |
| `ModuleNotFoundError: trl` | Run `pip install -e ".[dpo]"` |
| `bitsandbytes` errors on CPU | Expected on CPU-only machines — QLoRA requires a CUDA GPU |
| CUDA version mismatch | Reinstall PyTorch with the correct `--index-url` (see GPU Setup above) |