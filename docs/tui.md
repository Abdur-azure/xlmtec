# TUI — Interactive Terminal Interface

`lmtool` ships with a full interactive terminal UI built on [Textual](https://textual.textualize.io/).
It wraps every CLI command in a keyboard-navigable form, streams live output, and displays results — all without leaving your terminal.

---

## Install & launch

```bash
pip install "textual>=0.52.0"
lmtool tui
```

---

## Home screen

```
┌─ lmtool ── LLM Fine-Tuning Toolkit ──────────────────────────── 12:34 ─┐
│                                                                                │
│        lmtool  --  LLM Fine-Tuning Toolkit                              │
│     Tab / Arrow keys to navigate   Enter or Click to select   Q to quit       │
│                                                                                │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐     │
│  │ 🚀  Train           │ │ 📊  Evaluate        │ │ ⚡  Benchmark       │     │
│  │                     │ │                     │ │                     │     │
│  │ Fine-tune with LoRA,│ │ Score a checkpoint  │ │ Compare base vs     │     │
│  │ QLoRA, DPO and more │ │ (ROUGE, BLEU, Perp) │ │ fine-tuned side-by  │     │
│  └─────────────────────┘ └─────────────────────┘ └─────────────────────┘     │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐     │
│  │ ☁️  Upload          │ │ 🔀  Merge           │ │ 💡  Recommend       │     │
│  │                     │ │                     │ │                     │     │
│  │ Push adapter or     │ │ Merge LoRA adapter  │ │ Get optimal config  │     │
│  │ merged model to Hub │ │ into standalone mdl │ │ for your hardware   │     │
│  └─────────────────────┘ └─────────────────────┘ └─────────────────────┘     │
│                                                                                │
│  q Quit  h Home                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Global keybindings

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `h` | Return to home screen |
| `Escape` | Return to home screen |
| `Tab` | Next field / next card |
| `Shift+Tab` | Previous field / previous card |
| `Arrow keys` | Navigate cards on home screen |
| `Enter` | Select card / submit form |
| `Ctrl+S` | Submit form (any form screen) |
| `Ctrl+C` | Cancel running command / quit |

---

## Commands

### Train

Fine-tune a model using any supported training method.

**Fields:**
- **Model name or path** — HuggingFace model id or local path (e.g. `gpt2`)
- **Training method** — LoRA, QLoRA, Full Fine-Tuning, Instruction Tuning, DPO, Response Distillation, Feature Distillation
- **Dataset path** — local `.jsonl` / `.json` / `.csv` file
- **Epochs** — number of training epochs (default: 3)
- **Learning rate** — default `2e-4`
- **Output directory** — where the adapter/model is saved

```
┌─ Train ─────────────────────────────────────────────────────────────────────┐
│ 🚀  Train — Fine-tune a model                                                │
│                                                                              │
│ Model name or path *                                                         │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │ gpt2                                                                     │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
│ Training method *                                                            │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │ LoRA (recommended)                                                     ▼ │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
│ ...                                                                          │
│  ▶ Run Training    ← Back                                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Evaluate

Score a checkpoint against a test dataset.

**Fields:** Model path, dataset path, metrics (ROUGE-1/2/L, BLEU, Perplexity), max samples, optional report file.

### Benchmark

Compare a base model against a fine-tuned model side-by-side.

**Fields:** Base model, fine-tuned model path, dataset, metrics, max samples, optional report.

### Upload

Push a model or adapter to HuggingFace Hub.

**Fields:**
- **Model path** — local adapter or model directory
- **Repository ID** — `username/repo-name`
- **HF Token** — masked input field; also reads `HF_TOKEN` env var
- **Private** — toggle to make repo private
- **Merge LoRA adapter** — toggle to merge adapter before upload; reveals **Base model** field

### Merge

Fuse a LoRA adapter into its base model to produce a standalone model.

**Fields:** Adapter directory, base model, output directory, dtype (float32 / float16 / bfloat16).

### Recommend

Analyse a model's size and available VRAM, then output an optimal training config.

**Fields:** Model name, optional output path to save the generated YAML config.

---

## Running screen

When you submit a form, the TUI switches to a live running view:

```
┌─ lmtool ─────────────────────────────────────────── 12:35 ─┐
│  ⚙  Training  gpt2  [lora]                              00:42    │
├───────────────────────────────────────────────────────────────────┤
│ $ lmtool train --model gpt2 --dataset ./data/...            │
│ Loading model gpt2...                                             │
│ ✓ Model loaded (124M parameters)                                  │
│ Loading dataset ./data/sample.jsonl...                            │
│ ✓ Dataset loaded: 500 samples                                     │
│ Epoch 1/3: 100%|████████████| 125/125 [00:38<00:00]              │
│ Train loss: 2.341                                                 │
│ ...                                                               │
├───────────────────────────────────────────────────────────────────┤
│  Running…                                        Cancel           │
└───────────────────────────────────────────────────────────────────┘
```

Press `q` or click **Cancel** to abort the running command.

---

## Result screen

On completion the TUI shows a result summary:

```
┌─ lmtool ─────────────────────────────────────────────────────┐
│                                                                     │
│          ✅  Command completed successfully                         │
│                                                                     │
│  Results                                                            │
│  ┌─────────────────┬────────────────────────┐                      │
│  │ Metric          │ Value                  │                      │
│  │─────────────────┼────────────────────────│                      │
│  │ Exit code       │ 0                      │                      │
│  │ Status          │ ✅ Success             │                      │
│  │ Duration        │ 00:42                  │                      │
│  └─────────────────┴────────────────────────┘                      │
│                                                                     │
│              🏠  Home          ✕  Quit                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token — used by Upload if `--token` is not filled in |
| `HUGGING_FACE_HUB_TOKEN` | Alternative HF token env var (also accepted) |

---

## Headless testing

The TUI ships with a full Pilot test suite:

```bash
pip install "pytest-asyncio>=0.21.0"
pytest tests/test_tui.py -v
```

All tests run headlessly without a terminal — safe for CI.