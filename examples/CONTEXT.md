# examples/ — Context

Runnable examples. Every config must work after running `generate_sample_data.py`.

## Files

| File | Purpose |
|------|---------|
| `generate_sample_data.py` | Generates `data/sample.jsonl` (500 rows, causal LM) and `data/instructions.jsonl` (300 rows, alpaca). Pure stdlib, zero deps. |
| `configs/lora_gpt2.yaml` | LoRA on GPT-2, CPU-safe, points to `data/sample.jsonl` |
| `configs/qlora_llama.yaml` | QLoRA on LLaMA-3.2-1B, 6–8GB VRAM, HF Hub dataset |
| `configs/instruction_tuning.yaml` | InstructionTrainer on GPT-2, points to `data/instructions.jsonl` |
| `configs/full_finetuning.yaml` | FullFineTuner on GPT-2, small dataset, CPU-safe |

## Quickstart

```bash
python examples/generate_sample_data.py
lmtool train --config examples/configs/lora_gpt2.yaml
```

## Rules

- `generate_sample_data.py` must use **stdlib only** — it runs before `pip install -e .` succeeds
- All config `path:` fields must point to files created by `generate_sample_data.py`
- `qlora_llama.yaml` is the only config that requires a GPU and HF token — note this clearly in the file header
- Example configs double as integration test fixtures — keep them runnable