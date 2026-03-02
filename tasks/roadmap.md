# Feature Roadmap

Validated methods for commercial relevance. Build order: simplest → most complex.

---

## Fine-Tuning

| Method | Status | Sprint |
|--------|--------|--------|
| LoRA | ✅ Built | Sprint 2 |
| QLoRA | ✅ Built | Sprint 2 |
| Full Fine-Tuning | ✅ Built | Sprint 2 |
| Instruction Tuning | ✅ Built | Sprint 2 |
| DPO | ✅ Built | Sprint 8 |

## Distillation

| Method | Status | Sprint |
|--------|--------|--------|
| Response Distillation | ✅ Built | Sprint 23 |
| Feature Distillation | ⬜ Not started | — |

## Pruning

| Method | Status | Sprint |
|--------|--------|--------|
| Structured Pruning | ⬜ Not started | — |
| WANDA | ⬜ Not started | — |

---

## Build Order

1. **Response Distillation** ✅ — KL divergence, student mimics teacher logits. Single dep: `transformers`.
2. **Feature Distillation** — student matches teacher hidden states. Builds on #1.
3. **Structured Pruning** — remove attention heads / FFN layers. No retraining loop.
4. **WANDA** — weight + activation pruning, zero-shot, no gradient pass.

---

## Validation Notes

- Methods selected for commercial relevance (industry + OSS community signal, 2024).
- Cut: LoRA Distillation (niche), Speculative Distillation (inference concern, not training),
  Magnitude Pruning (superseded by WANDA), Movement Pruning (BERT-era, not validated on modern LLMs).