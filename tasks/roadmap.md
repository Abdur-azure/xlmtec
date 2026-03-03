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
| Feature Distillation | ✅ Built | Sprint 24 |

## Pruning

| Method | Status | Sprint |
|--------|--------|--------|
| Structured Pruning | ⬜ Not started | — |
| WANDA | ⬜ Not started | — |

---

## Build Order

1. **Response Distillation** ✅ — KL divergence, student mimics teacher logits. Single dep: `transformers`.
2. **Feature Distillation** ✅ — MSE on hidden states + KL on logits. Builds on #1.
3. **Structured Pruning** — remove attention heads / FFN layers. No retraining loop.
4. **WANDA** — weight + activation pruning, zero-shot, no gradient pass.

---

## Validation Notes

- Methods selected for commercial relevance (industry + OSS community signal, 2024).
- Cut: LoRA Distillation (niche), Speculative Distillation (inference concern, not training),
  Magnitude Pruning (superseded by WANDA), Movement Pruning (BERT-era, not validated on modern LLMs).

---

## TUI (Interactive Terminal Interface)

> **Rule: Do not touch any existing CLI, trainer, or test files during TUI sprints.**
> The TUI is purely additive. All Typer commands remain unchanged. All existing tests must stay green throughout all TUI sprints.

| Sprint | Name | Status | Delivers |
|--------|------|--------|----------|
| 25 | TUI Foundation | ⬜ Not started | App skeleton, home screen, 6 cards, `finetune-cli tui` entry point |
| 26 | TUI Train & Recommend | ⬜ Not started | Form→Run→Result flow proven on 2 commands |
| 27 | TUI Evaluate, Benchmark, Merge | ⬜ Not started | 3 more command screens reusing Sprint 26 patterns |
| 28 | TUI Upload + Polish | ⬜ Not started | Final screen, CSS theme, UX polish, docs |

### New files only — nothing existing is modified

```
finetune_cli/tui/
  __init__.py
  app.py               root Textual App, screen router, global keybindings
  app.css              Textual CSS theme (Sprint 28)
  screens/
    __init__.py
    home.py            6-card grid, keyboard + mouse navigation
    train.py           model, method dropdown, dataset, epochs, output
    recommend.py       model name, output path
    evaluate.py        checkpoint path, metrics multi-select
    benchmark.py       base model, finetuned path, dataset, metrics
    merge.py           adapter path, base model, dtype selector
    upload.py          path, repo_id, token (masked), private toggle
    running.py         Worker thread + live LogPanel, Ctrl+C cancel
    result.py          success/error display, metrics table, back button
  widgets/
    __init__.py
    command_card.py    styled card, hover state, label + description
    log_panel.py       scrolling live log output
    metric_table.py    Rich table widget for results

tests/test_tui.py      Textual Pilot headless tests (added each sprint)
docs/tui.md            TUI usage guide (Sprint 28)
```

### UX flow

```
Home Screen (Sprint 25)
  6 command cards — arrow key nav + mouse click

  Sprint 26:  Train → Form → Running → Result → back
              Recommend → Form → Running → Result → back

  Sprint 27:  Evaluate  → Form → Running → Result → back
              Benchmark → Form → Running → Result → back
              Merge     → Form → Running → Result → back

  Sprint 28:  Upload    → Form → Running → Result → back
              + CSS theme + validation polish + docs

Global keybindings (every screen):
  q / ctrl+c    quit
  esc / h       back to home
  tab           next field
  enter         submit form
```

### Dependency

```
textual>=0.52.0   add to pyproject.toml [project.dependencies] in Sprint 25
```

### Version targets

| Sprint | Version |
|--------|---------|
| 25 | TUI Foundation | 3.8.0 |
| 26 | TUI Train & Recommend | 3.9.0 |
| 27 | TUI Evaluate, Benchmark, Merge | 3.10.0 |
| 28 | TUI Upload + Polish | 3.11.0 |