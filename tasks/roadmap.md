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
| Structured Pruning | ✅ Built | Sprint 29 |
| WANDA | ✅ Built | Sprint 30 |

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
| 25 | TUI Foundation | ✅ Complete | App skeleton, home screen, 6 cards, `xlmtec tui` entry point |
| 26 | TUI Train & Recommend | ✅ Complete | Form→Run→Result flow proven on 2 commands |
| 27 | TUI Evaluate, Benchmark, Merge | ✅ Complete | 3 more command screens reusing Sprint 26 patterns |
| 28 | TUI Upload + Polish | ✅ Complete | Final screen, CSS theme, UX polish, docs |

### New files only — nothing existing is modified

```
xlmtec/tui/
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

---

## CLI & Tooling (Sprints 34–44)

| Sprint | Name | Status | Delivers |
|--------|------|--------|----------|
| 34 | AI Integrations | ✅ Complete | Claude/Gemini/Codex providers, ai-suggest, 24 tests |
| 35 | CLI UX Polish | ✅ Complete | --version, dry-run, config validate, rich panels, 22 tests |
| 36 | Model Hub | ✅ Complete | hub search/info/trending, HfApi wrapper, 21 tests |
| 37 | Docs Overhaul | ✅ Complete | README, installation, usage, ai_integrations, hub docs |
| 38 | Checkpoint Resume | ✅ Complete | CheckpointManager, xlmtec resume, 22 tests |
| 39 | Config Templates | ✅ Complete | 7 built-in templates, xlmtec template list/show/use, 26 tests |
| 40 | Evaluation Dashboard | ✅ Complete | RunReader, RunComparator, xlmtec dashboard compare/show, 22 tests |
| 41 | Export Formats | ✅ Complete | ONNX/GGUF/safetensors backends, xlmtec export, 30 tests |
| 42 | Batch Inference | ✅ Complete | DataLoader, BatchPredictor, xlmtec predict, 24 tests |
| 43 | Plugin System | ✅ Complete | PluginStore, PluginLoader, xlmtec plugin add/list/remove, 21 tests |
| 44 | CHANGELOG + Audit | ✅ Complete | CHANGELOG Sprints 34-43, audit_repo all new files, v3.24.0 |

---

## Upcoming Sprints (45–49)

| Sprint | Name | Version | Delivers |
|--------|------|---------|----------|
| 45 | CONTEXT.md Sweep | 3.25.0 | CONTEXT.md for hub, checkpoints, templates, dashboard, export, inference, plugins |
| 46 | Docs Complete | 3.26.0 | mkdocs pages for export, predict, plugin, template, dashboard, resume commands |
| 47 | Training Notifications | 3.27.0 | Slack/email/desktop alert on training finish; `--notify` flag |
| 48 | Hyperparameter Sweep | 3.28.0 | `xlmtec sweep config.yaml` — Optuna grid/random search over lr/batch/r |
| 49 | CI Hardening | 3.29.0 | Python 3.10/3.11/3.12 matrix, coverage badge, ruff + mypy in CI |