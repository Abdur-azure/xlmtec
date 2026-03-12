"""BenchmarkScreen — form for `xlmtec evaluate benchmark`."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.validation import Length
from textual.widgets import Button, Checkbox, Footer, Header, Input, Label

_METRIC_OPTIONS = [
    ("rouge1", "ROUGE-1"),
    ("rouge2", "ROUGE-2"),
    ("rougeL", "ROUGE-L"),
    ("bleu", "BLEU"),
    ("perplexity", "Perplexity"),
]


class BenchmarkScreen(Screen):
    """Form screen for `xlmtec evaluate benchmark`.

    Collects: base model, fine-tuned model path, dataset, metrics,
    max-samples, optional report path.
    Submits → RunningScreen.
    """

    BINDINGS = [
        Binding("escape", "go_home", "Home", show=True),
        Binding("ctrl+s", "submit", "Submit", show=True),
    ]

    DEFAULT_CSS = """
    BenchmarkScreen {
        background: $background;
    }

    BenchmarkScreen .form-container {
        padding: 1 4;
        height: 1fr;
    }

    BenchmarkScreen .form-title {
        color: $accent;
        text-style: bold;
        height: 3;
        content-align: left middle;
        padding: 0 0 1 0;
    }

    BenchmarkScreen .field-label {
        color: $text-muted;
        text-style: bold;
        height: 1;
        margin: 1 0 0 0;
    }

    BenchmarkScreen .field-hint {
        color: $text-muted;
        height: 1;
        margin: 0 0 0 1;
    }

    BenchmarkScreen .metrics-group {
        layout: grid;
        grid-size: 3 2;
        grid-gutter: 0 2;
        height: 4;
        margin: 0 0 0 1;
    }

    BenchmarkScreen .form-actions {
        height: 5;
        align: left middle;
        layout: horizontal;
        padding: 1 0;
    }

    BenchmarkScreen Button {
        margin: 0 2 0 0;
        min-width: 16;
    }

    BenchmarkScreen .validation-error {
        color: $error;
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with ScrollableContainer(classes="form-container"):
            yield Label("⚡  Benchmark — compare base vs fine-tuned", classes="form-title")

            yield Label("Base model name or path *", classes="field-label")
            yield Label("e.g. gpt2, meta-llama/Llama-3.2-1B", classes="field-hint")
            yield Input(
                placeholder="gpt2",
                id="input-base",
                validators=[Length(minimum=1)],
            )

            yield Label("Fine-tuned model path *", classes="field-label")
            yield Label("Path to saved adapter or full fine-tuned model", classes="field-hint")
            yield Input(
                placeholder="./outputs/gpt2_lora",
                id="input-finetuned",
                validators=[Length(minimum=1)],
            )

            yield Label("Dataset path *", classes="field-label")
            yield Label("Local .jsonl / .json / .csv test file", classes="field-hint")
            yield Input(
                placeholder="./data/sample.jsonl",
                id="input-dataset",
                validators=[Length(minimum=1)],
            )

            yield Label("Metrics  (select one or more)", classes="field-label")
            with Horizontal(classes="metrics-group"):
                for metric_id, metric_label in _METRIC_OPTIONS:
                    default = metric_id in ("rouge1", "rouge2", "rougeL")
                    yield Checkbox(
                        metric_label,
                        value=default,
                        id=f"chk-{metric_id}",
                    )

            yield Label("Max samples", classes="field-label")
            yield Input(
                placeholder="100",
                value="100",
                id="input-max-samples",
            )

            yield Label("Save report to  (optional)", classes="field-label")
            yield Input(
                placeholder="./benchmark_report.md",
                id="input-report",
            )

            yield Label("", classes="validation-error", id="validation-msg")

            with Horizontal(classes="form-actions"):
                yield Button("⚡  Run Benchmark", variant="primary", id="btn-submit")
                yield Button("← Back", variant="default", id="btn-back")

        yield Footer()

    # ── Handlers ──────────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-submit":
            self.action_submit()
        elif event.button.id == "btn-back":
            self.action_go_home()

    def action_go_home(self) -> None:
        self.app.switch_screen("home")

    def action_submit(self) -> None:
        base = self.query_one("#input-base", Input).value.strip()
        finetuned = self.query_one("#input-finetuned", Input).value.strip()
        dataset = self.query_one("#input-dataset", Input).value.strip()
        max_samples = self.query_one("#input-max-samples", Input).value.strip()
        report = self.query_one("#input-report", Input).value.strip()

        errors = []
        if not base:
            errors.append("Base model is required.")
        if not finetuned:
            errors.append("Fine-tuned path is required.")
        if not dataset:
            errors.append("Dataset path is required.")

        selected_metrics = [
            metric_id
            for metric_id, _ in _METRIC_OPTIONS
            if self.query_one(f"#chk-{metric_id}", Checkbox).value
        ]
        if not selected_metrics:
            errors.append("Select at least one metric.")

        if errors:
            self.query_one("#validation-msg", Label).update(
                "[red]" + "  •  ".join(errors) + "[/red]"
            )
            return

        command = [
            "xlmtec",
            "evaluate",
            "benchmark",
            "--base",
            base,
            "--finetuned",
            finetuned,
            "--dataset",
            dataset,
        ]
        for m in selected_metrics:
            command += ["--metric", m]
        if max_samples:
            command += ["--max-samples", max_samples]
        if report:
            command += ["--report", report]

        from xlmtec.tui.screens.running import RunningScreen

        self.app.switch_screen(
            RunningScreen(
                command=command,
                title=f"Benchmark  {base}  vs  {finetuned}",
                subtitle=f"dataset={dataset}",
            )
        )
