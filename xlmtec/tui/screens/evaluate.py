"""EvaluateScreen — form for `xlmtec evaluate`."""

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


class EvaluateScreen(Screen):
    """Form screen for the `xlmtec evaluate` command.

    Collects: model/checkpoint path, dataset path, metrics (multi-select
    checkboxes), max-samples, optional report output path.
    Submits → RunningScreen with the built CLI command.
    """

    BINDINGS = [
        Binding("escape", "go_home", "Home", show=True),
        Binding("ctrl+s", "submit", "Submit", show=True),
    ]

    DEFAULT_CSS = """
    EvaluateScreen {
        background: $background;
    }

    EvaluateScreen .form-container {
        padding: 1 4;
        height: 1fr;
    }

    EvaluateScreen .form-title {
        color: $accent;
        text-style: bold;
        height: 3;
        content-align: left middle;
        padding: 0 0 1 0;
    }

    EvaluateScreen .field-label {
        color: $text-muted;
        text-style: bold;
        height: 1;
        margin: 1 0 0 0;
    }

    EvaluateScreen .field-hint {
        color: $text-muted;
        height: 1;
        margin: 0 0 0 1;
    }

    EvaluateScreen .metrics-group {
        layout: grid;
        grid-size: 3 2;
        grid-gutter: 0 2;
        height: 4;
        margin: 0 0 0 1;
    }

    EvaluateScreen .form-actions {
        height: 5;
        align: left middle;
        layout: horizontal;
        padding: 1 0;
    }

    EvaluateScreen Button {
        margin: 0 2 0 0;
        min-width: 16;
    }

    EvaluateScreen .validation-error {
        color: $error;
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with ScrollableContainer(classes="form-container"):
            yield Label("📊  Evaluate — score a checkpoint", classes="form-title")

            yield Label("Model / checkpoint path *", classes="field-label")
            yield Label("Path to saved adapter or full model", classes="field-hint")
            yield Input(
                placeholder="./outputs/gpt2_lora",
                id="input-model",
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
                placeholder="./eval_report.md",
                id="input-report",
            )

            yield Label("", classes="validation-error", id="validation-msg")

            with Horizontal(classes="form-actions"):
                yield Button("📊  Run Evaluation", variant="primary", id="btn-submit")
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
        model = self.query_one("#input-model", Input).value.strip()
        dataset = self.query_one("#input-dataset", Input).value.strip()
        max_samples = self.query_one("#input-max-samples", Input).value.strip()
        report = self.query_one("#input-report", Input).value.strip()

        errors = []
        if not model:
            errors.append("Model path is required.")
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

        command = ["xlmtec", "evaluate", model, "--dataset", dataset]
        for m in selected_metrics:
            command += ["--metrics", m]
        if max_samples:
            command += ["--max-samples", max_samples]
        if report:
            command += ["--save-report", report]

        from xlmtec.tui.screens.running import RunningScreen

        self.app.switch_screen(
            RunningScreen(
                command=command,
                title=f"Evaluate  {model}",
                subtitle=f"metrics={','.join(selected_metrics)}",
            )
        )
