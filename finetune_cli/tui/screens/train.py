"""TrainScreen — form for launching a fine-tuning run."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.validation import Length, Number
from textual.widgets import Button, Footer, Header, Input, Label, Select

# Training method options — mirrors TrainingMethod enum values
_METHOD_OPTIONS = [
    ("LoRA (recommended)", "lora"),
    ("QLoRA — 4-bit quantised", "qlora"),
    ("Full Fine-Tuning", "full_finetuning"),
    ("Instruction Tuning", "instruction_tuning"),
    ("DPO", "dpo"),
    ("Response Distillation", "vanilla_distillation"),
    ("Feature Distillation", "feature_distillation"),
]


class TrainScreen(Screen):
    """Form screen for the `finetune-cli train` command.

    Collects: model name, training method, dataset path, epochs,
    learning rate, output directory. Submits → RunningScreen.
    """

    BINDINGS = [
        Binding("escape", "go_home", "Home", show=True),
        Binding("ctrl+s", "submit", "Submit", show=True),
    ]

    DEFAULT_CSS = """
    TrainScreen {
        background: $background;
    }

    TrainScreen .form-container {
        padding: 1 4;
        height: 1fr;
    }

    TrainScreen .form-title {
        color: $accent;
        text-style: bold;
        height: 3;
        content-align: left middle;
        padding: 0 0 1 0;
    }

    TrainScreen .field-label {
        color: $text-muted;
        text-style: bold;
        height: 1;
        margin: 1 0 0 0;
    }

    TrainScreen .field-hint {
        color: $text-muted;
        height: 1;
        margin: 0 0 0 1;
    }

    TrainScreen Input {
        margin: 0 0 0 0;
    }

    TrainScreen Select {
        margin: 0 0 0 0;
    }

    TrainScreen .form-actions {
        height: 5;
        align: left middle;
        layout: horizontal;
        padding: 1 0;
    }

    TrainScreen Button {
        margin: 0 2 0 0;
        min-width: 16;
    }

    TrainScreen .validation-error {
        color: $error;
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with ScrollableContainer(classes="form-container"):
            yield Label("🚀  Train — Fine-tune a model", classes="form-title")

            yield Label("Model name or path *", classes="field-label")
            yield Label("e.g. gpt2, meta-llama/Llama-3.2-1B", classes="field-hint")
            yield Input(
                placeholder="gpt2",
                id="input-model",
                validators=[Length(minimum=1)],
            )

            yield Label("Training method *", classes="field-label")
            yield Select(
                options=_METHOD_OPTIONS,
                id="select-method",
                allow_blank=False,
            )

            yield Label("Dataset path *", classes="field-label")
            yield Label("Local .jsonl / .json / .csv file path", classes="field-hint")
            yield Input(
                placeholder="./data/sample.jsonl",
                id="input-dataset",
                validators=[Length(minimum=1)],
            )

            yield Label("Number of epochs", classes="field-label")
            yield Input(
                placeholder="3",
                value="3",
                id="input-epochs",
                validators=[Number(minimum=1, maximum=100)],
            )

            yield Label("Learning rate", classes="field-label")
            yield Input(
                placeholder="2e-4",
                value="2e-4",
                id="input-lr",
            )

            yield Label("Output directory *", classes="field-label")
            yield Input(
                placeholder="./outputs/my_model",
                id="input-output",
                validators=[Length(minimum=1)],
            )

            yield Label("", classes="validation-error", id="validation-msg")

            with Horizontal(classes="form-actions"):
                yield Button("▶  Run Training", variant="primary", id="btn-submit")
                yield Button("← Back", variant="default", id="btn-back")

        yield Footer()

    async def on_mount(self) -> None:
        """Set Select default after widget is fully mounted."""
        self.query_one("#select-method", Select).value = "lora"

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
        method = self.query_one("#select-method", Select).value
        dataset = self.query_one("#input-dataset", Input).value.strip()
        epochs = self.query_one("#input-epochs", Input).value.strip()
        lr = self.query_one("#input-lr", Input).value.strip()
        output = self.query_one("#input-output", Input).value.strip()

        # Basic validation
        errors = []
        if not model:
            errors.append("Model name is required.")
        if not dataset:
            errors.append("Dataset path is required.")
        if not output:
            errors.append("Output directory is required.")
        if not epochs.isdigit() or int(epochs) < 1:
            errors.append("Epochs must be a positive integer.")

        if errors:
            self.query_one("#validation-msg", Label).update(
                "[red]" + "  •  ".join(errors) + "[/red]"
            )
            return

        command = [
            "finetune-cli", "train",
            "--model", model,
            "--dataset", dataset,
            "--method", str(method),
            "--epochs", epochs,
            "--lr", lr,
            "--output", output,
        ]

        from finetune_cli.tui.screens.running import RunningScreen
        self.app.switch_screen(
            RunningScreen(
                command=command,
                title=f"Training  {model}  [{method}]",
                subtitle=f"dataset={dataset}  epochs={epochs}",
            )
        )
