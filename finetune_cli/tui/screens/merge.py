"""MergeScreen — form for `finetune-cli merge`."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.validation import Length
from textual.widgets import Button, Footer, Header, Input, Label, Select

_DTYPE_OPTIONS = [
    ("float32  (default, safest)", "float32"),
    ("float16  (half precision, saves ~50% disk)", "float16"),
    ("bfloat16 (brain float, Ampere+ GPUs)", "bfloat16"),
]


class MergeScreen(Screen):
    """Form screen for `finetune-cli merge`.

    Collects: adapter directory, base model name, output directory, dtype.
    Submits → RunningScreen with the built CLI command.
    """

    BINDINGS = [
        Binding("escape", "go_home", "Home", show=True),
        Binding("ctrl+s", "submit", "Submit", show=True),
    ]

    DEFAULT_CSS = """
    MergeScreen {
        background: $background;
    }

    MergeScreen .form-container {
        padding: 2 4;
        height: 1fr;
    }

    MergeScreen .form-title {
        color: $accent;
        text-style: bold;
        height: 3;
        content-align: left middle;
        padding: 0 0 1 0;
    }

    MergeScreen .field-label {
        color: $text-muted;
        text-style: bold;
        height: 1;
        margin: 1 0 0 0;
    }

    MergeScreen .field-hint {
        color: $text-muted;
        height: 1;
        margin: 0 0 0 1;
    }

    MergeScreen .form-actions {
        height: 5;
        align: left middle;
        layout: horizontal;
        padding: 1 0;
    }

    MergeScreen Button {
        margin: 0 2 0 0;
        min-width: 16;
    }

    MergeScreen .validation-error {
        color: $error;
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with ScrollableContainer(classes="form-container"):
            yield Label("🔀  Merge — fuse LoRA adapter into standalone model", classes="form-title")

            yield Label("Adapter directory *", classes="field-label")
            yield Label("Path to the saved LoRA adapter", classes="field-hint")
            yield Input(
                placeholder="./outputs/gpt2_lora",
                id="input-adapter",
                validators=[Length(minimum=1)],
            )

            yield Label("Base model name or path *", classes="field-label")
            yield Label("Must match the model used during training", classes="field-hint")
            yield Input(
                placeholder="gpt2",
                id="input-base",
                validators=[Length(minimum=1)],
            )

            yield Label("Output directory *", classes="field-label")
            yield Label("Where the merged standalone model will be saved", classes="field-hint")
            yield Input(
                placeholder="./outputs/gpt2_merged",
                id="input-output",
                validators=[Length(minimum=1)],
            )

            yield Label("Dtype", classes="field-label")
            yield Select(
                options=_DTYPE_OPTIONS,
                id="select-dtype",
                allow_blank=False,
            )

            yield Label("", classes="validation-error", id="validation-msg")

            with Horizontal(classes="form-actions"):
                yield Button("🔀  Run Merge", variant="primary", id="btn-submit")
                yield Button("← Back", variant="default", id="btn-back")

        yield Footer()

    async def on_mount(self) -> None:
        """Set Select default after widget is fully mounted."""
        self.query_one("#select-dtype", Select).value = "float32"

    # ── Handlers ──────────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-submit":
            self.action_submit()
        elif event.button.id == "btn-back":
            self.action_go_home()

    def action_go_home(self) -> None:
        self.app.switch_screen("home")

    def action_submit(self) -> None:
        adapter = self.query_one("#input-adapter", Input).value.strip()
        base = self.query_one("#input-base", Input).value.strip()
        output = self.query_one("#input-output", Input).value.strip()
        dtype = self.query_one("#select-dtype", Select).value

        errors = []
        if not adapter:
            errors.append("Adapter directory is required.")
        if not base:
            errors.append("Base model is required.")
        if not output:
            errors.append("Output directory is required.")

        if errors:
            self.query_one("#validation-msg", Label).update(
                "[red]" + "  •  ".join(errors) + "[/red]"
            )
            return

        command = [
            "finetune-cli", "merge",
            adapter,
            output,
            "--base-model", base,
            "--dtype", str(dtype),
        ]

        from finetune_cli.tui.screens.running import RunningScreen
        self.app.switch_screen(
            RunningScreen(
                command=command,
                title=f"Merge  {adapter}  →  {output}",
                subtitle=f"base={base}  dtype={dtype}",
            )
        )
