"""RecommendScreen — form for `xlmtec recommend`."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.validation import Length
from textual.widgets import Button, Footer, Header, Input, Label


class RecommendScreen(Screen):
    """Form screen for the `xlmtec recommend` command.

    Collects: model name, optional output path.
    Submits → RunningScreen which streams the recommendation output.
    """

    BINDINGS = [
        Binding("escape", "go_home", "Home", show=True),
        Binding("ctrl+s", "submit", "Submit", show=True),
    ]

    DEFAULT_CSS = """
    RecommendScreen {
        background: $background;
    }

    RecommendScreen .form-container {
        padding: 2 4;
        height: 1fr;
    }

    RecommendScreen .form-title {
        color: $accent;
        text-style: bold;
        height: 3;
        content-align: left middle;
        padding: 0 0 1 0;
    }

    RecommendScreen .field-label {
        color: $text-muted;
        text-style: bold;
        height: 1;
        margin: 1 0 0 0;
    }

    RecommendScreen .field-hint {
        color: $text-muted;
        height: 1;
        margin: 0 0 0 1;
    }

    RecommendScreen .form-actions {
        height: 5;
        align: left middle;
        layout: horizontal;
        padding: 1 0;
    }

    RecommendScreen Button {
        margin: 0 2 0 0;
        min-width: 18;
    }

    RecommendScreen .validation-error {
        color: $error;
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(classes="form-container"):
            yield Label(
                "💡  Recommend — get the optimal config for your hardware", classes="form-title"
            )

            yield Label("Model name or path *", classes="field-label")
            yield Label("e.g. gpt2, meta-llama/Llama-3.2-1B", classes="field-hint")
            yield Input(
                placeholder="gpt2",
                id="input-model",
                validators=[Length(minimum=1)],
            )

            yield Label("Save config to file  (optional)", classes="field-label")
            yield Label("Leave blank to print to terminal only", classes="field-hint")
            yield Input(
                placeholder="./my_config.yaml",
                id="input-output",
            )

            yield Label("", classes="validation-error", id="validation-msg")

            with Horizontal(classes="form-actions"):
                yield Button("💡  Get Recommendation", variant="primary", id="btn-submit")
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
        output = self.query_one("#input-output", Input).value.strip()

        if not model:
            self.query_one("#validation-msg", Label).update("[red]Model name is required.[/red]")
            return

        command = ["xlmtec", "recommend", model]
        if output:
            command += ["--output", output]

        from xlmtec.tui.screens.running import RunningScreen

        self.app.switch_screen(
            RunningScreen(
                command=command,
                title=f"Recommend  —  {model}",
                subtitle="Analysing model size and VRAM…",
            )
        )
