"""UploadScreen — form for `xlmtec upload`."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.validation import Length
from textual.widgets import Button, Footer, Header, Input, Label, Switch


class UploadScreen(Screen):
    """Form screen for `xlmtec upload`.

    Collects: model path, repo_id, HF token (masked), private toggle,
    merge-adapter toggle, base model (shown only when merge-adapter is on),
    optional commit message.
    Submits → RunningScreen.
    """

    BINDINGS = [
        Binding("escape", "go_home", "Home", show=True),
        Binding("ctrl+s", "submit", "Submit", show=True),
    ]

    DEFAULT_CSS = """
    UploadScreen {
        background: $background;
    }

    UploadScreen .form-container {
        padding: 1 4;
        height: 1fr;
    }

    UploadScreen .form-title {
        color: $accent;
        text-style: bold;
        height: 3;
        content-align: left middle;
        padding: 0 0 1 0;
    }

    UploadScreen .field-label {
        color: $text-muted;
        text-style: bold;
        height: 1;
        margin: 1 0 0 0;
    }

    UploadScreen .field-hint {
        color: $text-muted;
        height: 1;
        margin: 0 0 0 1;
    }

    UploadScreen .toggle-row {
        height: 3;
        layout: horizontal;
        align: left middle;
        margin: 1 0 0 0;
    }

    UploadScreen .toggle-label {
        color: $text;
        width: 28;
        content-align: left middle;
    }

    UploadScreen .form-actions {
        height: 5;
        align: left middle;
        layout: horizontal;
        padding: 1 0;
    }

    UploadScreen Button {
        margin: 0 2 0 0;
        min-width: 16;
    }

    UploadScreen .validation-error {
        color: $error;
        height: 1;
    }

    UploadScreen .hidden {
        display: none;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with ScrollableContainer(classes="form-container"):
            yield Label("☁️  Upload — push model to HuggingFace Hub", classes="form-title")

            yield Label("Model / adapter path *", classes="field-label")
            yield Label("Local directory of the saved model or adapter", classes="field-hint")
            yield Input(
                placeholder="./outputs/gpt2_lora",
                id="input-model-path",
                validators=[Length(minimum=1)],
            )

            yield Label("Repository ID *", classes="field-label")
            yield Label("e.g. username/my-model", classes="field-hint")
            yield Input(
                placeholder="username/my-model",
                id="input-repo-id",
                validators=[Length(minimum=3)],
            )

            yield Label("HuggingFace token *", classes="field-label")
            yield Label(
                "Stored locally only — never logged. Set HF_TOKEN env var to skip.",
                classes="field-hint",
            )
            yield Input(
                placeholder="hf_...",
                id="input-token",
                password=True,
                validators=[Length(minimum=1)],
            )

            yield Label("Commit message", classes="field-label")
            yield Input(
                placeholder="Upload fine-tuned model",
                value="Upload fine-tuned model",
                id="input-message",
            )

            with Horizontal(classes="toggle-row"):
                yield Label("Private repository", classes="toggle-label")
                yield Switch(value=False, id="switch-private", animate=False)

            with Horizontal(classes="toggle-row"):
                yield Label("Merge LoRA adapter first", classes="toggle-label")
                yield Switch(value=False, id="switch-merge", animate=False)

            # Base model input — only shown when merge is toggled on
            yield Label("Base model name *", classes="field-label hidden", id="label-base")
            yield Input(
                placeholder="gpt2",
                id="input-base",
                classes="hidden",
            )

            yield Label("", classes="validation-error", id="validation-msg")

            with Horizontal(classes="form-actions"):
                yield Button("☁️  Upload", variant="primary", id="btn-submit")
                yield Button("← Back", variant="default", id="btn-back")

        yield Footer()

    # ── Toggle merge switch shows/hides base model field ─────────────────

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "switch-merge":
            hidden = not event.value
            self.query_one("#label-base", Label).set_class(hidden, "hidden")
            self.query_one("#input-base", Input).set_class(hidden, "hidden")

    # ── Handlers ──────────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-submit":
            self.action_submit()
        elif event.button.id == "btn-back":
            self.action_go_home()

    def action_go_home(self) -> None:
        self.app.switch_screen("home")

    def action_submit(self) -> None:
        model_path = self.query_one("#input-model-path", Input).value.strip()
        repo_id = self.query_one("#input-repo-id", Input).value.strip()
        token = self.query_one("#input-token", Input).value.strip()
        message = self.query_one("#input-message", Input).value.strip()
        private = self.query_one("#switch-private", Switch).value
        merge = self.query_one("#switch-merge", Switch).value
        base = self.query_one("#input-base", Input).value.strip()

        errors = []
        if not model_path:
            errors.append("Model path is required.")
        if not repo_id or "/" not in repo_id:
            errors.append("Repo ID must be in format username/repo-name.")
        if not token:
            errors.append("HF token is required (or set HF_TOKEN env var).")
        if merge and not base:
            errors.append("Base model is required when merging adapter.")

        if errors:
            self.query_one("#validation-msg", Label).update(
                "[red]" + "  •  ".join(errors) + "[/red]"
            )
            return

        command = [
            "xlmtec",
            "upload",
            model_path,
            repo_id,
            "--token",
            token,
            "--message",
            message or "Upload fine-tuned model",
        ]
        if private:
            command.append("--private")
        if merge:
            command += ["--merge-adapter", "--base-model", base]

        from xlmtec.tui.screens.running import RunningScreen

        self.app.switch_screen(
            RunningScreen(
                command=command,
                title=f"Upload  →  {repo_id}",
                subtitle=f"{'private' if private else 'public'}"
                + (f"  merge={base}" if merge else ""),
            )
        )
