"""ResultScreen — displays command outcome with a metric table and back button."""

from typing import Any, Dict, List

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label

from xlmtec.tui.widgets.metric_table import MetricTable


class ResultScreen(Screen):
    """Shows the result of a completed command.

    Args:
        success:    Whether the command exited successfully.
        metrics:    Dict of metric name → value to display in the table.
        log_lines:  Optional list of captured log lines (reserved for future use).
    """

    BINDINGS = [
        Binding("h", "go_home", "Home", show=True),
        Binding("escape", "go_home", "Home", show=False),
        Binding("q", "quit_app", "Quit", show=True),
    ]

    DEFAULT_CSS = """
    ResultScreen {
        background: $background;
    }

    ResultScreen .result-banner {
        height: 5;
        content-align: center middle;
        text-align: center;
        text-style: bold;
        width: 100%;
        margin: 1 0;
    }

    ResultScreen .result-success {
        color: $success;
        background: $surface;
        border: tall $success;
    }

    ResultScreen .result-failure {
        color: $error;
        background: $surface;
        border: tall $error;
    }

    ResultScreen .result-body {
        padding: 1 4;
        height: 1fr;
    }

    ResultScreen .result-section-label {
        color: $text-muted;
        text-style: bold;
        margin: 1 0 0 0;
    }

    ResultScreen .result-actions {
        height: 5;
        align: center middle;
        layout: horizontal;
    }

    ResultScreen Button {
        margin: 0 2;
        min-width: 16;
    }
    """

    def __init__(
        self,
        success: bool,
        metrics: Dict[str, Any],
        log_lines: List[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._success = success
        self._metrics = metrics
        self._log_lines = log_lines or []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        banner_class = (
            "result-banner result-success" if self._success else "result-banner result-failure"
        )
        banner_text = (
            "✅  Command completed successfully" if self._success else "❌  Command failed"
        )
        yield Label(banner_text, classes=banner_class)

        with Vertical(classes="result-body"):
            if self._metrics:
                yield Label("Results", classes="result-section-label")
                yield MetricTable(id="result-metrics")

        with Horizontal(classes="result-actions"):
            yield Button("🏠  Home", variant="primary", id="btn-home")
            yield Button("✕  Quit", variant="error", id="btn-quit")

        yield Footer()

    async def on_mount(self) -> None:
        if self._metrics:
            self.query_one(MetricTable).populate(self._metrics)

    # ── Button handlers ───────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-home":
            self.action_go_home()
        elif event.button.id == "btn-quit":
            self.action_quit_app()

    def action_go_home(self) -> None:
        self.app.switch_screen("home")

    def action_quit_app(self) -> None:
        self.app.exit()
