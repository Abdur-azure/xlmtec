"""
FinetuneApp — root Textual application for finetune-cli TUI.

Entry point: `finetune-cli tui`

Global keybindings (all screens):
    q           quit the application
    h / escape  return to home screen
    ctrl+c      force quit
"""

from textual.app import App, ComposeResult
from textual.binding import Binding

from finetune_cli.tui.screens.home import HomeScreen


class FinetuneApp(App):
    """Root Textual App — manages screen stack and global bindings."""

    TITLE = "finetune-cli"
    SUB_TITLE = "LLM Fine-Tuning Toolkit"

    # All screens registered here so they can be pushed by name.
    # Sprint 26-28 will add: train, evaluate, benchmark, merge, upload, recommend,
    # running, result screens.
    SCREENS = {
        "home": HomeScreen,
    }

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True, priority=True),
        Binding("h", "go_home", "Home", show=True),
        Binding("escape", "go_home", "Home", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
    ]

    CSS = """
    /* ── Global theme ──────────────────────────────────────────────────── */
    Screen {
        background: $background;
    }

    /* ── Header ─────────────────────────────────────────────────────────── */
    Header {
        background: $primary-darken-2;
        color: $text;
        text-style: bold;
    }

    /* ── Footer ─────────────────────────────────────────────────────────── */
    Footer {
        background: $primary-darken-3;
        color: $text-muted;
    }

    /* ── Notifications ───────────────────────────────────────────────────── */
    Toast {
        background: $surface;
        border: tall $accent;
        color: $text;
    }
    """

    async def on_mount(self) -> None:
        """Push the home screen as the initial screen on app start."""
        await self.push_screen("home")

    def action_go_home(self) -> None:
        """Return to home screen — switch replaces current screen cleanly."""
        self.switch_screen("home")

    def action_quit(self) -> None:
        self.exit()


def run() -> None:
    """Launch the TUI app. Called by `finetune-cli tui`."""
    FinetuneApp().run()