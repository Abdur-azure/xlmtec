"""
xlmtecApp — root Textual application for xlmtec TUI.

Entry point: `xlmtec tui`

Global keybindings (all screens):
    q           quit the application
    h / escape  return to home screen
    ctrl+c      force quit
"""

from pathlib import Path

from textual.app import App
from textual.binding import Binding

from xlmtec.tui.screens.home import HomeScreen


class xlmtecApp(App):
    """Root Textual App — manages screen stack and global bindings."""

    TITLE = "xlmtec"
    SUB_TITLE = "LLM Fine-Tuning Toolkit"

    # External CSS theme — Sprint 28
    CSS_PATH = Path(__file__).parent / "app.css"

    # Screens registered by name for push/switch.
    SCREENS = {
        "home": HomeScreen,
    }

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True, priority=True),
        Binding("h", "go_home", "Home", show=True),
        Binding("escape", "go_home", "Home", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
    ]

    async def on_mount(self) -> None:
        """Push the home screen as the initial screen on app start."""
        await self.push_screen("home")

    def action_go_home(self) -> None:
        """Return to home screen — switch replaces current screen cleanly."""
        self.switch_screen("home")

    def action_quit(self) -> None:
        self.exit()


def run() -> None:
    """Launch the TUI app. Called by `xlmtec tui`."""
    xlmtecApp().run()
