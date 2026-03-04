"""LogPanel widget — scrolling live log output for the RunningScreen."""

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import RichLog
from textual.reactive import reactive


class LogPanel(Widget):
    """Scrolling log panel that streams command output lines.

    Usage:
        panel = self.query_one(LogPanel)
        panel.write_line("[green]Starting...[/green]")
        panel.clear()
    """

    DEFAULT_CSS = """
    LogPanel {
        height: 1fr;
        border: round $surface-lighten-2;
        background: $surface;
        padding: 0 1;
    }

    LogPanel RichLog {
        height: 1fr;
        background: $surface;
        scrollbar-color: $accent;
    }
    """

    auto_scroll: reactive[bool] = reactive(True)

    def compose(self) -> ComposeResult:
        yield RichLog(
            highlight=True,
            markup=True,
            wrap=True,
            id="log-output",
        )

    def write_line(self, text: str) -> None:
        """Append a line of text to the log."""
        log = self.query_one(RichLog)
        log.write(text)
        if self.auto_scroll:
            log.scroll_end(animate=False)

    def clear(self) -> None:
        """Clear all log content."""
        self.query_one(RichLog).clear()