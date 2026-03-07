"""RunningScreen — live command execution with streaming log output.

Runs the CLI command in a background thread via Textual's @work decorator.
Output lines are streamed to LogPanel in real time.
Ctrl+C / Q cancels the running worker and returns to home.
"""

import subprocess
import time
from typing import List

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label
from textual.worker import Worker, WorkerState

from xlmtec.tui.widgets.log_panel import LogPanel


class RunningScreen(Screen):
    """Executes a CLI command and streams its output to a live log panel.

    Args:
        command:     List of command tokens to execute (e.g. ["xlmtec", "train", ...]).
        title:       Display title shown in the header area.
        subtitle:    Short subtitle line below the title.
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel", "Cancel", show=True, priority=True),
        Binding("q", "cancel", "Cancel", show=True),
    ]

    DEFAULT_CSS = """
    RunningScreen {
        background: $background;
    }

    RunningScreen .running-header {
        height: 3;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $surface-lighten-2;
        layout: horizontal;
    }

    RunningScreen .running-title {
        color: $accent;
        text-style: bold;
        width: 1fr;
    }

    RunningScreen .elapsed-label {
        color: $text-muted;
        text-align: right;
        width: 20;
    }

    RunningScreen .status-bar {
        height: 3;
        padding: 0 2;
        background: $surface;
        border-top: solid $surface-lighten-2;
        layout: horizontal;
        align: left middle;
    }

    RunningScreen .status-text {
        color: $text-muted;
        width: 1fr;
    }

    RunningScreen Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        command: List[str],
        title: str = "Running",
        subtitle: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._command = command
        self._title = title
        self._subtitle = subtitle
        self._start_time: float = 0.0
        self._result_data: dict = {}
        self._success: bool = False

    # ── Compose ──────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(classes="running-header"):
            yield Label(f"⚙  {self._title}", classes="running-title")
            yield Label("00:00", classes="elapsed-label", id="elapsed-label")
        yield LogPanel(id="running-log")
        with Horizontal(classes="status-bar"):
            yield Label("Running…", classes="status-text", id="status-text")
            yield Button("Cancel", variant="error", id="btn-cancel")
        yield Footer()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def on_mount(self) -> None:
        self._start_time = time.monotonic()
        self.set_interval(1.0, self._tick_elapsed)
        self._run_command()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _tick_elapsed(self) -> None:
        elapsed = int(time.monotonic() - self._start_time)
        m, s = divmod(elapsed, 60)
        self.query_one("#elapsed-label", Label).update(f"{m:02d}:{s:02d}")

    def _log(self, text: str) -> None:
        self.query_one(LogPanel).write_line(text)

    def _set_status(self, text: str) -> None:
        self.query_one("#status-text", Label).update(text)

    # ── Worker ────────────────────────────────────────────────────────────

    @work(thread=True, exclusive=True, name="command-worker")
    def _run_command(self) -> None:
        """Run the CLI command in a background thread, streaming output lines."""
        self._log(f"[bold cyan]$ {' '.join(self._command)}[/bold cyan]\n")

        try:
            proc = subprocess.Popen(
                self._command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert proc.stdout is not None
            for line in proc.stdout:
                self._log(line.rstrip())

            proc.wait()
            self._success = proc.returncode == 0
            self._result_data = {
                "Exit code": str(proc.returncode),
                "Status": "✅ Success" if self._success else "❌ Failed",
                "Duration": self._elapsed_str(),
                "Command": " ".join(self._command[:3]) + ("…" if len(self._command) > 3 else ""),
            }

        except FileNotFoundError:
            self._log(f"[red]Error: command not found: {self._command[0]}[/red]")
            self._success = False
            self._result_data = {
                "Status": "❌ Failed",
                "Error": f"Command not found: {self._command[0]}",
            }
        except Exception as exc:
            self._log(f"[red]Unexpected error: {exc}[/red]")
            self._success = False
            self._result_data = {"Status": "❌ Failed", "Error": str(exc)}

    def _elapsed_str(self) -> str:
        elapsed = int(time.monotonic() - self._start_time)
        m, s = divmod(elapsed, 60)
        return f"{m:02d}:{s:02d}"

    # ── Worker events ─────────────────────────────────────────────────────

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.SUCCESS:
            self.app.call_from_thread(self._on_command_finished)
        elif event.state == WorkerState.CANCELLED:
            self.app.call_from_thread(self._on_cancelled)
        elif event.state == WorkerState.ERROR:
            self.app.call_from_thread(self._on_error, str(event.worker.error))

    def _on_command_finished(self) -> None:
        self._set_status("Done — pushing result screen…")
        from xlmtec.tui.screens.result import ResultScreen
        self.app.switch_screen(
            ResultScreen(
                success=self._success,
                metrics=self._result_data,
                log_lines=[],
            )
        )

    def _on_cancelled(self) -> None:
        self._log("\n[yellow]Command cancelled.[/yellow]")
        self._set_status("Cancelled")
        self.query_one("#btn-cancel", Button).disabled = True

    def _on_error(self, error: str) -> None:
        self._log(f"\n[red]Worker error: {error}[/red]")
        self._set_status("Error")

    # ── Button / key handlers ─────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.action_cancel()

    def action_cancel(self) -> None:
        workers = self.app.workers
        for w in workers:
            w.cancel()
        self.app.switch_screen("home")
