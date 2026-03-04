"""HomeScreen -- main menu with 6 command cards in a 2x3 grid."""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Header, Label
from textual.containers import Grid
from textual import events

from finetune_cli.tui.widgets.command_card import CommandCard


_COMMANDS = [
    {"id": "train",     "label": "Train",     "description": "Fine-tune with LoRA, QLoRA, DPO and more",       "icon": "🚀"},
    {"id": "evaluate",  "label": "Evaluate",  "description": "Score a checkpoint (ROUGE, BLEU, Perplexity)",   "icon": "📊"},
    {"id": "benchmark", "label": "Benchmark", "description": "Compare base vs fine-tuned side-by-side",        "icon": "⚡"},
    {"id": "upload",    "label": "Upload",    "description": "Push adapter or merged model to HF Hub",         "icon": "☁️"},
    {"id": "merge",     "label": "Merge",     "description": "Merge LoRA adapter into standalone model",       "icon": "🔀"},
    {"id": "recommend", "label": "Recommend", "description": "Get optimal config for your hardware",           "icon": "💡"},
]

# Grid is 3 columns x 2 rows -- arrow offsets map to that layout.
_GRID_COLS = 3


class HomeScreen(Screen):
    """Main navigation screen -- 6 command cards in a 2x3 grid.

    Keyboard navigation:
        Arrow keys   move focus between cards (wraps at edges)
        Tab          cycle focus forward
        Shift+Tab    cycle focus backward
        Enter        select focused card
        Click        select any card
        Q            quit the application
    """

    DEFAULT_CSS = """
    HomeScreen {
        background: $background;
    }

    HomeScreen .home-title {
        text-align: center;
        color: $accent;
        text-style: bold;
        padding: 1 0;
        width: 100%;
        height: 3;
    }

    HomeScreen .home-subtitle {
        text-align: center;
        color: $text-muted;
        padding: 0 0 1 0;
        width: 100%;
        height: 2;
    }

    HomeScreen .card-grid {
        layout: grid;
        grid-size: 3 2;
        grid-gutter: 1 2;
        padding: 0 3;
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label(
            "finetune-cli  --  LLM Fine-Tuning Toolkit",
            classes="home-title",
        )
        yield Label(
            "Tab / Arrow keys to navigate   Enter or Click to select   Q to quit",
            classes="home-subtitle",
        )
        with Grid(classes="card-grid"):
            for cmd in _COMMANDS:
                yield CommandCard(
                    command_id=cmd["id"],
                    label=cmd["label"],
                    description=cmd["description"],
                    icon=cmd["icon"],
                    id=f"card-{cmd['id']}",
                )
        yield Footer()

    def on_mount(self) -> None:
        """Focus the first card on screen mount."""
        cards = list(self.query(CommandCard))
        if cards:
            cards[0].focus()

    def on_command_card_selected(self, event: CommandCard.Selected) -> None:
        """Route card selection to the appropriate screen.

        Sprint 25: shows a notification stub.
        Sprint 26+ will push real command screens.
        """
        self.app.notify(
            f"[bold]{event.command_id.capitalize()}[/bold] screen coming in Sprint 26",
            title="Coming soon",
            severity="information",
        )

    def on_key(self, event: events.Key) -> None:
        """Arrow key navigation across the 3-column card grid."""
        if event.key not in ("up", "down", "left", "right"):
            return

        cards = list(self.query(CommandCard))
        if not cards:
            return

        focused = self.focused
        if focused not in cards:
            cards[0].focus()
            event.stop()
            return

        idx = cards.index(focused)
        total = len(cards)

        if event.key == "right":
            next_idx = (idx + 1) % total
        elif event.key == "left":
            next_idx = (idx - 1) % total
        elif event.key == "down":
            next_idx = (idx + _GRID_COLS) % total
        else:  # up
            next_idx = (idx - _GRID_COLS) % total

        cards[next_idx].focus()
        event.stop()