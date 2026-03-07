"""CommandCard widget — a styled, focusable card for the home screen."""

from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Label


class CommandCard(Widget):
    """Styled card representing a single CLI command.

    Posts a ``CommandCard.Selected`` message when clicked or activated
    with the Enter key.

    Args:
        command_id: Internal identifier string (e.g. "train").
        label:      Bold title text shown at top of card.
        description: Short description line below the label.
        icon:       Emoji or symbol shown beside the label.
    """

    can_focus = True

    DEFAULT_CSS = """
    CommandCard {
        width: 1fr;
        height: 9;
        border: tall $surface-lighten-2;
        background: $surface;
        padding: 1 2;
        layout: vertical;
        content-align: left middle;
    }

    CommandCard:hover {
        border: tall $accent;
        background: $surface-lighten-1;
    }

    CommandCard:focus {
        border: tall $accent;
        background: $surface-lighten-1;
    }

    CommandCard .card-header {
        color: $text;
        text-style: bold;
        height: 2;
        width: 100%;
    }

    CommandCard .card-description {
        color: $text-muted;
        height: 3;
        width: 100%;
    }
    """

    class Selected(Message):
        """Posted when this card is activated by click or Enter."""

        def __init__(self, command_id: str) -> None:
            super().__init__()
            self.command_id = command_id

    def __init__(
        self,
        command_id: str,
        label: str,
        description: str,
        icon: str = "▶",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.command_id = command_id
        self._label = label
        self._description = description
        self._icon = icon

    def compose(self) -> ComposeResult:
        yield Label(f"{self._icon}  {self._label}", classes="card-header")
        yield Label(self._description, classes="card-description")

    def on_click(self) -> None:
        self.post_message(self.Selected(self.command_id))

    def on_key(self, event) -> None:
        if event.key == "enter":
            event.stop()
            self.post_message(self.Selected(self.command_id))
