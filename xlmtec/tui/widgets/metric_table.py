"""MetricTable widget — displays key/value result pairs in a DataTable."""

from typing import Any, Dict

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable


class MetricTable(Widget):
    """DataTable widget for displaying result metrics.

    Usage:
        table = self.query_one(MetricTable)
        table.populate({"Train Loss": "0.312", "Epochs": "3", "Steps": "120"})
    """

    DEFAULT_CSS = """
    MetricTable {
        height: auto;
        border: round $surface-lighten-2;
        background: $surface;
        padding: 0 1;
    }

    MetricTable DataTable {
        height: auto;
        background: $surface;
    }
    """

    def compose(self) -> ComposeResult:
        # FIX line 33: explicit type annotation required by mypy [var-annotated]
        table: DataTable = DataTable(show_cursor=False, id="metric-data-table")
        table.add_columns("Metric", "Value")
        yield table

    def populate(self, metrics: Dict[str, Any]) -> None:
        """Fill the table with metric rows, clearing previous data first."""
        table = self.query_one(DataTable)
        table.clear()
        for key, value in metrics.items():
            table.add_row(str(key), str(value))