"""
xlmtec.inference.data_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reads JSONL and CSV input files for batch inference.
Auto-detects the text column when not specified.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

# Column names tried in order when auto-detecting
_TEXT_COLUMN_CANDIDATES = [
    "text",
    "input",
    "prompt",
    "sentence",
    "content",
    "question",
    "context",
    "document",
    "instruction",
]


@dataclass
class InferenceRecord:
    """A single input record for inference."""

    index: int
    text: str
    source: dict  # original row, preserved in output


class DataLoader:
    """Load input records from JSONL or CSV files."""

    def __init__(self, path: Path, text_column: str | None = None) -> None:
        self.path = Path(path)
        self.text_column = text_column

    def load(self) -> list[InferenceRecord]:
        """Load all records from the input file.

        Returns:
            List of :class:`InferenceRecord` in file order.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the format is unsupported or text column not found.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Input file not found: {self.path}")

        suffix = self.path.suffix.lower()
        if suffix == ".jsonl" or suffix == ".json":
            rows = self._load_jsonl()
        elif suffix == ".csv":
            rows = self._load_csv()
        else:
            raise ValueError(f"Unsupported file format {suffix!r}. Use .jsonl or .csv")

        if not rows:
            raise ValueError(f"Input file is empty: {self.path}")

        col = self.text_column or self._detect_column(rows[0])
        records = []
        for i, row in enumerate(rows):
            if col not in row:
                raise ValueError(
                    f"Column {col!r} not found in row {i}.\n"
                    f"Available columns: {', '.join(row.keys())}\n"
                    f"Use --text-column to specify the correct column."
                )
            records.append(InferenceRecord(index=i, text=str(row[col]), source=row))

        return records

    def detect_column(self) -> str:
        """Return the auto-detected text column name without loading all records."""
        if self.text_column:
            return self.text_column
        rows = self._load_jsonl() if self.path.suffix.lower() != ".csv" else self._load_csv()
        if not rows:
            raise ValueError(f"Input file is empty: {self.path}")
        return self._detect_column(rows[0])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_jsonl(self) -> list[dict]:
        rows = []
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
        return rows

    def _load_csv(self) -> list[dict]:
        rows = []
        with self.path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows

    def _detect_column(self, row: dict) -> str:
        for candidate in _TEXT_COLUMN_CANDIDATES:
            if candidate in row:
                return candidate
        # Fall back to first column
        first = next(iter(row))
        return first
