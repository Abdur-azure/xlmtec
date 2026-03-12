"""
xlmtec.inference.writer
~~~~~~~~~~~~~~~~~~~~~~~~
Writes batch inference predictions to JSONL or CSV.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PredictionRecord:
    """A single prediction result."""

    index: int
    input_text: str
    prediction: str
    source: dict  # original input row


class PredictionWriter:
    """Write predictions to JSONL or CSV output file."""

    def __init__(self, path: Path, fmt: str | None = None) -> None:
        self.path = Path(path)
        # Auto-detect format from extension if not specified
        self.fmt = fmt or self._detect_fmt()

    def write(self, records: list[PredictionRecord]) -> int:
        """Write all prediction records to the output file.

        Returns:
            Number of records written.

        Raises:
            ValueError: If format is unsupported.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.fmt == "jsonl":
            self._write_jsonl(records)
        elif self.fmt == "csv":
            self._write_csv(records)
        else:
            raise ValueError(f"Unsupported output format {self.fmt!r}. Use 'jsonl' or 'csv'.")
        return len(records)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_fmt(self) -> str:
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            return "csv"
        return "jsonl"  # default

    def _write_jsonl(self, records: list[PredictionRecord]) -> None:
        lines = []
        for r in records:
            row = dict(r.source)
            row["prediction"] = r.prediction
            row["_index"] = r.index
            lines.append(json.dumps(row, ensure_ascii=False))
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_csv(self, records: list[PredictionRecord]) -> None:
        if not records:
            self.path.write_text("", encoding="utf-8")
            return

        # Build fieldnames: original columns + prediction
        base_fields = list(records[0].source.keys())
        extra_fields = [f for f in ["prediction", "_index"] if f not in base_fields]
        fieldnames = base_fields + extra_fields

        with self.path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in records:
                row = dict(r.source)
                row["prediction"] = r.prediction
                row["_index"] = r.index
                writer.writerow(row)
