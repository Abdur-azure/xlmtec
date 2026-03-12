"""
tests/test_inference.py
~~~~~~~~~~~~~~~~~~~~~~~~
Tests for batch inference — DataLoader, PredictionWriter, and CLI logic.
All model I/O mocked — no torch/transformers needed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from xlmtec.inference.data_loader import DataLoader
from xlmtec.inference.writer import PredictionRecord, PredictionWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


class TestDataLoader:
    def test_loads_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        _write_jsonl(f, [{"text": "hello"}, {"text": "world"}])
        records = DataLoader(f).load()
        assert len(records) == 2
        assert records[0].text == "hello"
        assert records[1].text == "world"

    def test_loads_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        _write_csv(f, [{"text": "foo"}, {"text": "bar"}])
        records = DataLoader(f).load()
        assert len(records) == 2
        assert records[0].text == "foo"

    def test_auto_detects_text_column(self, tmp_path):
        f = tmp_path / "data.jsonl"
        _write_jsonl(f, [{"text": "hello", "label": 1}])
        records = DataLoader(f).load()
        assert records[0].text == "hello"

    def test_auto_detects_input_column(self, tmp_path):
        f = tmp_path / "data.jsonl"
        _write_jsonl(f, [{"input": "hi there"}])
        records = DataLoader(f).load()
        assert records[0].text == "hi there"

    def test_explicit_text_column(self, tmp_path):
        f = tmp_path / "data.jsonl"
        _write_jsonl(f, [{"body": "test text"}])
        records = DataLoader(f, text_column="body").load()
        assert records[0].text == "test text"

    def test_wrong_column_raises(self, tmp_path):
        f = tmp_path / "data.jsonl"
        _write_jsonl(f, [{"text": "hello"}])
        with pytest.raises(ValueError, match="not found"):
            DataLoader(f, text_column="nonexistent").load()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataLoader(tmp_path / "missing.jsonl").load()

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            DataLoader(f).load()

    def test_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported"):
            DataLoader(f).load()

    def test_indices_are_sequential(self, tmp_path):
        f = tmp_path / "data.jsonl"
        _write_jsonl(f, [{"text": f"row{i}"} for i in range(5)])
        records = DataLoader(f).load()
        assert [r.index for r in records] == list(range(5))

    def test_source_row_preserved(self, tmp_path):
        f = tmp_path / "data.jsonl"
        _write_jsonl(f, [{"text": "hello", "label": "positive"}])
        records = DataLoader(f).load()
        assert records[0].source["label"] == "positive"

    def test_invalid_json_raises(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text("{bad json}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            DataLoader(f).load()


# ---------------------------------------------------------------------------
# PredictionWriter
# ---------------------------------------------------------------------------


class TestPredictionWriter:
    def _records(self, n=3) -> list[PredictionRecord]:
        return [
            PredictionRecord(
                index=i,
                input_text=f"input {i}",
                prediction=f"pred {i}",
                source={"text": f"input {i}", "id": i},
            )
            for i in range(n)
        ]

    def test_writes_jsonl(self, tmp_path):
        out = tmp_path / "out.jsonl"
        writer = PredictionWriter(out, fmt="jsonl")
        n = writer.write(self._records())
        assert n == 3
        lines = [json.loads(l) for l in out.read_text().splitlines() if l]
        assert len(lines) == 3
        assert lines[0]["prediction"] == "pred 0"

    def test_writes_csv(self, tmp_path):
        out = tmp_path / "out.csv"
        writer = PredictionWriter(out, fmt="csv")
        writer.write(self._records())
        rows = list(csv.DictReader(out.open()))
        assert len(rows) == 3
        assert rows[1]["prediction"] == "pred 1"

    def test_auto_detects_csv_from_extension(self, tmp_path):
        out = tmp_path / "out.csv"
        writer = PredictionWriter(out)
        assert writer.fmt == "csv"

    def test_auto_detects_jsonl_by_default(self, tmp_path):
        out = tmp_path / "out.jsonl"
        writer = PredictionWriter(out)
        assert writer.fmt == "jsonl"

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "out.jsonl"
        PredictionWriter(out).write(self._records(1))
        assert out.exists()

    def test_source_fields_preserved_in_jsonl(self, tmp_path):
        out = tmp_path / "out.jsonl"
        records = [PredictionRecord(0, "input", "pred", {"text": "input", "label": "pos"})]
        PredictionWriter(out, fmt="jsonl").write(records)
        row = json.loads(out.read_text().splitlines()[0])
        assert row["label"] == "pos"
        assert row["prediction"] == "pred"

    def test_invalid_format_raises(self, tmp_path):
        out = tmp_path / "out.parquet"
        with pytest.raises(ValueError, match="Unsupported"):
            PredictionWriter(out, fmt="parquet").write([])


# ---------------------------------------------------------------------------
# run_predict CLI logic
# ---------------------------------------------------------------------------


class TestRunPredict:
    def _setup(self, tmp_path) -> tuple[Path, Path, Path]:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        data_file = tmp_path / "data.jsonl"
        _write_jsonl(data_file, [{"text": f"sample {i}"} for i in range(5)])
        output_file = tmp_path / "predictions.jsonl"
        return model_dir, data_file, output_file

    def test_dry_run_returns_0(self, tmp_path):
        from xlmtec.cli.commands.predict import run_predict

        model, data, out = self._setup(tmp_path)
        code = run_predict(model, data, out, "jsonl", None, 8, 128, 1.0, "cpu", dry_run=True)
        assert code == 0

    def test_missing_model_returns_1(self, tmp_path):
        from xlmtec.cli.commands.predict import run_predict

        _, data, out = self._setup(tmp_path)
        code = run_predict(
            tmp_path / "no_model", data, out, "jsonl", None, 8, 128, 1.0, "cpu", dry_run=True
        )
        assert code == 1

    def test_missing_data_returns_1(self, tmp_path):
        from xlmtec.cli.commands.predict import run_predict

        model, _, out = self._setup(tmp_path)
        code = run_predict(
            model, tmp_path / "no_data.jsonl", out, "jsonl", None, 8, 128, 1.0, "cpu", dry_run=True
        )
        assert code == 1

    def test_invalid_format_returns_1(self, tmp_path):
        from xlmtec.cli.commands.predict import run_predict

        model, data, out = self._setup(tmp_path)
        code = run_predict(model, data, out, "parquet", None, 8, 128, 1.0, "cpu", dry_run=True)
        assert code == 1

    def test_csv_input_dry_run(self, tmp_path):
        from xlmtec.cli.commands.predict import run_predict

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        data_file = tmp_path / "data.csv"
        _write_csv(data_file, [{"text": f"row {i}"} for i in range(3)])
        out = tmp_path / "predictions.csv"
        code = run_predict(model_dir, data_file, out, "csv", None, 8, 128, 1.0, "cpu", dry_run=True)
        assert code == 0
