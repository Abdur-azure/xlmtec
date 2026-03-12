"""
tests/test_dashboard.py
~~~~~~~~~~~~~~~~~~~~~~~~
Tests for evaluation dashboard — RunReader, RunComparator, ComparisonResult.
Pure filesystem using tmp_path — no ML dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xlmtec.dashboard.comparator import RunComparator
from xlmtec.dashboard.reader import RunMetrics, RunReader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(
    tmp_path: Path,
    name: str,
    steps: int = 100,
    epoch: float = 1.0,
    best_metric: float | None = None,
    log_history: list | None = None,
    config: dict | None = None,
) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir()

    state = {
        "global_step": steps,
        "epoch": epoch,
        "best_metric": best_metric,
        "best_metric_key": "eval_loss" if best_metric else "",
        "train_runtime": 120.0,
        "train_samples_per_second": 42.0,
        "log_history": log_history
        or [
            {"step": 50, "epoch": 0.5, "loss": 2.5, "learning_rate": 2e-4},
            {"step": 100, "epoch": 1.0, "eval_loss": 1.8, "eval_rouge1": 0.35},
        ],
    }
    (run_dir / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")

    if config:
        import yaml

        (run_dir / "config.yaml").write_text(yaml.dump(config), encoding="utf-8")

    return run_dir


# ---------------------------------------------------------------------------
# RunReader
# ---------------------------------------------------------------------------


class TestRunReader:
    def test_reads_basic_fields(self, tmp_path):
        run_dir = _make_run(tmp_path, "run1", steps=200, epoch=2.0)
        info = RunReader(run_dir).read()
        assert info.name == "run1"
        assert info.total_steps == 200
        assert info.total_epochs == 2.0

    def test_reads_log_history(self, tmp_path):
        run_dir = _make_run(tmp_path, "run1")
        info = RunReader(run_dir).read()
        assert len(info.history) == 2
        assert isinstance(info.history[0], RunMetrics)

    def test_parses_train_loss(self, tmp_path):
        run_dir = _make_run(tmp_path, "run1")
        info = RunReader(run_dir).read()
        assert info.final_train_loss == 2.5

    def test_parses_eval_loss(self, tmp_path):
        run_dir = _make_run(tmp_path, "run1")
        info = RunReader(run_dir).read()
        assert info.best_eval_loss == 1.8

    def test_reads_config_yaml(self, tmp_path):
        cfg = {"method": "lora", "model": {"name": "gpt2"}}
        run_dir = _make_run(tmp_path, "run1", config=cfg)
        info = RunReader(run_dir).read()
        assert info.config["method"] == "lora"

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RunReader(tmp_path / "nonexistent").read()

    def test_missing_trainer_state_raises(self, tmp_path):
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        with pytest.raises(ValueError, match="trainer_state.json"):
            RunReader(run_dir).read()

    def test_runtime_parsed(self, tmp_path):
        run_dir = _make_run(tmp_path, "run1")
        info = RunReader(run_dir).read()
        assert info.train_runtime_seconds == 120.0
        assert info.train_samples_per_second == 42.0

    def test_has_eval_metrics(self, tmp_path):
        run_dir = _make_run(tmp_path, "run1")
        info = RunReader(run_dir).read()
        assert info.has_eval_metrics is True

    def test_no_eval_metrics(self, tmp_path):
        history = [{"step": 50, "epoch": 0.5, "loss": 2.5}]
        run_dir = _make_run(tmp_path, "run1", log_history=history)
        info = RunReader(run_dir).read()
        assert info.has_eval_metrics is False


# ---------------------------------------------------------------------------
# RunComparator
# ---------------------------------------------------------------------------


class TestRunComparator:
    def test_compare_two_runs(self, tmp_path):
        r1 = _make_run(tmp_path, "run1", best_metric=0.85)
        r2 = _make_run(tmp_path, "run2", best_metric=0.92)
        result = RunComparator().compare([r1, r2])
        assert len(result.runs) == 2
        assert result.winner is not None
        assert result.winner.name == "run2"

    def test_winner_by_best_metric(self, tmp_path):
        r1 = _make_run(tmp_path, "run1", best_metric=0.70)
        r2 = _make_run(tmp_path, "run2", best_metric=0.90)
        result = RunComparator().compare([r1, r2])
        assert result.winner.name == "run2"

    def test_winner_by_eval_loss_when_no_metric(self, tmp_path):
        history_good = [{"step": 100, "epoch": 1.0, "eval_loss": 1.2}]
        history_bad = [{"step": 100, "epoch": 1.0, "eval_loss": 2.5}]
        r1 = _make_run(tmp_path, "run1", log_history=history_good)
        r2 = _make_run(tmp_path, "run2", log_history=history_bad)
        result = RunComparator().compare([r1, r2])
        assert result.winner.name == "run1"

    def test_compare_three_runs(self, tmp_path):
        r1 = _make_run(tmp_path, "run1", best_metric=0.70)
        r2 = _make_run(tmp_path, "run2", best_metric=0.85)
        r3 = _make_run(tmp_path, "run3", best_metric=0.95)
        result = RunComparator().compare([r1, r2, r3])
        assert result.winner.name == "run3"
        assert len(result.runs) == 3

    def test_invalid_dir_skipped(self, tmp_path):
        r1 = _make_run(tmp_path, "run1", best_metric=0.8)
        result = RunComparator().compare([r1, tmp_path / "bad"])
        assert len(result.runs) == 1

    def test_all_invalid_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No valid runs"):
            RunComparator().compare([tmp_path / "bad1", tmp_path / "bad2"])

    def test_result_has_metric_used(self, tmp_path):
        r1 = _make_run(tmp_path, "run1", best_metric=0.8)
        r2 = _make_run(tmp_path, "run2", best_metric=0.9)
        result = RunComparator().compare([r1, r2])
        assert result.metric_used != ""

    def test_winner_reason_contains_value(self, tmp_path):
        r1 = _make_run(tmp_path, "run1", best_metric=0.8)
        r2 = _make_run(tmp_path, "run2", best_metric=0.9)
        result = RunComparator().compare([r1, r2])
        assert "0.9" in result.winner_reason


# ---------------------------------------------------------------------------
# diff_configs
# ---------------------------------------------------------------------------


class TestDiffConfigs:
    def test_detects_different_model(self, tmp_path):
        r1 = _make_run(tmp_path, "run1", config={"model": {"name": "gpt2"}})
        r2 = _make_run(tmp_path, "run2", config={"model": {"name": "bert"}})
        info1 = RunReader(r1).read()
        info2 = RunReader(r2).read()
        diffs = RunComparator().diff_configs(info1, info2)
        assert "model" in diffs

    def test_no_diff_on_same_config(self, tmp_path):
        cfg = {"method": "lora"}
        r1 = _make_run(tmp_path, "run1", config=cfg)
        r2 = _make_run(tmp_path, "run2", config=cfg)
        info1 = RunReader(r1).read()
        info2 = RunReader(r2).read()
        assert RunComparator().diff_configs(info1, info2) == {}

    def test_detects_missing_key(self, tmp_path):
        r1 = _make_run(tmp_path, "run1", config={"method": "lora", "extra": "val"})
        r2 = _make_run(tmp_path, "run2", config={"method": "lora"})
        info1 = RunReader(r1).read()
        info2 = RunReader(r2).read()
        diffs = RunComparator().diff_configs(info1, info2)
        assert "extra" in diffs


# ---------------------------------------------------------------------------
# ComparisonResult
# ---------------------------------------------------------------------------


class TestComparisonResult:
    def test_run_names(self, tmp_path):
        r1 = _make_run(tmp_path, "run1", best_metric=0.8)
        r2 = _make_run(tmp_path, "run2", best_metric=0.9)
        result = RunComparator().compare([r1, r2])
        assert set(result.run_names) == {"run1", "run2"}
