"""
tests/test_checkpoints.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for checkpoint discovery, validation, and resume command.
No real training — all filesystem work uses tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from xlmtec.checkpoints.manager import CheckpointInfo, CheckpointManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checkpoint(
    parent: Path, step: int, epoch: float = 1.0, best_metric: float | None = None
) -> Path:
    """Create a fake HF Trainer checkpoint directory."""
    ckpt = parent / f"checkpoint-{step}"
    ckpt.mkdir()
    state = {"epoch": epoch, "global_step": step}
    if best_metric is not None:
        state["best_metric"] = best_metric
    (ckpt / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    return ckpt


def _make_output_dir(tmp_path: Path, steps: list[int]) -> Path:
    out = tmp_path / "output"
    out.mkdir()
    for step in steps:
        _make_checkpoint(out, step, epoch=step / 100)
    return out


VALID_CONFIG = {
    "method": "lora",
    "model": {"name": "gpt2"},
    "dataset": {"source": "local_file", "path": "data/train.jsonl"},
    "training": {"output_dir": "output/run1", "num_epochs": 3},
}


# ---------------------------------------------------------------------------
# CheckpointInfo
# ---------------------------------------------------------------------------


class TestCheckpointInfo:
    def test_str_with_metric(self, tmp_path):
        ckpt = CheckpointInfo(
            path=tmp_path / "checkpoint-500", step=500, epoch=1.0, best_metric=0.92
        )
        assert "500" in str(ckpt)
        assert "0.9200" in str(ckpt)

    def test_str_without_metric(self, tmp_path):
        ckpt = CheckpointInfo(path=tmp_path / "checkpoint-100", step=100, epoch=0.5)
        assert "100" in str(ckpt)


# ---------------------------------------------------------------------------
# CheckpointManager.list_checkpoints
# ---------------------------------------------------------------------------


class TestListCheckpoints:
    def test_returns_sorted_by_step(self, tmp_path):
        out = _make_output_dir(tmp_path, [500, 100, 250])
        manager = CheckpointManager(out)
        results = manager.list_checkpoints()
        assert [c.step for c in results] == [100, 250, 500]

    def test_ignores_non_checkpoint_dirs(self, tmp_path):
        out = _make_output_dir(tmp_path, [100])
        (out / "logs").mkdir()
        (out / "final_model").mkdir()
        results = CheckpointManager(out).list_checkpoints()
        assert len(results) == 1

    def test_ignores_dirs_without_trainer_state(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        (out / "checkpoint-100").mkdir()  # no trainer_state.json
        results = CheckpointManager(out).list_checkpoints()
        assert results == []

    def test_missing_output_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Output directory not found"):
            CheckpointManager(tmp_path / "nonexistent").list_checkpoints()

    def test_empty_dir_returns_empty_list(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        assert CheckpointManager(out).list_checkpoints() == []

    def test_parses_epoch_from_state(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        _make_checkpoint(out, step=200, epoch=2.5)
        results = CheckpointManager(out).list_checkpoints()
        assert results[0].epoch == 2.5

    def test_parses_best_metric(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        _make_checkpoint(out, step=300, best_metric=0.88)
        results = CheckpointManager(out).list_checkpoints()
        assert results[0].best_metric == 0.88


# ---------------------------------------------------------------------------
# CheckpointManager.latest
# ---------------------------------------------------------------------------


class TestLatest:
    def test_returns_highest_step(self, tmp_path):
        out = _make_output_dir(tmp_path, [100, 200, 500])
        assert CheckpointManager(out).latest().step == 500

    def test_raises_if_no_checkpoints(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        with pytest.raises(ValueError, match="No checkpoints found"):
            CheckpointManager(out).latest()


# ---------------------------------------------------------------------------
# CheckpointManager.get
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_by_name(self, tmp_path):
        out = _make_output_dir(tmp_path, [100, 500])
        ckpt = CheckpointManager(out).get("checkpoint-100")
        assert ckpt.step == 100

    def test_get_by_step_number(self, tmp_path):
        out = _make_output_dir(tmp_path, [100, 500])
        ckpt = CheckpointManager(out).get("500")
        assert ckpt.step == 500

    def test_not_found_raises(self, tmp_path):
        out = _make_output_dir(tmp_path, [100])
        with pytest.raises(ValueError, match="not found"):
            CheckpointManager(out).get("checkpoint-999")


# ---------------------------------------------------------------------------
# CheckpointManager.summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_contains_steps(self, tmp_path):
        out = _make_output_dir(tmp_path, [100, 200])
        summary = CheckpointManager(out).summary()
        assert "100" in summary
        assert "200" in summary
        assert "latest" in summary

    def test_summary_on_missing_dir(self, tmp_path):
        summary = CheckpointManager(tmp_path / "missing").summary()
        assert "not found" in summary.lower() or "Output directory" in summary


# ---------------------------------------------------------------------------
# resume_training (direct logic test)
# ---------------------------------------------------------------------------


class TestResumeTraining:
    def _make_env(self, tmp_path) -> tuple[Path, Path]:
        out = _make_output_dir(tmp_path, [100, 500])
        cfg_file = out / "config.yaml"
        cfg_file.write_text(yaml.dump(VALID_CONFIG), encoding="utf-8")
        return out, cfg_file

    def test_dry_run_returns_0(self, tmp_path):
        from xlmtec.cli.commands.resume import resume_training

        out, cfg = self._make_env(tmp_path)
        code = resume_training(out, cfg, None, None, dry_run=True)
        assert code == 0

    def test_missing_output_dir_returns_1(self, tmp_path):
        from xlmtec.cli.commands.resume import resume_training

        code = resume_training(tmp_path / "nonexistent", None, None, None, dry_run=True)
        assert code == 1

    def test_specific_checkpoint_dry_run(self, tmp_path):
        from xlmtec.cli.commands.resume import resume_training

        out, cfg = self._make_env(tmp_path)
        code = resume_training(out, cfg, "checkpoint-100", None, dry_run=True)
        assert code == 0

    def test_bad_checkpoint_returns_1(self, tmp_path):
        from xlmtec.cli.commands.resume import resume_training

        out, cfg = self._make_env(tmp_path)
        code = resume_training(out, cfg, "checkpoint-999", None, dry_run=True)
        assert code == 1

    def test_missing_config_returns_1(self, tmp_path):
        from xlmtec.cli.commands.resume import resume_training

        out = _make_output_dir(tmp_path, [100])
        # No config.yaml in output dir
        code = resume_training(out, None, None, None, dry_run=True)
        assert code == 1

    def test_explicit_config_path(self, tmp_path):
        from xlmtec.cli.commands.resume import resume_training

        out = _make_output_dir(tmp_path, [100])
        cfg = tmp_path / "my_config.yaml"
        cfg.write_text(yaml.dump(VALID_CONFIG), encoding="utf-8")
        code = resume_training(out, cfg, None, None, dry_run=True)
        assert code == 0
