"""
tests/test_ux.py
Sprint 35 UX tests — all logic tested directly, no CLI runner path issues.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

VALID_CONFIG = {
    "method": "lora",
    "model": {"name": "gpt2"},
    "dataset": {"source": "local_file", "path": "data/train.jsonl"},
    "lora": {"r": 16, "alpha": 32},
    "training": {
        "output_dir": "output/run1",
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
    },
}


@pytest.fixture
def valid_config_file(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(VALID_CONFIG), encoding="utf-8")
    return p


@pytest.fixture
def invalid_yaml_file(tmp_path: Path) -> Path:
    p = tmp_path / "bad.yaml"
    p.write_text("key: [unclosed", encoding="utf-8")
    return p


@pytest.fixture
def list_yaml_file(tmp_path: Path) -> Path:
    p = tmp_path / "list.yaml"
    p.write_text("- item1\n- item2\n", encoding="utf-8")
    return p


@pytest.fixture
def high_epoch_file(tmp_path: Path) -> Path:
    cfg = {**VALID_CONFIG, "training": {**VALID_CONFIG["training"], "num_epochs": 20}}
    p = tmp_path / "warn.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# get_version
# ---------------------------------------------------------------------------


class TestGetVersion:
    def test_returns_string(self):
        from xlmtec.cli.ux import get_version

        v = get_version()
        assert isinstance(v, str) and len(v) > 0

    def test_fallback_on_missing_package(self):
        from xlmtec.cli.ux import get_version

        with patch("xlmtec.cli.ux.version", side_effect=PackageNotFoundError("xlmtec")):
            v = get_version()
        assert v == "0.0.0-dev"


# ---------------------------------------------------------------------------
# UX panels
# ---------------------------------------------------------------------------


class TestUXPanels:
    def test_print_error_no_crash(self):
        from xlmtec.cli.ux import print_error

        print_error("Test error", "Something went wrong")

    def test_print_success_no_crash(self):
        from xlmtec.cli.ux import print_success

        print_success("Done", "Everything worked fine")

    def test_print_warning_no_crash(self):
        from xlmtec.cli.ux import print_warning

        print_warning("Check your config")

    def test_print_dry_run_table_no_crash(self):
        from xlmtec.cli.ux import print_dry_run_table

        print_dry_run_table([("Model", "gpt2"), ("Epochs", "3")])


# ---------------------------------------------------------------------------
# dry_run — tested directly
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_valid_config_returns_0(self, valid_config_file):
        from xlmtec.cli.commands.dry_run import execute_dry_run

        with patch("xlmtec.cli.commands.dry_run.PipelineConfig") as MockCfg:
            MockCfg.model_construct.return_value = MagicMock()
            assert execute_dry_run(valid_config_file) == 0

    def test_missing_file_returns_1(self, tmp_path):
        from xlmtec.cli.commands.dry_run import execute_dry_run

        assert execute_dry_run(tmp_path / "nonexistent.yaml") == 1

    def test_invalid_yaml_returns_1(self, invalid_yaml_file):
        from xlmtec.cli.commands.dry_run import execute_dry_run

        assert execute_dry_run(invalid_yaml_file) == 1

    def test_model_construct_error_returns_1(self, valid_config_file):
        from xlmtec.cli.commands.dry_run import execute_dry_run

        with patch("xlmtec.cli.commands.dry_run.PipelineConfig") as MockCfg:
            MockCfg.model_construct.side_effect = Exception("bad config")
            assert execute_dry_run(valid_config_file) == 1


# ---------------------------------------------------------------------------
# validate_config — tested directly (bypasses typer argument parsing)
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_config_returns_0(self, valid_config_file):
        from xlmtec.cli.commands.config_validate import validate_config

        with patch("xlmtec.cli.commands.config_validate.PipelineConfig") as MockCfg:
            MockCfg.return_value = MagicMock()
            assert validate_config(valid_config_file) == 0

    def test_missing_file_returns_1(self, tmp_path):
        from xlmtec.cli.commands.config_validate import validate_config

        assert validate_config(tmp_path / "nonexistent.yaml") == 1

    def test_invalid_yaml_returns_1(self, invalid_yaml_file):
        from xlmtec.cli.commands.config_validate import validate_config

        assert validate_config(invalid_yaml_file) == 1

    def test_non_dict_yaml_returns_1(self, list_yaml_file):
        from xlmtec.cli.commands.config_validate import validate_config

        assert validate_config(list_yaml_file) == 1

    def test_validation_error_returns_1(self, valid_config_file):
        from pydantic import ValidationError

        from xlmtec.cli.commands.config_validate import validate_config

        with patch("xlmtec.cli.commands.config_validate.PipelineConfig") as MockCfg:
            MockCfg.side_effect = ValidationError.from_exception_data("PipelineConfig", [])
            assert validate_config(valid_config_file) == 1

    def test_strict_fails_on_warnings(self, high_epoch_file):
        from xlmtec.cli.commands.config_validate import validate_config

        with patch("xlmtec.cli.commands.config_validate.PipelineConfig") as MockCfg:
            MockCfg.return_value = MagicMock()
            assert validate_config(high_epoch_file, strict=True) == 1

    def test_warnings_non_strict_returns_0(self, high_epoch_file):
        from xlmtec.cli.commands.config_validate import validate_config

        with patch("xlmtec.cli.commands.config_validate.PipelineConfig") as MockCfg:
            MockCfg.return_value = MagicMock()
            assert validate_config(high_epoch_file, strict=False) == 0


# ---------------------------------------------------------------------------
# _check_warnings
# ---------------------------------------------------------------------------


class TestCheckWarnings:
    def test_no_warnings_on_normal_config(self):
        from xlmtec.cli.commands.config_validate import _check_warnings

        assert _check_warnings(VALID_CONFIG) == []

    def test_warns_high_epochs(self):
        from xlmtec.cli.commands.config_validate import _check_warnings

        cfg = {**VALID_CONFIG, "training": {"num_epochs": 20}}
        assert len(_check_warnings(cfg)) == 1

    def test_warns_high_lr(self):
        from xlmtec.cli.commands.config_validate import _check_warnings

        cfg = {**VALID_CONFIG, "training": {"learning_rate": 0.1}}
        assert len(_check_warnings(cfg)) == 1


# ---------------------------------------------------------------------------
# progress utilities
# ---------------------------------------------------------------------------


class TestTaskProgress:
    def test_runs_without_error(self):
        from xlmtec.cli.ux import task_progress

        with task_progress("Testing..."):
            pass


class TestMakeTrainingProgress:
    def test_returns_progress_instance(self):
        from rich.progress import Progress

        from xlmtec.cli.ux import make_training_progress

        assert isinstance(make_training_progress(), Progress)
