"""
tests/test_sweep.py
~~~~~~~~~~~~~~~~~~~~
Unit tests for the hyperparameter sweep system.

Optuna and training stack are mocked — no GPU, no real model downloads,
no real Optuna trials.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from xlmtec.sweep.config import ParamSpec, SweepConfig
from xlmtec.sweep.runner import SweepResult, SweepRunner, TrialResult, _apply_params

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def float_spec() -> ParamSpec:
    return ParamSpec(type="float", low=1e-5, high=1e-3, log=True)


@pytest.fixture
def int_spec() -> ParamSpec:
    return ParamSpec(type="int", low=4, high=32)


@pytest.fixture
def categorical_spec() -> ParamSpec:
    return ParamSpec(type="categorical", choices=[2, 4, 8])


@pytest.fixture
def valid_sweep_dict() -> Dict[str, Any]:
    return {
        "n_trials": 5,
        "metric": "train_loss",
        "direction": "minimize",
        "output_dir": "./sweep_results",
        "sampler": "tpe",
        "params": {
            "training.learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
            "training.batch_size": {"type": "categorical", "choices": [2, 4, 8]},
            "lora.r": {"type": "int", "low": 4, "high": 32},
        },
    }


@pytest.fixture
def base_config_dict() -> Dict[str, Any]:
    return {
        "model": {"name": "gpt2"},
        "dataset": {"source": "local_file", "path": "./data.jsonl"},
        "tokenization": {"max_length": 128},
        "training": {
            "method": "lora",
            "output_dir": "./output",
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 2e-4,
        },
        "lora": {"r": 8, "lora_alpha": 16},
    }


def _write_sweep_yaml(path: Path, base: Dict, sweep: Dict) -> Path:
    import yaml

    cfg = {**base, "sweep": sweep}
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


# ============================================================================
# ParamSpec
# ============================================================================


class TestParamSpec:
    def test_float_spec_valid(self, float_spec):
        float_spec.validate("lr")  # should not raise

    def test_int_spec_valid(self, int_spec):
        int_spec.validate("r")

    def test_categorical_spec_valid(self, categorical_spec):
        categorical_spec.validate("batch_size")

    def test_float_missing_low_raises(self):
        spec = ParamSpec(type="float", high=1e-3)
        with pytest.raises(ValueError, match="low.*high"):
            spec.validate("lr")

    def test_float_high_lte_low_raises(self):
        spec = ParamSpec(type="float", low=1e-3, high=1e-5)
        with pytest.raises(ValueError, match="must be > .low"):
            spec.validate("lr")

    def test_float_log_requires_positive_low(self):
        spec = ParamSpec(type="float", low=0.0, high=1e-3, log=True)
        with pytest.raises(ValueError, match="log=True requires low > 0"):
            spec.validate("lr")

    def test_categorical_empty_choices_raises(self):
        spec = ParamSpec(type="categorical", choices=[])
        with pytest.raises(ValueError, match="non-empty"):
            spec.validate("method")

    def test_invalid_type_raises(self):
        spec = ParamSpec(type="string")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="type must be one of"):
            spec.validate("x")

    def test_from_dict_float(self):
        spec = ParamSpec.from_dict("lr", {"type": "float", "low": 1e-5, "high": 1e-3, "log": True})
        assert spec.type == "float"
        assert spec.log is True

    def test_from_dict_missing_type_raises(self):
        with pytest.raises(ValueError, match="missing required field 'type'"):
            ParamSpec.from_dict("lr", {"low": 1e-5, "high": 1e-3})


# ============================================================================
# SweepConfig
# ============================================================================


class TestSweepConfig:
    def test_valid_config_parses(self, valid_sweep_dict):
        cfg = SweepConfig.from_dict(valid_sweep_dict)
        assert cfg.n_trials == 5
        assert cfg.metric == "train_loss"
        assert cfg.direction == "minimize"
        assert len(cfg.params) == 3

    def test_defaults_applied(self):
        cfg = SweepConfig.from_dict(
            {
                "params": {"lora.r": {"type": "int", "low": 4, "high": 32}},
            }
        )
        assert cfg.n_trials == 10
        assert cfg.sampler == "tpe"
        assert cfg.metric == "train_loss"

    def test_invalid_direction_raises(self, valid_sweep_dict):
        valid_sweep_dict["direction"] = "sideways"
        with pytest.raises(ValueError, match="direction"):
            SweepConfig.from_dict(valid_sweep_dict)

    def test_invalid_sampler_raises(self, valid_sweep_dict):
        valid_sweep_dict["sampler"] = "bayesian"
        with pytest.raises(ValueError, match="sampler"):
            SweepConfig.from_dict(valid_sweep_dict)

    def test_n_trials_zero_raises(self, valid_sweep_dict):
        valid_sweep_dict["n_trials"] = 0
        with pytest.raises(ValueError, match="n_trials"):
            SweepConfig.from_dict(valid_sweep_dict)

    def test_empty_params_raises(self, valid_sweep_dict):
        valid_sweep_dict["params"] = {}
        with pytest.raises(ValueError, match="params"):
            SweepConfig.from_dict(valid_sweep_dict)


# ============================================================================
# _apply_params
# ============================================================================


class TestApplyParams:
    def test_nested_path_applied(self, base_config_dict):
        result = _apply_params(base_config_dict, {"training.learning_rate": 1e-4})
        assert result["training"]["learning_rate"] == 1e-4

    def test_does_not_mutate_original(self, base_config_dict):
        original_lr = base_config_dict["training"]["learning_rate"]
        _apply_params(base_config_dict, {"training.learning_rate": 9e-9})
        assert base_config_dict["training"]["learning_rate"] == original_lr

    def test_creates_missing_intermediate_key(self, base_config_dict):
        result = _apply_params(base_config_dict, {"new_section.key": "value"})
        assert result["new_section"]["key"] == "value"

    def test_multiple_params_applied(self, base_config_dict):
        result = _apply_params(
            base_config_dict,
            {
                "training.batch_size": 8,
                "lora.r": 16,
            },
        )
        assert result["training"]["batch_size"] == 8
        assert result["lora"]["r"] == 16


# ============================================================================
# SweepRunner (Optuna mocked)
# ============================================================================


def _make_mock_optuna(metric_value: float = 0.42):
    """Build a complete mock optuna module."""
    mock_trial = MagicMock()
    mock_trial.number = 0
    mock_trial.suggest_float.return_value = 2e-4
    mock_trial.suggest_int.return_value = 8
    mock_trial.suggest_categorical.return_value = 4

    mock_best = MagicMock()
    mock_best.number = 0
    mock_best.value = metric_value
    mock_best.params = {"training.learning_rate": 2e-4, "lora.r": 8}

    mock_study = MagicMock()
    mock_study.best_trial = mock_best

    def fake_optimize(fn, n_trials, timeout):
        fn(mock_trial)

    mock_study.optimize.side_effect = fake_optimize

    mock_optuna = MagicMock()
    mock_optuna.create_study.return_value = mock_study
    mock_optuna.logging.WARNING = 30
    mock_optuna.TrialPruned = Exception  # simple stand-in
    mock_optuna.samplers.TPESampler.return_value = MagicMock()
    mock_optuna.samplers.RandomSampler.return_value = MagicMock()

    return mock_optuna, mock_study, mock_trial


class TestSweepRunner:
    def _make_runner(self, base_config_dict, valid_sweep_dict) -> SweepRunner:
        cfg = SweepConfig.from_dict(valid_sweep_dict)
        return SweepRunner(base_config_dict, cfg)

    def _mock_training_result(self, train_loss: float = 0.42):
        result = MagicMock()
        result.train_loss = train_loss
        result.eval_loss = None
        result.steps_completed = 10
        result.epochs_completed = 1
        return result

    @patch("xlmtec.sweep.runner.SweepRunner._objective")
    def test_run_calls_optimize(self, mock_obj, base_config_dict, valid_sweep_dict):
        mock_optuna, mock_study, _ = _make_mock_optuna()
        with patch.dict(
            "sys.modules",
            {
                "optuna": mock_optuna,
                "optuna.samplers": mock_optuna.samplers,
                "optuna.logging": mock_optuna.logging,
            },
        ):
            runner = self._make_runner(base_config_dict, valid_sweep_dict)
            runner.run(n_trials=1)
        mock_study.optimize.assert_called_once()

    def test_run_returns_sweep_result(self, base_config_dict, valid_sweep_dict):
        mock_optuna, mock_study, _ = _make_mock_optuna(metric_value=0.35)
        with patch.dict(
            "sys.modules",
            {
                "optuna": mock_optuna,
                "optuna.samplers": mock_optuna.samplers,
                "optuna.logging": mock_optuna.logging,
            },
        ):
            with patch.object(SweepRunner, "_objective", return_value=0.35):
                runner = self._make_runner(base_config_dict, valid_sweep_dict)
                runner._trials = [
                    TrialResult(0, {"training.learning_rate": 2e-4}, 0.35, Path("./t0"))
                ]
                result = runner.run(n_trials=1)

        assert isinstance(result, SweepResult)
        assert result.best_metric == pytest.approx(0.35)
        assert result.best_trial == 0

    def test_run_raises_without_optuna(self, base_config_dict, valid_sweep_dict):
        runner = self._make_runner(base_config_dict, valid_sweep_dict)
        with patch.dict("sys.modules", {"optuna": None}):
            with pytest.raises(ImportError, match="optuna"):
                runner.run()

    def test_n_trials_override(self, base_config_dict, valid_sweep_dict):
        mock_optuna, mock_study, _ = _make_mock_optuna()
        with patch.dict(
            "sys.modules",
            {
                "optuna": mock_optuna,
                "optuna.samplers": mock_optuna.samplers,
                "optuna.logging": mock_optuna.logging,
            },
        ):
            with patch.object(SweepRunner, "_objective", return_value=0.5):
                runner = self._make_runner(base_config_dict, valid_sweep_dict)
                runner._trials = []
                runner.run(n_trials=3)
        _, kwargs = mock_study.optimize.call_args
        n = kwargs.get("n_trials") or mock_study.optimize.call_args[0][1]
        assert n == 3


# ============================================================================
# CLI: run_sweep
# ============================================================================


class TestRunSweep:
    def _sweep_raw(self) -> Dict[str, Any]:
        return {
            "n_trials": 2,
            "metric": "train_loss",
            "direction": "minimize",
            "output_dir": "./sweep_out",
            "params": {
                "training.learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
            },
        }

    def _base_raw(self) -> Dict[str, Any]:
        return {
            "model": {"name": "gpt2"},
            "dataset": {"source": "local_file", "path": "./data.jsonl"},
            "tokenization": {"max_length": 128},
            "training": {
                "method": "lora",
                "output_dir": "./output",
                "num_epochs": 1,
                "batch_size": 4,
                "learning_rate": 2e-4,
            },
            "lora": {"r": 8, "lora_alpha": 16},
        }

    def test_dry_run_returns_0(self, tmp_path):
        from xlmtec.cli.commands.sweep import run_sweep

        cfg = _write_sweep_yaml(tmp_path / "sweep.yaml", self._base_raw(), self._sweep_raw())
        code = run_sweep(cfg, n_trials=None, dry_run=True)
        assert code == 0

    def test_missing_file_returns_1(self, tmp_path):
        from xlmtec.cli.commands.sweep import run_sweep

        code = run_sweep(tmp_path / "nonexistent.yaml", n_trials=None, dry_run=True)
        assert code == 1

    def test_missing_sweep_section_returns_1(self, tmp_path):
        import yaml

        from xlmtec.cli.commands.sweep import run_sweep

        path = tmp_path / "no_sweep.yaml"
        path.write_text(yaml.dump(self._base_raw()), encoding="utf-8")
        code = run_sweep(path, n_trials=None, dry_run=True)
        assert code == 1

    def test_invalid_trials_flag_returns_1(self, tmp_path):
        from xlmtec.cli.commands.sweep import run_sweep

        cfg = _write_sweep_yaml(tmp_path / "sweep.yaml", self._base_raw(), self._sweep_raw())
        code = run_sweep(cfg, n_trials=0, dry_run=True)
        assert code == 1

    def test_trials_override_applied(self, tmp_path):
        from xlmtec.cli.commands.sweep import run_sweep

        cfg = _write_sweep_yaml(tmp_path / "sweep.yaml", self._base_raw(), self._sweep_raw())
        # dry-run so no actual training; just verifying it doesn't crash + override accepted
        code = run_sweep(cfg, n_trials=7, dry_run=True)
        assert code == 0

    def test_invalid_base_config_returns_1(self, tmp_path):
        import yaml

        from xlmtec.cli.commands.sweep import run_sweep

        # base config with missing required 'method'
        bad_base = {
            "model": {"name": "gpt2"},
            "dataset": {"source": "local_file", "path": "./data.jsonl"},
            "tokenization": {"max_length": 128},
            "training": {},  # missing method
        }
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump({**bad_base, "sweep": self._sweep_raw()}), encoding="utf-8")
        code = run_sweep(path, n_trials=None, dry_run=True)
        assert code == 1

    def test_invalid_yaml_returns_1(self, tmp_path):
        from xlmtec.cli.commands.sweep import run_sweep

        path = tmp_path / "bad.yaml"
        path.write_text("key: [unclosed", encoding="utf-8")
        code = run_sweep(path, n_trials=None, dry_run=True)
        assert code == 1
