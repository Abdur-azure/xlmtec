"""
xlmtec.sweep.runner
~~~~~~~~~~~~~~~~~~~~
SweepRunner — runs Optuna trials over a PipelineConfig search space.

Each trial:
1. Suggests param values from Optuna.
2. Applies them to a deep-copy of the base config dict.
3. Constructs a ``PipelineConfig``, loads model + data, calls
   ``TrainerFactory.train()``.
4. Returns the target metric value to Optuna.

Optuna is imported lazily inside ``SweepRunner.run()`` so that
``xlmtec --help`` stays fast even without the ``[sweep]`` extra installed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger
from .config import SweepConfig

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Outcome of a single Optuna trial."""

    trial_number: int
    params: Dict[str, Any]
    metric_value: float
    output_dir: Path
    failed: bool = False
    error: str = ""


@dataclass
class SweepResult:
    """Aggregated results from a completed sweep."""

    best_params: Dict[str, Any]
    best_metric: float
    best_trial: int
    n_completed: int
    n_failed: int
    trials: List[TrialResult] = field(default_factory=list)
    direction: str = "minimize"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_params(base: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy *base* and apply dotted-path *params* into it.

    e.g. ``{"training.learning_rate": 2e-4, "lora.r": 8}`` becomes::

        result["training"]["learning_rate"] = 2e-4
        result["lora"]["r"] = 8
    """
    cfg = copy.deepcopy(base)
    for key, value in params.items():
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return cfg


def _suggest_params(trial: Any, sweep_cfg: SweepConfig) -> Dict[str, Any]:
    """Ask Optuna trial to suggest values for all params in the sweep."""
    suggested: Dict[str, Any] = {}
    for name, spec in sweep_cfg.params.items():
        if spec.type == "float":
            suggested[name] = trial.suggest_float(
                name, spec.low, spec.high, log=spec.log
            )
        elif spec.type == "int":
            suggested[name] = trial.suggest_int(
                name, int(spec.low), int(spec.high)
            )
        elif spec.type == "categorical":
            suggested[name] = trial.suggest_categorical(name, spec.choices)
    return suggested


def _build_optuna_sampler(sampler_name: str) -> Any:
    """Return an Optuna sampler instance by name."""
    import optuna

    if sampler_name == "tpe":
        return optuna.samplers.TPESampler()
    if sampler_name == "random":
        return optuna.samplers.RandomSampler()
    if sampler_name == "grid":
        raise ValueError(
            "sampler='grid' requires Optuna GridSampler search_space, "
            "which is not yet supported. Use 'tpe' or 'random'."
        )
    raise ValueError(f"Unknown sampler: '{sampler_name}'")


# ---------------------------------------------------------------------------
# SweepRunner
# ---------------------------------------------------------------------------


class SweepRunner:
    """Run a hyperparameter sweep over a base PipelineConfig.

    Args:
        base_config_dict:  Raw dict of the base ``PipelineConfig`` (not the
                           sweep section) — loaded from YAML.
        sweep_config:      Parsed ``SweepConfig``.
        study_name:        Optional Optuna study name (default: 'xlmtec_sweep').
    """

    def __init__(
        self,
        base_config_dict: Dict[str, Any],
        sweep_config: SweepConfig,
        study_name: str = "xlmtec_sweep",
    ) -> None:
        self.base_config_dict = base_config_dict
        self.sweep_config = sweep_config
        self.study_name = study_name
        self._trials: List[TrialResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, n_trials: Optional[int] = None) -> SweepResult:
        """Execute the sweep and return aggregated results.

        Args:
            n_trials: Override ``sweep_config.n_trials`` if provided.

        Returns:
            SweepResult with best params and all trial outcomes.

        Raises:
            ImportError: If optuna is not installed.
        """
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "optuna is required for hyperparameter sweep. "
                "Install it with: pip install xlmtec[sweep]"
            ) from exc

        # Silence Optuna's default verbose logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        n = n_trials or self.sweep_config.n_trials
        self._trials = []

        sampler = _build_optuna_sampler(self.sweep_config.sampler)
        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.sweep_config.direction,
            sampler=sampler,
        )

        study.optimize(
            self._objective,
            n_trials=n,
            timeout=self.sweep_config.timeout,
        )

        best = study.best_trial
        n_failed = sum(1 for t in self._trials if t.failed)

        return SweepResult(
            best_params=best.params,
            best_metric=best.value,
            best_trial=best.number,
            n_completed=len(self._trials) - n_failed,
            n_failed=n_failed,
            trials=list(self._trials),
            direction=self.sweep_config.direction,
        )

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------

    def _objective(self, trial: Any) -> float:
        """Called by Optuna once per trial."""
        import optuna

        suggested = _suggest_params(trial, self.sweep_config)
        trial_dir = Path(self.sweep_config.output_dir) / f"trial_{trial.number}"

        # Inject output_dir so each trial saves to its own folder
        suggested["training.output_dir"] = str(trial_dir)

        cfg_dict = _apply_params(self.base_config_dict, suggested)

        logger.info(f"Trial {trial.number}: {suggested}")

        try:
            from ..core.config import PipelineConfig
            from ..models.loader import load_model_and_tokenizer
            from ..data import prepare_dataset
            from ..trainers import TrainerFactory

            pipeline_cfg = PipelineConfig(**cfg_dict)
            model, tokenizer = load_model_and_tokenizer(pipeline_cfg.model.to_config())
            dataset = prepare_dataset(
                pipeline_cfg.dataset.to_config(),
                pipeline_cfg.tokenization.to_config(),
                tokenizer,
            )

            lora_cfg = pipeline_cfg.lora.to_config() if pipeline_cfg.lora else None
            result = TrainerFactory.train(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                training_config=pipeline_cfg.training.to_config(),
                lora_config=lora_cfg,
            )

            metric_value = getattr(result, self.sweep_config.metric, None)
            if metric_value is None:
                raise ValueError(
                    f"Metric '{self.sweep_config.metric}' not found in TrainingResult. "
                    f"Available: train_loss, eval_loss, steps_completed, epochs_completed."
                )

            self._trials.append(TrialResult(
                trial_number=trial.number,
                params=suggested,
                metric_value=float(metric_value),
                output_dir=trial_dir,
            ))
            return float(metric_value)

        except Exception as exc:
            logger.warning(f"Trial {trial.number} failed: {exc}")
            self._trials.append(TrialResult(
                trial_number=trial.number,
                params=suggested,
                metric_value=float("inf"),
                output_dir=trial_dir,
                failed=True,
                error=str(exc),
            ))
            raise optuna.TrialPruned() from exc