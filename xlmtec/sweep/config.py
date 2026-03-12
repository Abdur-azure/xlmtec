"""
xlmtec.sweep.config
~~~~~~~~~~~~~~~~~~~~
Data types for hyperparameter sweep configuration.

Sweep YAML format (extends a normal PipelineConfig YAML with a ``sweep:`` section):

.. code-block:: yaml

    model:
      name: gpt2
    dataset:
      source: local_file
      path: ./data/sample.jsonl
    tokenization:
      max_length: 128
    training:
      method: lora
      output_dir: ./sweep_output
      num_epochs: 1
    lora:
      r: 8
      lora_alpha: 16

    sweep:
      n_trials: 20
      metric: train_loss
      direction: minimize
      output_dir: ./sweep_results
      sampler: tpe
      params:
        training.learning_rate:
          type: float
          low: 1.0e-5
          high: 1.0e-3
          log: true
        training.batch_size:
          type: categorical
          choices: [2, 4, 8]
        lora.r:
          type: int
          low: 4
          high: 32
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

# ---------------------------------------------------------------------------
# ParamSpec — one searchable hyper-parameter
# ---------------------------------------------------------------------------

ParamType = Literal["float", "int", "categorical"]
Direction = Literal["minimize", "maximize"]
Sampler = Literal["tpe", "random", "grid"]

_VALID_SAMPLERS = {"tpe", "random", "grid"}
_VALID_DIRECTIONS = {"minimize", "maximize"}
_VALID_PARAM_TYPES = {"float", "int", "categorical"}


@dataclass
class ParamSpec:
    """Defines the search space for a single hyper-parameter.

    For ``float`` and ``int`` params supply ``low`` and ``high``.
    For ``categorical`` params supply ``choices``.

    ``log`` (float only): sample on a log scale — useful for learning rates.

    Dotted-path keys (e.g. ``training.learning_rate``, ``lora.r``) are
    resolved against the base PipelineConfig dict before each trial.
    """

    type: ParamType
    low: Optional[float] = None  # float / int
    high: Optional[float] = None  # float / int
    log: bool = False  # float only — log-scale sampling
    choices: Optional[List[Any]] = None  # categorical

    # ------------------------------------------------------------------

    def validate(self, name: str) -> None:
        """Raise ValueError if spec is inconsistent."""
        if self.type not in _VALID_PARAM_TYPES:
            raise ValueError(
                f"ParamSpec '{name}': type must be one of {sorted(_VALID_PARAM_TYPES)}, "
                f"got '{self.type}'"
            )
        if self.type in ("float", "int"):
            if self.low is None or self.high is None:
                raise ValueError(
                    f"ParamSpec '{name}': 'low' and 'high' are required for type='{self.type}'"
                )
            if self.high <= self.low:
                raise ValueError(
                    f"ParamSpec '{name}': 'high' ({self.high}) must be > 'low' ({self.low})"
                )
            if self.log and self.low <= 0:
                raise ValueError(
                    f"ParamSpec '{name}': log=True requires low > 0, got low={self.low}"
                )
        if self.type == "categorical":
            if not self.choices:
                raise ValueError(
                    f"ParamSpec '{name}': 'choices' must be a non-empty list for type='categorical'"
                )

    @classmethod
    def from_dict(cls, name: str, d: Dict[str, Any]) -> "ParamSpec":
        """Parse a ParamSpec from a YAML-sourced dict and validate it."""
        ptype = d.get("type")
        if ptype is None:
            raise ValueError(f"ParamSpec '{name}': missing required field 'type'")

        spec = cls(
            type=ptype,
            low=d.get("low"),
            high=d.get("high"),
            log=bool(d.get("log", False)),
            choices=d.get("choices"),
        )
        spec.validate(name)
        return spec


# ---------------------------------------------------------------------------
# SweepConfig — top-level sweep configuration
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    """Full sweep configuration parsed from the ``sweep:`` section of a YAML.

    Attributes:
        n_trials:    Number of Optuna trials to run.
        metric:      Field from ``TrainingResult`` to optimise (e.g. 'train_loss').
        direction:   'minimize' or 'maximize'.
        output_dir:  Root directory for per-trial output sub-directories.
        params:      Mapping of dotted-path param name → ParamSpec.
        sampler:     Optuna sampler: 'tpe' (default), 'random', or 'grid'.
        timeout:     Optional wall-clock timeout in seconds for the whole study.
    """

    n_trials: int
    metric: str
    direction: Direction
    output_dir: str
    params: Dict[str, ParamSpec]
    sampler: Sampler = "tpe"
    timeout: Optional[float] = None

    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ValueError for any invalid top-level fields."""
        if self.n_trials < 1:
            raise ValueError(f"SweepConfig: n_trials must be >= 1, got {self.n_trials}")
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"SweepConfig: direction must be one of {sorted(_VALID_DIRECTIONS)}, "
                f"got '{self.direction}'"
            )
        if self.sampler not in _VALID_SAMPLERS:
            raise ValueError(
                f"SweepConfig: sampler must be one of {sorted(_VALID_SAMPLERS)}, "
                f"got '{self.sampler}'"
            )
        if not self.params:
            raise ValueError("SweepConfig: 'params' must contain at least one parameter")
        if not self.metric:
            raise ValueError("SweepConfig: 'metric' must not be empty")
        for name, spec in self.params.items():
            spec.validate(name)  # re-validate in context of full config

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SweepConfig":
        """Parse and validate from a YAML-sourced dict (the 'sweep' subsection)."""
        params_raw: Dict[str, Any] = d.get("params", {})
        if not isinstance(params_raw, dict):
            raise ValueError("SweepConfig: 'params' must be a mapping")

        params = {name: ParamSpec.from_dict(name, spec) for name, spec in params_raw.items()}

        cfg = cls(
            n_trials=int(d.get("n_trials", 10)),
            metric=str(d.get("metric", "train_loss")),
            direction=d.get("direction", "minimize"),
            output_dir=str(d.get("output_dir", "./sweep_results")),
            params=params,
            sampler=d.get("sampler", "tpe"),
            timeout=d.get("timeout"),
        )
        cfg.validate()
        return cfg
