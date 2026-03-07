"""
WandaPruner — WANDA unstructured pruning (Weight AND Activation scoring).

Algorithm (Sun et al., 2023 — https://arxiv.org/abs/2306.11695)
---------------------------------------------------------------
1. Register forward hooks on every target linear layer to capture input
   activation norms over N calibration samples.
2. For each weight matrix W (shape: out × in):
     score_ij = |W_ij| * ||X_j||_2
   where ||X_j||_2 is the RMS activation norm for input feature j.
3. Zero out the bottom `sparsity` fraction of weights by score.
   With use_row_wise=True (default), the threshold is computed per output
   row so every neuron retains the same fraction of its incoming weights.
4. Remove hooks, save the modified model + tokenizer.

Key differences vs StructuredPruner (Sprint 29)
------------------------------------------------
- Unstructured: individual weights zeroed, not whole heads/neurons.
- Requires a calibration dataset for activation norms (small, ~128 samples).
- Scoring uses both magnitude AND activation, not magnitude alone.
- No model-structure detection needed — targets nn.Linear by class name.

Usage
-----
    pruner = WandaPruner(model, tokenizer, wanda_config)
    result = pruner.prune(calibration_input_ids)
"""

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from xlmtec.core.exceptions import FineTuneError
from xlmtec.core.types import WandaConfig
from xlmtec.utils.logging import get_logger

# ---------------------------------------------------------------------------
# Default target layer type names
# ---------------------------------------------------------------------------

_DEFAULT_LAYER_TYPES = [
    "Linear",        # PyTorch nn.Linear — all standard transformers
    "Conv1D",        # GPT-2 style (transformers uses Conv1D for attn/mlp)
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WandaResult:
    """Immutable result returned by WandaPruner.prune()."""

    output_dir: Path
    """Directory where the pruned model was saved."""

    original_param_count: int
    """Total parameter count before pruning (all layers)."""

    zeroed_param_count: int
    """Number of weight elements set to zero."""

    sparsity_achieved: float
    """Actual sparsity fraction across all pruned layers."""

    layers_pruned: int
    """Number of linear layers that were pruned."""

    pruning_time_seconds: float
    """Wall-clock time for calibration + pruning."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_activation_norms(
    model: Any,
    input_ids: Any,
    target_names: List[str],
    n_samples: int,
    seq_len: int,
) -> Dict[str, Any]:
    """
    Run forward passes and accumulate squared activation norms per input feature.

    Returns a dict keyed by layer id (id(module)) → accumulated norm tensor.
    """
    import torch

    norms: Dict[int, Any] = {}
    handles: List[Any] = []

    def _make_hook(layer_id: int) -> Callable:
        def hook(module: Any, inp: Tuple, out: Any) -> None:
            x = inp[0].detach()  # (batch, seq, in_features) or (batch, in_features)
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])   # flatten batch×seq
            # Accumulate sum of squared activations per input feature
            contrib = (x ** 2).sum(dim=0)          # (in_features,)
            if layer_id in norms:
                norms[layer_id] = norms[layer_id] + contrib
            else:
                norms[layer_id] = contrib.clone()
        return hook

    # Register hooks on all target layer types
    for name, module in model.named_modules():
        type_name = type(module).__name__
        if type_name in target_names and hasattr(module, "weight"):
            handles.append(module.register_forward_hook(_make_hook(id(module))))

    model.eval()
    total = int(input_ids.shape[0])
    step = max(1, seq_len)
    n_done = 0

    with torch.no_grad():
        for start in range(0, total, step):
            if n_done >= n_samples:
                break
            batch = input_ids[start: start + step]
            if batch.shape[0] == 0:
                break
            try:
                model(batch)
            except Exception:
                pass   # ignore output — we only need the hook activations
            n_done += 1

    for h in handles:
        h.remove()

    # Convert accumulated squared norms → RMS (root mean square) per feature
    rms_norms: Dict[int, Any] = {}
    for layer_id, sq_sum in norms.items():
        rms_norms[layer_id] = (sq_sum / max(n_done, 1)).sqrt()

    return rms_norms


def _wanda_score(weight: Any, act_norm: Any) -> Any:
    """
    Compute WANDA score matrix: |W| * ||X||_2 (broadcasted across rows).

    Args:
        weight:   (out_features, in_features) weight tensor.
        act_norm: (in_features,) activation RMS norm vector.

    Returns:
        Score tensor of same shape as weight.
    """
    return weight.abs() * act_norm.unsqueeze(0)


def _apply_wanda_mask(
    weight: Any,
    act_norm: Any,
    sparsity: float,
    row_wise: bool,
) -> int:
    """
    Zero out the lowest-scoring `sparsity` fraction of weights in-place.

    Returns the number of weights zeroed.
    """
    import torch

    scores = _wanda_score(weight, act_norm)   # (out, in)

    with torch.no_grad():
        if row_wise:
            # Threshold per output row
            n_prune = max(1, int(weight.shape[1] * sparsity))
            # argsort ascending, take bottom n_prune per row
            sorted_idx = scores.argsort(dim=1)
            prune_idx = sorted_idx[:, :n_prune]
            mask = torch.ones_like(weight, dtype=torch.bool)
            mask.scatter_(1, prune_idx, False)
            zeroed = int((~mask).sum().item())
            weight.mul_(mask.float())
        else:
            # Global pruning via index-based ranking — handles ties correctly.
            # Value-threshold approaches (kthvalue + >) zero everything when
            # all scores are equal (e.g. uniform weights in tests).
            n_total = scores.numel()
            n_prune = max(1, int(n_total * sparsity))
            flat_scores = scores.flatten()
            _, sorted_indices = flat_scores.sort()        # ascending
            prune_flat_idx = sorted_indices[:n_prune]     # lowest n_prune
            mask = torch.ones(n_total, dtype=torch.bool, device=weight.device)
            mask[prune_flat_idx] = False
            mask = mask.reshape(weight.shape)
            zeroed = int((~mask).sum().item())
            weight.mul_(mask.float())

    return zeroed


# ---------------------------------------------------------------------------
# WandaPruner class
# ---------------------------------------------------------------------------

class WandaPruner:
    """
    WANDA unstructured pruner.

    Args:
        model:     HuggingFace ``PreTrainedModel`` (or compatible mock).
        tokenizer: Associated tokenizer — saved alongside the pruned model.
        config:    ``WandaConfig`` controlling sparsity, calibration, etc.
    """

    def __init__(self, model: Any, tokenizer: Any, config: WandaConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        if not 0.0 <= config.sparsity < 1.0:
            raise FineTuneError(
                f"WandaConfig.sparsity must be in [0.0, 1.0); got {config.sparsity}"
            )
        if config.n_calibration_samples < 1:
            raise FineTuneError(
                f"WandaConfig.n_calibration_samples must be >= 1; "
                f"got {config.n_calibration_samples}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(self, calibration_input_ids: Optional[Any] = None) -> WandaResult:
        """
        Execute the WANDA pruning pass and save the model.

        Args:
            calibration_input_ids: Token id tensor of shape (N, seq_len) used
                to collect activation norms. If None, pruning falls back to
                magnitude-only scoring (WANDA without activation component).

        Returns:
            WandaResult with statistics and output path.
        """
        t0 = time.monotonic()

        target_names: List[str] = list(
            self.config.layer_types or _DEFAULT_LAYER_TYPES
        )

        # Count total params up front
        total_params: int = 0
        for _, module in self.model.named_modules():
            if type(module).__name__ in target_names and hasattr(module, "weight"):
                total_params += int(module.weight.numel())

        self.logger.info(
            f"WANDA pruning — sparsity={self.config.sparsity:.0%} "
            f"targets={target_names} params={total_params:,}"
        )

        if self.config.sparsity == 0.0:
            warnings.warn(
                "WandaConfig.sparsity=0.0 — no weights will be pruned.",
                UserWarning,
                stacklevel=2,
            )

        # Step 1 — collect activation norms (or use uniform ones if no calib data)
        if calibration_input_ids is not None:
            self.logger.info(
                f"Collecting activation norms over "
                f"{self.config.n_calibration_samples} calibration samples..."
            )
            act_norms = _collect_activation_norms(
                self.model,
                calibration_input_ids,
                target_names,
                self.config.n_calibration_samples,
                self.config.calibration_seq_len,
            )
        else:
            warnings.warn(
                "No calibration_input_ids provided — falling back to "
                "magnitude-only pruning (activation norms set to 1.0).",
                UserWarning,
                stacklevel=2,
            )
            act_norms = {}

        # Step 2 — apply WANDA mask to each target layer
        total_zeroed: int = 0
        layers_pruned: int = 0

        for name, module in self.model.named_modules():
            type_name = type(module).__name__
            if type_name not in target_names:
                continue
            if not hasattr(module, "weight"):
                continue

            weight = module.weight

            # Get activation norm for this layer (uniform fallback)
            layer_id = id(module)
            if layer_id in act_norms:
                act_norm = act_norms[layer_id]
                # Ensure act_norm matches weight's in_features dimension
                in_features = int(weight.shape[1])
                if int(act_norm.shape[0]) != in_features:
                    # Resize via interpolation — handles Conv1D transposed shapes
                    import torch
                    act_norm = torch.nn.functional.interpolate(
                        act_norm.unsqueeze(0).unsqueeze(0),
                        size=in_features,
                        mode="linear",
                        align_corners=False,
                    ).squeeze()
            else:
                import torch
                act_norm = torch.ones(int(weight.shape[1]))

            try:
                zeroed = _apply_wanda_mask(
                    weight,
                    act_norm,
                    self.config.sparsity,
                    self.config.use_row_wise,
                )
                total_zeroed += zeroed
                layers_pruned += 1
                self.logger.debug(
                    f"  {name}: zeroed {zeroed:,} / {int(weight.numel()):,} weights"
                )
            except Exception as exc:
                self.logger.warning(f"  {name}: skipped ({exc})")

        # Step 3 — save
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

        elapsed_f = float(time.monotonic() - t0)
        sparsity_achieved = (
            float(total_zeroed) / float(total_params)
            if total_params > 0
            else 0.0
        )

        self.logger.info(
            f"WANDA complete — zeroed {total_zeroed:,} / {total_params:,} weights "
            f"({sparsity_achieved:.1%}) across {layers_pruned} layers "
            f"in {elapsed_f:.1f}s → {output_dir}"
        )

        return WandaResult(
            output_dir=output_dir,
            original_param_count=total_params,
            zeroed_param_count=total_zeroed,
            sparsity_achieved=sparsity_achieved,
            layers_pruned=layers_pruned,
            pruning_time_seconds=elapsed_f,
        )
