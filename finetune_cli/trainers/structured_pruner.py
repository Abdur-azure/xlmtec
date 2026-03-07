"""
StructuredPruner — remove low-importance attention heads from transformer models.

Algorithm
---------
1. For every attention layer, score each head by the mean absolute weight
   magnitude of its query projection (or whichever projection is available).
2. Sort heads by score ascending (lowest = least important).
3. Zero-out the weight rows/columns corresponding to the bottom ``sparsity``
   fraction of heads, respecting ``min_heads_per_layer``.
4. Save the modified model + tokenizer to ``config.output_dir``.

This is *soft* structured pruning: head weights are zeroed, not physically
removed from the weight matrix.  The model shape is unchanged so it runs
with any standard HuggingFace inference stack — no custom PEFT or runtime
required.  For hard pruning (actual dimension reduction) see WANDA (Sprint 30).

Usage
-----
    pruner = StructuredPruner(model, tokenizer, pruning_config)
    result = pruner.prune()
"""

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from finetune_cli.core.exceptions import FineTuneError
from finetune_cli.core.types import PruningConfig
from finetune_cli.utils.logging import get_logger

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PruningResult:
    """Immutable result returned by StructuredPruner.prune()."""

    output_dir: Path
    """Directory where the pruned model was saved."""

    original_param_count: int
    """Total parameter count before pruning."""

    zeroed_param_count: int
    """Number of parameters set to zero during pruning."""

    sparsity_achieved: float
    """Actual fraction of attention-head params zeroed (may differ from target
    if clamped by min_heads_per_layer)."""

    heads_pruned_per_layer: Dict[str, int]
    """Map of layer name → number of heads zeroed in that layer."""

    pruning_time_seconds: float
    """Wall-clock time for the pruning operation."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_params(model: Any) -> int:
    """Return total parameter count via numel() — works on real and mock models."""
    total = 0
    for p in model.parameters():
        total += int(p.numel())
    return total


def _head_importance_scores(weight: Any, num_heads: int) -> List[float]:
    """
    Score each attention head by mean absolute weight magnitude.

    Args:
        weight:    2-D weight tensor (out_features, in_features).
        num_heads: Number of attention heads this projection serves.

    Returns:
        List of float scores, one per head (ascending = less important).
    """
    # Avoid real torch import at module level — import lazily
    import torch

    rows_per_head = weight.shape[0] // num_heads
    scores = []
    for h in range(num_heads):
        start = h * rows_per_head
        end = start + rows_per_head
        scores.append(weight[start:end].abs().mean().item())
    return scores


def _zero_head_rows(weight: Any, head_indices: List[int], num_heads: int) -> int:
    """
    Zero the rows of ``weight`` that belong to each head in ``head_indices``.

    Returns the number of parameters zeroed.
    """
    import torch

    rows_per_head = weight.shape[0] // num_heads
    zeroed = 0
    with torch.no_grad():
        for h in head_indices:
            start = h * rows_per_head
            end = start + rows_per_head
            zeroed += int(weight[start:end].numel())
            weight[start:end] = 0.0
    return zeroed


# ---------------------------------------------------------------------------
# Pruner class
# ---------------------------------------------------------------------------

class StructuredPruner:
    """
    Soft structured pruner for transformer attention heads.

    Args:
        model:     A HuggingFace ``PreTrainedModel`` (or compatible mock).
        tokenizer: Associated tokenizer — saved alongside the pruned model.
        config:    ``PruningConfig`` controlling sparsity, method, etc.
    """

    # Candidate attribute names for the attention module inside a layer.
    # Different model families use different names.
    _ATTN_ATTR_CANDIDATES = ("self_attn", "attention", "self_attention", "attn")

    # Candidate attribute names for the query projection weight.
    _QUERY_PROJ_CANDIDATES = ("q_proj", "query", "query_key_value", "c_attn")

    def __init__(self, model: Any, tokenizer: Any, config: PruningConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        if not 0.0 <= config.sparsity < 1.0:
            raise FineTuneError(
                f"PruningConfig.sparsity must be in [0.0, 1.0); got {config.sparsity}"
            )
        if config.method not in ("heads", "ffn"):
            raise FineTuneError(
                f"PruningConfig.method must be 'heads' or 'ffn'; got {config.method!r}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(self) -> PruningResult:
        """
        Execute the pruning pass and save the model.

        Returns:
            PruningResult with statistics and output path.
        """
        t0 = time.monotonic()

        original_count = _count_params(self.model)
        original_count = int(original_count)  # ensure plain int, not mock/tensor
        self.logger.info(
            f"Starting structured pruning — sparsity={self.config.sparsity:.0%} "
            f"method={self.config.method} params={int(original_count):,}"
        )

        if self.config.sparsity == 0.0:
            warnings.warn(
                "PruningConfig.sparsity=0.0 — no heads will be pruned.",
                UserWarning,
                stacklevel=2,
            )

        if self.config.method == "heads":
            zeroed, heads_per_layer = self._prune_attention_heads()
        else:
            zeroed, heads_per_layer = self._prune_ffn_neurons()

        # Save
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

        elapsed = time.monotonic() - t0

        # Ensure all values are plain Python primitives — never mock/tensor types
        zeroed_int: int = int(zeroed)
        original_int: int = int(original_count)
        achieved = zeroed_int / original_int if original_int > 0 else 0.0
        elapsed_f: float = float(elapsed)

        self.logger.info(
            f"Pruning complete — zeroed {zeroed_int:,} params "
            f"({achieved:.1%} sparsity) in {elapsed_f:.1f}s → {output_dir}"
        )

        return PruningResult(
            output_dir=output_dir,
            original_param_count=original_int,
            zeroed_param_count=zeroed_int,
            sparsity_achieved=achieved,
            heads_pruned_per_layer=heads_per_layer,
            pruning_time_seconds=elapsed_f,
        )

    # ------------------------------------------------------------------
    # Attention-head pruning
    # ------------------------------------------------------------------

    def _prune_attention_heads(self) -> Tuple[int, Dict[str, int]]:
        """Zero out lowest-importance attention head rows across all layers."""
        total_zeroed = 0
        heads_per_layer: Dict[str, int] = {}

        layers = self._find_transformer_layers()
        if not layers:
            self.logger.warning(
                "No transformer layers found — model structure not recognised. "
                "Pruning skipped."
            )
            return 0, {}

        for layer_name, layer in layers:
            attn = self._find_attention_module(layer)
            if attn is None:
                continue

            q_proj = self._find_query_proj(attn)
            if q_proj is None:
                continue

            num_heads = self._infer_num_heads(layer, attn)
            if num_heads <= 0:
                continue

            scores = _head_importance_scores(q_proj.weight, num_heads)
            n_prune = max(
                0,
                min(
                    int(num_heads * self.config.sparsity),
                    num_heads - self.config.min_heads_per_layer,
                ),
            )

            if n_prune == 0:
                heads_per_layer[layer_name] = 0
                continue

            # Indices of lowest-scoring heads
            sorted_heads = sorted(range(num_heads), key=lambda i: scores[i])
            prune_indices = sorted_heads[:n_prune]

            zeroed = _zero_head_rows(q_proj.weight, prune_indices, num_heads)
            total_zeroed += zeroed
            heads_per_layer[layer_name] = n_prune

            self.logger.debug(
                f"  {layer_name}: pruned {n_prune}/{num_heads} heads "
                f"({zeroed:,} params zeroed)"
            )

        return total_zeroed, heads_per_layer

    def _prune_ffn_neurons(self) -> Tuple[int, Dict[str, int]]:
        """Zero out lowest-importance FFN neuron rows (fc1/gate_proj weights)."""
        total_zeroed = 0
        neurons_per_layer: Dict[str, int] = {}

        layers = self._find_transformer_layers()
        if not layers:
            self.logger.warning("No transformer layers found — pruning skipped.")
            return 0, {}

        for layer_name, layer in layers:
            ffn_proj = self._find_ffn_proj(layer)
            if ffn_proj is None:
                continue

            num_neurons = ffn_proj.weight.shape[0]
            n_prune = max(0, int(num_neurons * self.config.sparsity))
            if n_prune == 0:
                neurons_per_layer[layer_name] = 0
                continue

            import torch
            scores = ffn_proj.weight.abs().mean(dim=1).tolist()
            sorted_neurons = sorted(range(num_neurons), key=lambda i: scores[i])
            prune_indices = sorted_neurons[:n_prune]

            zeroed = _zero_head_rows(ffn_proj.weight, prune_indices, num_neurons)
            total_zeroed += zeroed
            neurons_per_layer[layer_name] = n_prune

        return total_zeroed, neurons_per_layer

    # ------------------------------------------------------------------
    # Model-structure helpers
    # ------------------------------------------------------------------

    def _find_transformer_layers(self) -> List[Tuple[str, Any]]:
        """
        Return (name, module) pairs for each transformer block.
        Searches model.model.layers, model.transformer.h, model.model.decoder.layers, etc.
        """
        candidates = [
            ("model.layers",          lambda m: m.model.layers),
            ("transformer.h",         lambda m: m.transformer.h),
            ("model.decoder.layers",  lambda m: m.model.decoder.layers),
            ("encoder.layers",        lambda m: m.encoder.layers),
            ("bert.encoder.layer",    lambda m: m.bert.encoder.layer),
        ]
        for path, accessor in candidates:
            try:
                layers = accessor(self.model)
                return [(f"{path}[{i}]", layer) for i, layer in enumerate(layers)]
            except AttributeError:
                continue
        return []

    def _find_attention_module(self, layer: Any) -> Optional[Any]:
        for attr in self._ATTN_ATTR_CANDIDATES:
            mod = getattr(layer, attr, None)
            if mod is not None:
                return mod
        return None

    def _find_query_proj(self, attn: Any) -> Optional[Any]:
        for attr in self._QUERY_PROJ_CANDIDATES:
            mod = getattr(attn, attr, None)
            if mod is not None and hasattr(mod, "weight"):
                return mod
        return None

    def _find_ffn_proj(self, layer: Any) -> Optional[Any]:
        """Find the first FFN expansion projection in a layer."""
        candidates = (
            ("mlp", ("gate_proj", "fc1", "dense_h_to_4h", "wi")),
            ("feed_forward", ("w1", "fc1")),
        )
        for mlp_attr, proj_names in candidates:
            mlp = getattr(layer, mlp_attr, None)
            if mlp is None:
                continue
            for proj_name in proj_names:
                proj = getattr(mlp, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    return proj
        return None

    def _infer_num_heads(self, layer: Any, attn: Any) -> int:
        """
        Infer the number of attention heads from the layer's config or fallback attributes.
        """
        # Try model-level config first
        model_config = getattr(self.model, "config", None)
        if model_config is not None:
            for attr in ("num_attention_heads", "num_heads", "n_head"):
                val = getattr(model_config, attr, None)
                if val is not None:
                    return int(val)
        # Try attn module itself
        for attr in ("num_heads", "num_attention_heads", "n_heads"):
            val = getattr(attn, attr, None)
            if val is not None:
                return int(val)
        return 0
