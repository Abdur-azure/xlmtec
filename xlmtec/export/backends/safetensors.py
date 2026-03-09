"""
xlmtec.export.backends.safetensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Export a model to safetensors format.

safetensors is a safe, fast serialisation format from HuggingFace.
It avoids the pickle-based security risks of .bin/.pt files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SafetensorsResult:
    output_path: Path
    file_size_mb: float
    num_files: int


def export_safetensors(
    model_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> SafetensorsResult:
    """Convert a HuggingFace model directory to safetensors format.

    Args:
        model_dir:  Source model directory (from training or merge).
        output_dir: Destination directory for safetensors files.
        dry_run:    If True, validate only — do not write files.

    Returns:
        SafetensorsResult with path and size info.

    Raises:
        FileNotFoundError: If model_dir does not exist.
        ImportError: If safetensors or transformers are not installed.
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            f"safetensors export requires transformers and safetensors.\n"
            f"Install with: pip install xlmtec[ml]\n"
            f"Original error: {exc}"
        ) from exc

    if dry_run:
        return SafetensorsResult(
            output_path=output_dir / "model.safetensors",
            file_size_mb=0.0,
            num_files=0,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # Save model weights as safetensors
    state_dict = model.state_dict()
    out_file = output_dir / "model.safetensors"
    save_file(state_dict, str(out_file))

    # Save tokenizer alongside
    tokenizer.save_pretrained(str(output_dir))

    size_mb = out_file.stat().st_size / (1024 * 1024)
    num_files = len(list(output_dir.iterdir()))

    return SafetensorsResult(
        output_path=out_file,
        file_size_mb=size_mb,
        num_files=num_files,
    )