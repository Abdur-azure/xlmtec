"""
xlmtec.export.backends.gguf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Export a HuggingFace model to GGUF format for llama.cpp.

GGUF models run locally with Ollama, LM Studio, llama.cpp, etc.

Two-step process:
  1. Convert HF model → GGUF using llama.cpp convert_hf_to_gguf.py
  2. Optionally quantise with llama.cpp quantize binary

Requires llama.cpp to be cloned and built on the system, or
the gguf-py package for conversion-only.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GgufResult:
    output_path: Path
    file_size_mb: float
    quantize_type: str


VALID_QUANTIZE = {"q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"}


def export_gguf(
    model_dir: Path,
    output_dir: Path,
    quantize: str = "q4_0",
    llama_cpp_dir: Path | None = None,
    dry_run: bool = False,
) -> GgufResult:
    """Export a model to GGUF format using llama.cpp convert script.

    Args:
        model_dir:      Source model directory.
        output_dir:     Destination for .gguf file.
        quantize:       Quantisation type (default: q4_0).
        llama_cpp_dir:  Path to llama.cpp repo (auto-detected if None).
        dry_run:        Validate only, do not write files.

    Returns:
        GgufResult with path, size, and quantisation info.

    Raises:
        FileNotFoundError: If model_dir or llama.cpp convert script not found.
        ValueError: If quantize type is invalid.
        RuntimeError: If the conversion subprocess fails.
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if quantize not in VALID_QUANTIZE:
        raise ValueError(
            f"Invalid GGUF quantise type {quantize!r}.\n"
            f"Choose from: {', '.join(sorted(VALID_QUANTIZE))}"
        )

    # Try to find convert script
    convert_script = _find_convert_script(llama_cpp_dir)

    if dry_run:
        return GgufResult(
            output_path=output_dir / f"model-{quantize}.gguf",
            file_size_mb=0.0,
            quantize_type=quantize,
        )

    if convert_script is None:
        raise FileNotFoundError(
            "llama.cpp convert script not found.\n"
            "Clone llama.cpp and pass --llama-cpp-dir, or install gguf:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            "  pip install gguf"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"model-{quantize}.gguf"

    # Step 1: Convert to f16 GGUF
    f16_file = output_dir / "model-f16.gguf"
    _run_convert(convert_script, model_dir, f16_file)

    # Step 2: Quantise (skip if f16/f32 requested)
    if quantize in ("f16", "f32"):
        f16_file.rename(out_file)
    else:
        _run_quantize(llama_cpp_dir, f16_file, out_file, quantize)
        if f16_file.exists():
            f16_file.unlink()

    size_mb = out_file.stat().st_size / (1024 * 1024) if out_file.exists() else 0.0
    return GgufResult(output_path=out_file, file_size_mb=size_mb, quantize_type=quantize)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_convert_script(llama_cpp_dir: Path | None) -> Path | None:
    """Search common locations for llama.cpp convert script."""
    candidates = []
    if llama_cpp_dir:
        candidates += [
            llama_cpp_dir / "convert_hf_to_gguf.py",
            llama_cpp_dir / "convert.py",
        ]
    # Common install locations
    candidates += [
        Path("llama.cpp/convert_hf_to_gguf.py"),
        Path("llama.cpp/convert.py"),
        Path.home() / "llama.cpp/convert_hf_to_gguf.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _run_convert(script: Path, model_dir: Path, out_file: Path) -> None:
    cmd = [sys.executable, str(script), str(model_dir), "--outfile", str(out_file), "--outtype", "f16"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed:\n{result.stderr}")


def _run_quantize(llama_cpp_dir: Path | None, in_file: Path, out_file: Path, qtype: str) -> None:
    quantize_bin = _find_quantize_bin(llama_cpp_dir)
    if quantize_bin is None:
        raise FileNotFoundError(
            "llama.cpp quantize binary not found. Build llama.cpp first:\n"
            "  cd llama.cpp && make quantize"
        )
    cmd = [str(quantize_bin), str(in_file), str(out_file), qtype.upper()]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GGUF quantisation failed:\n{result.stderr}")


def _find_quantize_bin(llama_cpp_dir: Path | None) -> Path | None:
    candidates = []
    if llama_cpp_dir:
        candidates += [
            llama_cpp_dir / "quantize",
            llama_cpp_dir / "quantize.exe",
            llama_cpp_dir / "build/bin/quantize",
        ]
    candidates += [
        Path("llama.cpp/quantize"),
        Path("llama.cpp/build/bin/quantize"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None