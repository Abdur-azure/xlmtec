"""
xlmtec.export.formats
~~~~~~~~~~~~~~~~~~~~~~
ExportFormat enum and per-format metadata (dependencies, file extension, install hint).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExportFormat(str, Enum):
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    GGUF = "gguf"

    @classmethod
    def from_str(cls, value: str) -> "ExportFormat":
        try:
            return cls(value.lower().strip())
        except ValueError:
            available = ", ".join(f.value for f in cls)
            raise ValueError(f"Unknown export format {value!r}. Available: {available}")


@dataclass(frozen=True)
class FormatMeta:
    name: str
    extension: str
    description: str
    pip_extra: str          # xlmtec[<extra>] to install
    required_packages: list[str]
    quantize_options: list[str]


FORMAT_META: dict[ExportFormat, FormatMeta] = {
    ExportFormat.SAFETENSORS: FormatMeta(
        name="safetensors",
        extension=".safetensors",
        description="Safe, fast model serialisation format by HuggingFace.",
        pip_extra="ml",
        required_packages=["safetensors", "transformers"],
        quantize_options=[],
    ),
    ExportFormat.ONNX: FormatMeta(
        name="onnx",
        extension=".onnx",
        description="Open Neural Network Exchange — deploy to ONNX Runtime, TensorRT, etc.",
        pip_extra="onnx",
        required_packages=["optimum", "onnx", "onnxruntime"],
        quantize_options=["fp16", "int8"],
    ),
    ExportFormat.GGUF: FormatMeta(
        name="gguf",
        extension=".gguf",
        description="GGUF format for llama.cpp — run locally with Ollama, LM Studio, etc.",
        pip_extra="ml",
        required_packages=["transformers"],
        quantize_options=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"],
    ),
}


def get_format_meta(fmt: ExportFormat) -> FormatMeta:
    return FORMAT_META[fmt]