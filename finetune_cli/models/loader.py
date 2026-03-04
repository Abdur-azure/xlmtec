"""Model loading utilities."""

from typing import List, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.types import ModelConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)

_PATTERNS = [
    ["q_proj", "v_proj", "k_proj", "o_proj"],
    ["query", "value", "key", "dense"],
    ["c_attn", "c_proj"],
    ["qkv_proj", "out_proj"],
    ["fc1", "fc2"],
    ["up_proj", "down_proj", "gate_proj"],
]


def detect_target_modules(model: PreTrainedModel) -> List[str]:
    """Auto-detect LoRA target modules from model architecture."""
    leaf_names = {
        name.split(".")[-1]
        for name, module in model.named_modules()
        if not list(module.children())
    }
    for pattern in _PATTERNS:
        matched = [n for n in pattern if n in leaf_names]
        if len(matched) >= 2:
            logger.info(f"Auto-detected target modules: {matched}")
            return matched
    # Fallback
    keywords = ["proj", "fc", "dense", "query", "key", "value"]
    candidates = sorted(n for n in leaf_names if any(k in n.lower() for k in keywords))
    return candidates[:4] if candidates else ["q_proj", "v_proj"]


def load_model_and_tokenizer(config: ModelConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer from HuggingFace."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Loading tokenizer: {config.name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.name,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {config.name}")
    load_kwargs = dict(trust_remote_code=config.trust_remote_code)

    if config.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif config.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
    elif config.torch_dtype:
        load_kwargs["torch_dtype"] = config.torch_dtype

    if config.device.value != "auto":
        load_kwargs["device_map"] = config.device.value
    else:
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.name, **load_kwargs)
    model.train()
    return model, tokenizer