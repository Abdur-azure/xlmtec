"""
generate_sample_data.py — create sample datasets for all example configs.

No network required. Pure Python + stdlib only.

Usage:
    python examples/generate_sample_data.py

Outputs:
    data/sample.jsonl          — 500 rows, causal LM format  (lora, qlora, full fine-tuning)
    data/instructions.jsonl    — 300 rows, alpaca format      (instruction tuning)
    data/dpo_sample.jsonl      — 200 rows, DPO format         (prompt, chosen, rejected)

Run this once before executing any example config:
    lmtool train --config examples/configs/lora_gpt2.yaml
    lmtool train --config examples/configs/instruction_tuning.yaml
    lmtool train --config examples/configs/full_finetuning.yaml
    lmtool train --config examples/configs/dpo.yaml
"""

import json
import random
import sys
from pathlib import Path

# Always run from repo root
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)

# ============================================================================
# CAUSAL LM SAMPLE DATA
# ============================================================================

_TOPICS = [
    "machine learning", "neural networks", "natural language processing",
    "computer vision", "reinforcement learning", "transformers", "fine-tuning",
    "gradient descent", "backpropagation", "attention mechanisms",
    "Python programming", "data preprocessing", "model evaluation",
    "transfer learning", "embeddings", "tokenization", "regularization",
    "optimisation", "batch normalisation", "dropout",
]

_TEMPLATES = [
    "{topic} is a fundamental concept in modern AI research.",
    "Understanding {topic} requires a solid mathematical foundation.",
    "Practitioners use {topic} to solve real-world problems efficiently.",
    "Recent advances in {topic} have transformed the field significantly.",
    "The core idea behind {topic} is surprisingly elegant and intuitive.",
    "Many researchers focus their careers on improving {topic}.",
    "Applying {topic} to production systems requires careful engineering.",
    "Benchmarks for {topic} have become increasingly rigorous over time.",
    "Open-source tools have democratised access to {topic} research.",
    "The history of {topic} spans several decades of innovation.",
]


def generate_causal_lm_samples(n: int = 500) -> list:
    samples = []
    for i in range(n):
        topic = random.choice(_TOPICS)
        template = random.choice(_TEMPLATES)
        text = template.format(topic=topic)
        # Occasionally chain two sentences for variety
        if i % 3 == 0:
            topic2 = random.choice(_TOPICS)
            template2 = random.choice(_TEMPLATES)
            text += " " + template2.format(topic=topic2)
        samples.append({"text": text})
    return samples


# ============================================================================
# INSTRUCTION / ALPACA SAMPLE DATA
# ============================================================================

_INSTRUCTIONS = [
    ("Explain what {topic} is in one sentence.",
     "{topic} is a technique used in AI to {action}."),
    ("Give an example of {topic} in practice.",
     "A common example of {topic} is when a model {action}."),
    ("What are the main benefits of {topic}?",
     "The main benefits of {topic} include improved {benefit} and faster {speed}."),
    ("How does {topic} differ from traditional approaches?",
     "Unlike traditional methods, {topic} allows models to {action} without {constraint}."),
    ("When should you use {topic}?",
     "You should use {topic} when you need to {action} with limited {resource}."),
]

_ACTIONS = [
    "learn from data", "generalise to new examples", "adapt to new domains",
    "reduce overfitting", "improve performance", "process sequences",
    "extract features", "generate text", "classify inputs",
]

_BENEFITS = ["accuracy", "efficiency", "generalisation", "scalability"]
_SPEEDS = ["convergence", "inference", "training"]
_CONSTRAINTS = ["labelled data", "large compute", "manual feature engineering"]
_RESOURCES = ["data", "compute", "memory", "time"]


def generate_instruction_samples(n: int = 300) -> list:
    samples = []
    for i in range(n):
        topic = random.choice(_TOPICS)
        tmpl_instruction, tmpl_response = random.choice(_INSTRUCTIONS)
        instruction = tmpl_instruction.format(topic=topic)
        response = tmpl_response.format(
            topic=topic,
            action=random.choice(_ACTIONS),
            benefit=random.choice(_BENEFITS),
            speed=random.choice(_SPEEDS),
            constraint=random.choice(_CONSTRAINTS),
            resource=random.choice(_RESOURCES),
        )
        # ~30% of samples include an optional input context
        input_ctx = ""
        if i % 3 == 0:
            input_ctx = f"Context: This question is about {topic} in the context of deep learning."
        samples.append({
            "instruction": instruction,
            "input": input_ctx,
            "response": response,
        })
    return samples



# ============================================================================
# DPO SAMPLE DATA
# ============================================================================

_QUESTIONS = [
    "What is the best way to learn {topic}?",
    "How do you explain {topic} to a beginner?",
    "What are the key challenges in {topic}?",
    "Can you summarise the main ideas behind {topic}?",
    "What resources would you recommend for {topic}?",
]

_CHOSEN_TEMPLATES = [
    "A great way to learn {topic} is to start with the fundamentals, then "
    "build hands-on projects that apply the concepts progressively.",
    "To explain {topic} to a beginner: focus on the intuition first, use "
    "concrete examples, and avoid jargon until the core idea is clear.",
    "The key challenges in {topic} include data quality, compute constraints, "
    "and the gap between research benchmarks and real-world performance.",
    "{topic} is built on a few core ideas: representation learning, "
    "optimisation via gradient descent, and generalisation to unseen data.",
    "For {topic}, I'd recommend starting with fast.ai or the original papers, "
    "combined with a hands-on coding project to reinforce understanding.",
]

_REJECTED_TEMPLATES = [
    "Just Google it. There are plenty of tutorials online about {topic}.",
    "{topic} is very complex. You probably won't understand it without a PhD.",
    "I'm not sure. {topic} is hard to explain.",
    "{topic} is basically just statistics. Nothing special.",
    "Any book works. It doesn't really matter which one you pick for {topic}.",
]


def generate_dpo_samples(n: int = 200) -> list:
    samples = []
    for i in range(n):
        topic = random.choice(_TOPICS)
        question = random.choice(_QUESTIONS).format(topic=topic)
        chosen = random.choice(_CHOSEN_TEMPLATES).format(topic=topic)
        rejected = random.choice(_REJECTED_TEMPLATES).format(topic=topic)
        samples.append({
            "prompt": question,
            "chosen": chosen,
            "rejected": rejected,
        })
    return samples

# ============================================================================
# WRITE FILES
# ============================================================================

def write_jsonl(path: Path, records: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  ✓  {path.relative_to(ROOT)}  ({len(records)} rows)")


def main() -> None:
    print(f"\nGenerating sample datasets → {DATA_DIR.relative_to(ROOT)}/\n")

    causal = generate_causal_lm_samples(500)
    write_jsonl(DATA_DIR / "sample.jsonl", causal)

    instructions = generate_instruction_samples(300)
    write_jsonl(DATA_DIR / "instructions.jsonl", instructions)

    dpo = generate_dpo_samples(200)
    write_jsonl(DATA_DIR / "dpo_sample.jsonl", dpo)

    print(
        "\nDone. Run an example:\n"
        "  lmtool train --config examples/configs/lora_gpt2.yaml\n"
        "  lmtool train --config examples/configs/instruction_tuning.yaml\n"
        "  lmtool train --config examples/configs/dpo.yaml\n"
    )


if __name__ == "__main__":
    main()
