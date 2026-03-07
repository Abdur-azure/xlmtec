"""
finetune_cli.py — DEPRECATED (v1)

This file is the original monolithic v1 script. It has been superseded by the
v2 modular architecture in the `finetune_cli/` package.

Migration guide
---------------
v1 command:
    python finetune_cli.py

v2 equivalents:
    python -m finetune_cli.cli train --model gpt2 --dataset ./data.jsonl
    python -m finetune_cli.cli train --config examples/configs/lora_gpt2.yaml
    python -m finetune_cli.cli evaluate --model-path ./output --dataset ./data.jsonl
    python -m finetune_cli.cli benchmark gpt2 ./output --dataset ./data.jsonl
    python -m finetune_cli.cli upload ./output username/my-model

Install the v2 CLI for the `finetune-cli` shell command:
    pip install -e .
    finetune-cli --help

This file will be removed in a future release.
"""

import sys
import warnings


def main():
    warnings.warn(
        "\n\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║  finetune_cli.py (v1) is DEPRECATED                     ║\n"
        "║                                                          ║\n"
        "║  Please migrate to the v2 CLI:                          ║\n"
        "║    python -m finetune_cli.cli --help                    ║\n"
        "║                                                          ║\n"
        "║  Or install and use the shell command:                  ║\n"
        "║    pip install -e .                                      ║\n"
        "║    finetune-cli --help                                   ║\n"
        "║                                                          ║\n"
        "║  See CHANGELOG.md for full migration guide.             ║\n"
        "╚══════════════════════════════════════════════════════════╝\n",
        DeprecationWarning,
        stacklevel=2,
    )
    print("Exiting. Use `python -m finetune_cli.cli --help` instead.")
    sys.exit(0)


if __name__ == "__main__":
    main()
