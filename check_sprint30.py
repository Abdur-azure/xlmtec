"""Sprint 30 presence check — run from repo root: python check_sprint30.py"""
from pathlib import Path

SPRINT30_FILES = [
    "finetune_cli/trainers/wanda_pruner.py",
    "tests/test_wanda_pruner.py",
    "tests/test_wanda_cli.py",
    "examples/configs/wanda.yaml",
]

print("\n=== Sprint 30 file check ===")
all_present = True
for f in SPRINT30_FILES:
    exists = Path(f).exists()
    if not exists:
        all_present = False
    print(f"  {'✓' if exists else '✗ MISSING':<12} {f}")

print("\n=== Version check ===")
toml = Path("pyproject.toml").read_text()
for line in toml.splitlines():
    if line.strip().startswith("version"):
        print(f"  {line.strip()}  (expected: version = \"3.13.0\")")
        break

print("\n=== cli/main.py wanda command ===")
cli = Path("finetune_cli/cli/main.py").read_text()
print(f"  def wanda:       {'✓ present' if 'def wanda' in cli else '✗ MISSING'}")
print(f"  def prune:       {'✓ present' if 'def prune' in cli else '✗ MISSING'}")

print("\n=== core/types.py dataclasses ===")
types = Path("finetune_cli/core/types.py").read_text()
for cls in ["WandaConfig", "PruningConfig", "DistillationConfig", "FeatureDistillationConfig"]:
    print(f"  class {cls:<32} {'✓' if cls in types else '✗ MISSING'}")

print("\n=== trainers/__init__.py exports ===")
init = Path("finetune_cli/trainers/__init__.py").read_text()
for sym in ["WandaPruner", "WandaResult", "StructuredPruner", "PruningResult"]:
    print(f"  {sym:<32} {'✓' if sym in init else '✗ MISSING'}")

print("\n=== audit_repo.py Sprint 30 registered ===")
audit = Path("audit_repo.py").read_text()
for f in SPRINT30_FILES:
    fname = f.split("/")[-1]
    print(f"  {fname:<40} {'✓ registered' if fname in audit else '✗ NOT REGISTERED'}")

print()