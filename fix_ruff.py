#!/usr/bin/env python3
"""
fix_ruff.py — auto-fix all ruff issues in xlmtec/
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run once from the repo root. Fixes:
  W292  No newline at end of file       (ruff format)
  I001  Import block unsorted           (ruff format / --fix)
  E711  Comparison to None              (ruff check --fix)
  F401  Unused import in __init__.py    (already suppressed in pyproject.toml)
  ... and any other auto-fixable issues

Usage:
    python fix_ruff.py          # format + fix + report remaining
    python fix_ruff.py --check  # report only, no changes (dry-run)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
CHECK_ONLY = "--check" in sys.argv

BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def run(cmd: list[str], label: str) -> subprocess.CompletedProcess:
    print(f"\n{BOLD}{label}{RESET}")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines()[:30]:
            print(f"  {line}")
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines()[:10]:
            print(f"  [stderr] {line}")
    return result


def main() -> None:
    print(f"\n{BOLD}{'─'*60}")
    print("xlmtec ruff auto-fix")
    print(f"{'─'*60}{RESET}")

    if CHECK_ONLY:
        print(f"{YELLOW}  --check mode: reporting only, no changes{RESET}")

    # ── Step 1: ruff format (fixes W292, blank lines, trailing whitespace) ──
    fmt_cmd = [sys.executable, "-m", "ruff", "format", "xlmtec/", "tests/"]
    if CHECK_ONLY:
        fmt_cmd.append("--check")

    fmt = run(fmt_cmd, "Step 1: ruff format (fixes W292, blank lines, etc.)")
    if fmt.returncode == 0:
        print(f"  {GREEN}✓ ruff format: no changes needed{RESET}" if CHECK_ONLY
              else f"  {GREEN}✓ ruff format: done{RESET}")
    else:
        print(f"  {YELLOW}! ruff format: files need formatting (run without --check to fix){RESET}")

    # ── Step 2: ruff check --fix (fixes I001, E711, UP, etc.) ──────────────
    fix_cmd = [sys.executable, "-m", "ruff", "check", "xlmtec/", "tests/"]
    if not CHECK_ONLY:
        fix_cmd.append("--fix")

    fix = run(fix_cmd, "Step 2: ruff check --fix (fixes import order, comparisons, etc.)")
    if fix.returncode == 0:
        print(f"  {GREEN}✓ ruff check: 0 errors{RESET}")
    else:
        print(f"  {YELLOW}! ruff check: some errors remain (see above){RESET}")

    # ── Step 3: final check — count remaining errors ────────────────────────
    final = run(
        [sys.executable, "-m", "ruff", "check", "xlmtec/", "tests/", "--statistics"],
        "Step 3: final error count"
    )

    print(f"\n{BOLD}{'─'*60}{RESET}")
    if final.returncode == 0:
        print(f"\n{GREEN}{BOLD}  ✓ ALL RUFF ERRORS FIXED — ready to push.{RESET}\n")
    else:
        remaining = [l for l in final.stdout.splitlines() if l.strip() and not l.startswith("  ")]
        count = sum(int(l.split()[0]) for l in remaining if l.split() and l.split()[0].isdigit())
        print(f"\n{YELLOW}{BOLD}  {count} ruff error(s) remain that need manual fixes:{RESET}")
        print("\n  Run:  ruff check xlmtec/ --output-format=text")
        print("  Then fix each remaining error manually.\n")
        print("\n  Common manual fixes:")
        print("    E501  Line too long   → break the line or add # noqa: E501 (already ignored by config)")
        print("    F821  Undefined name  → add missing import")
        print("    E741  Ambiguous name  → rename variable (l → ln, O → obj, etc.)")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
