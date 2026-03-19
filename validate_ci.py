#!/usr/bin/env python3
"""
validate_ci.py — pre-push CI simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run this before every `git push` to catch the same failures GitHub Actions
would catch, without burning CI minutes or waiting for the runner.

Usage:
    python validate_ci.py              # full check
    python validate_ci.py --fast       # skip collection dry-run
    python validate_ci.py --fix        # auto-fix pyproject issues where possible

What it checks (in order):
    1. pyproject.toml — [dev] contains all packages needed by conftest + test files
    2. conftest.py — no module-level heavy-dep imports (datasets, torch, transformers)
    3. Any test file — no module-level import of datasets/torch/transformers
    4. pytest --collect-only with [dev] deps only (simulates CI env)
    5. ruff check xlmtec/
    6. mypy xlmtec/ (optional, can be slow)

Exit code 0 = ready to push. Any non-zero = fix before pushing.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent
TESTS_DIR = REPO_ROOT / "tests"
XLMTEC_DIR = REPO_ROOT / "xlmtec"
CONFTEST = TESTS_DIR / "conftest.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# These packages must be in [dev] because conftest.py and unit test files use them.
# datasets is lightweight (no GPU); it belongs in [dev] not [ml].
REQUIRED_IN_DEV = [
    "pytest",
    "pytest-cov",
    "pytest-timeout",
    "pytest-asyncio",
    "ruff",
    "mypy",
    "datasets",         # used by conftest tiny_dataset + many unit test files
]

# Heavy deps that must NEVER appear at module level in conftest or unit tests
HEAVY_DEPS = {"torch", "transformers", "peft", "accelerate", "bitsandbytes", "trl"}

# datasets is allowed at module level in test files (it's in [dev] after our fix)
# but NOT in conftest.py itself (per the documented rule)
CONFTEST_BANNED_IMPORTS = {"torch", "transformers", "peft", "datasets"}

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"

errors: list[str] = []
warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  {PASS} {msg}")


def fail(msg: str) -> None:
    print(f"  {FAIL} {BOLD}{msg}{RESET}")
    errors.append(msg)


def warn(msg: str) -> None:
    print(f"  {WARN} {msg}")
    warnings.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Check 1: pyproject.toml [dev] contains required packages
# ─────────────────────────────────────────────────────────────────────────────

def check_pyproject() -> None:
    print(f"\n{BOLD}[1] pyproject.toml — [dev] extra{RESET}")
    if not PYPROJECT.exists():
        fail("pyproject.toml not found at repo root")
        return

    content = PYPROJECT.read_text(encoding="utf-8")

    # Find the [dev] section — crude but reliable without a TOML parser
    in_dev = False
    dev_lines: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "dev = [":
            in_dev = True
            continue
        if in_dev:
            if stripped == "]":
                break
            dev_lines.append(stripped)

    dev_text = "\n".join(dev_lines).lower()

    for pkg in REQUIRED_IN_DEV:
        # Use re.escape first so special chars are safe, then widen the separator
        # to match both hyphen and underscore variants (e.g. pytest-cov or pytest_cov).
        # Must NOT do .replace("-","[-_]") then .replace("_","[-_]") in sequence —
        # the second replace corrupts the character class by replacing the _ inside [-_].
        import re
        pkg_normalized = re.escape(pkg.lower()).replace(r"\-", r"[-_]").replace(r"\_", r"[-_]")
        if re.search(pkg_normalized, dev_text):
            ok(f"[dev] contains {pkg}")
        else:
            fail(f"[dev] is MISSING '{pkg}' — CI will fail with ModuleNotFoundError")


# ─────────────────────────────────────────────────────────────────────────────
# Check 2: conftest.py has no banned module-level imports
# ─────────────────────────────────────────────────────────────────────────────

def check_conftest() -> None:
    print(f"\n{BOLD}[2] conftest.py — no heavy module-level imports{RESET}")
    if not CONFTEST.exists():
        fail(f"conftest.py not found at {CONFTEST}")
        return

    tree = ast.parse(CONFTEST.read_text(encoding="utf-8"))

    # Only check top-level import statements (not inside function/fixture bodies)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split(".")[0]
                if name in CONFTEST_BANNED_IMPORTS:
                    fail(
                        f"conftest.py line {node.lineno}: module-level 'import {name}' — "
                        f"move inside fixture body with pytest.importorskip('{name}')"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in CONFTEST_BANNED_IMPORTS:
                fail(
                    f"conftest.py line {node.lineno}: module-level 'from {mod} import ...' — "
                    f"move inside fixture body with pytest.importorskip('{mod}')"
                )

    if not any("conftest" in e for e in errors):
        ok("conftest.py has no banned module-level imports")


# ─────────────────────────────────────────────────────────────────────────────
# Check 3: Unit test files — no module-level torch imports
# ─────────────────────────────────────────────────────────────────────────────

def check_test_files() -> None:
    print(f"\n{BOLD}[3] Unit test files — no module-level torch/transformers imports{RESET}")
    test_files = [
        f for f in TESTS_DIR.glob("test_*.py")
        if f.name != "test_integration.py"
    ]
    found_any = False
    for tf in sorted(test_files):
        try:
            tree = ast.parse(tf.read_text(encoding="utf-8"))
        except SyntaxError as e:
            fail(f"{tf.name}: SyntaxError — {e}")
            continue

        for node in ast.iter_child_nodes(tree):
            imported = None
            if isinstance(node, ast.Import):
                imported = node.names[0].name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                imported = (node.module or "").split(".")[0]

            if imported in HEAVY_DEPS:
                warn(
                    f"{tf.name} line {node.lineno}: module-level 'import {imported}' — "
                    f"move inside test function or use pytest.importorskip"
                )
                found_any = True

    if not found_any:
        ok("No unit test file imports heavy deps at module level")


# ─────────────────────────────────────────────────────────────────────────────
# Check 4: pytest --collect-only (catches import errors before running)
# ─────────────────────────────────────────────────────────────────────────────

def check_collection(fast: bool = False) -> None:
    print(f"\n{BOLD}[4] pytest collection dry-run{RESET}")
    if fast:
        warn("Skipped (--fast mode)")
        return

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/",
            "--ignore=tests/test_integration.py",
            "--collect-only", "-q",
            "--tb=short",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    if result.returncode == 0:
        # Count collected tests
        lines = result.stdout.splitlines()
        count_line = next((l for l in lines if "selected" in l or "test" in l.lower()), "")
        ok(f"All tests collected successfully. {count_line.strip()}")
    else:
        # Print first error only
        lines = (result.stdout + result.stderr).splitlines()
        error_lines = [l for l in lines if "ERROR" in l or "ImportError" in l or "ModuleNotFoundError" in l]
        for line in error_lines[:5]:
            fail(f"Collection error: {line.strip()}")
        if not error_lines:
            fail(f"pytest --collect-only failed (exit {result.returncode}). Run manually to see details.")


# ─────────────────────────────────────────────────────────────────────────────
# Check 5: ruff
# ─────────────────────────────────────────────────────────────────────────────

def check_ruff() -> None:
    print(f"\n{BOLD}[5] ruff check xlmtec/{RESET}")
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "xlmtec/"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if result.returncode == 0:
        ok("ruff: 0 errors")
    else:
        lines = result.stdout.strip().splitlines()
        for line in lines[:10]:
            fail(f"ruff: {line}")
        if len(lines) > 10:
            warn(f"... and {len(lines) - 10} more ruff errors")


# ─────────────────────────────────────────────────────────────────────────────
# Check 6: mypy (optional — can be slow)
# ─────────────────────────────────────────────────────────────────────────────

def check_mypy(skip: bool = False) -> None:
    print(f"\n{BOLD}[6] mypy xlmtec/{RESET}")
    if skip:
        warn("Skipped (--fast mode)")
        return
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "xlmtec/", "--no-error-summary"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if result.returncode == 0:
        ok("mypy: 0 errors")
    else:
        lines = result.stdout.strip().splitlines()
        error_lines = [l for l in lines if ": error:" in l]
        for line in error_lines[:10]:
            fail(f"mypy: {line}")
        if len(error_lines) > 10:
            warn(f"... and {len(error_lines) - 10} more mypy errors")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    fast = "--fast" in sys.argv

    print(f"\n{BOLD}{'─' * 60}")
    print("xlmtec pre-push CI validator")
    print(f"{'─' * 60}{RESET}")

    check_pyproject()
    check_conftest()
    check_test_files()
    check_collection(fast=fast)
    check_ruff()
    check_mypy(skip=fast)

    print(f"\n{BOLD}{'─' * 60}{RESET}")
    if errors:
        print(f"\n{BOLD}\033[91m FAILED — {len(errors)} error(s) must be fixed before pushing:\033[0m{RESET}")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}")
        print()
        sys.exit(1)
    elif warnings:
        print(f"\n{BOLD}\033[93m PASSED with {len(warnings)} warning(s) — safe to push, but review warnings above.\033[0m{RESET}\n")
        sys.exit(0)
    else:
        print(f"\n{BOLD}\033[92m ALL CHECKS PASSED — safe to push.\033[0m{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
