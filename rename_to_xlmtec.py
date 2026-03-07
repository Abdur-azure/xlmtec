"""
rename_to_xlmtec.py
~~~~~~~~~~~~~~~~~~~
One-shot rename script: replaces every remaining `xlmtec` / `xlmtec`
reference in source files with `xlmtec`.

Run once from the repo root AFTER renaming the package folder:
    python rename_to_xlmtec.py

Safe to re-run — skips files with no changes.
"""

import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Replacement rules — applied in order
# ---------------------------------------------------------------------------
REPLACEMENTS = [
    # Python import paths (most specific first)
    ("xlmtec.cli.main:main", "xlmtec.cli.main:main"),
    ("xlmtec.cli.main:app", "xlmtec.cli.main:app"),
    # Module references in strings (patch targets, logging, etc.)
    ("xlmtec.", "xlmtec."),
    # Bare package name (imports, __name__ checks, folder refs in strings)
    ("xlmtec", "xlmtec"),
    # CLI command name in docs/strings
    ("xlmtec", "xlmtec"),
]

# File extensions to process
TEXT_EXTENSIONS = {
    ".py", ".md", ".toml", ".yaml", ".yml", ".txt", ".rst", ".cfg", ".ini",
}

# Directories to skip entirely
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "env", ".mypy_cache",
             ".ruff_cache", "dist", "build", "*.egg-info"}


def should_skip_dir(dirname: str) -> bool:
    return dirname in SKIP_DIRS or dirname.endswith(".egg-info")


def process_file(path: Path) -> bool:
    """Apply all replacements to a file. Returns True if file was changed."""
    try:
        original = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return False

    text = original
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    root = Path(".")
    changed = []
    skipped = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs in-place so os.walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]

        for filename in filenames:
            filepath = Path(dirpath) / filename
            if filepath.suffix not in TEXT_EXTENSIONS:
                continue

            if process_file(filepath):
                changed.append(str(filepath))
            else:
                skipped.append(str(filepath))

    print(f"\n{'='*55}")
    print(f"  RENAME COMPLETE — xlmtec → xlmtec")
    print(f"{'='*55}")
    print(f"  Files changed : {len(changed)}")
    print(f"  Files unchanged: {len(skipped)}")

    if changed:
        print("\n  Changed files:")
        for f in sorted(changed):
            print(f"    ✓  {f}")

    # Verify no stragglers
    print("\n  Verifying no remaining references...")
    remaining = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        for filename in filenames:
            filepath = Path(dirpath) / filename
            if filepath.suffix not in TEXT_EXTENSIONS:
                continue
            try:
                content = filepath.read_text(encoding="utf-8")
                if "xlmtec" in content or "xlmtec" in content:
                    remaining.append(str(filepath))
            except (UnicodeDecodeError, PermissionError):
                pass

    if remaining:
        print("\n  ⚠  Still contains old references:")
        for f in remaining:
            print(f"    ✗  {f}")
    else:
        print("  ✓  Zero remaining references to xlmtec or xlmtec\n")

    print("\n  Next steps:")
    print("    pip install -e .")
    print("    xlmtec --help")
    print("    pytest tests/ --co -q")
    print()


if __name__ == "__main__":
    main()