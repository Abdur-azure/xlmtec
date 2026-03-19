#!/usr/bin/env python3
"""
fix_duplicate_key.py — fixes the duplicate 'disable_error_code' key in pyproject.toml
Run from your repo root: python fix_duplicate_key.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent
PYPROJECT = ROOT / "pyproject.toml"

if not PYPROJECT.exists():
    print("✗ pyproject.toml not found — run this from the repo root")
    raise SystemExit(1)

content = PYPROJECT.read_text(encoding="utf-8")

# Find every occurrence of disable_error_code under [tool.mypy]
occurrences = [m.start() for m in re.finditer(r'disable_error_code\s*=.*', content)]

if len(occurrences) == 0:
    print("✗ No disable_error_code found — nothing to fix")
    raise SystemExit(1)

if len(occurrences) == 1:
    print("✓ Only one disable_error_code found — no duplicate, already clean")
    raise SystemExit(0)

print(f"Found {len(occurrences)} occurrences of disable_error_code — removing duplicates, keeping first")

# Keep only the first occurrence, remove all subsequent ones (including their newline)
# Remove lines that are duplicates (lines 2..n)
lines = content.splitlines(keepends=True)
seen = False
new_lines = []
for line in lines:
    if re.match(r'\s*disable_error_code\s*=', line):
        if not seen:
            new_lines.append(line)
            seen = True
        else:
            # skip the duplicate line
            print(f"  Removing duplicate: {line.rstrip()}")
    else:
        new_lines.append(line)

new_content = "".join(new_lines)
PYPROJECT.write_text(new_content, encoding="utf-8")
print("✓ pyproject.toml fixed")
print("\nNext step — confirm it parses cleanly:")
print("  ruff check xlmtec/")