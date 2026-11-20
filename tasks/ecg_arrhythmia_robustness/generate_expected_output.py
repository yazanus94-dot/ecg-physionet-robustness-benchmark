"""Regenerate `expected_output.json` from the deterministic fixture.

Usage: `python generate_expected_output.py`
"""

from __future__ import annotations

import json
from pathlib import Path

from reference_solution import make_fixture, solve_fixture


def generate() -> dict:
    """Run the reference solution on the fixture and return the output."""
    return solve_fixture(make_fixture())


def main() -> int:
    out = generate()
    p = Path(__file__).parent / "expected_output.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
