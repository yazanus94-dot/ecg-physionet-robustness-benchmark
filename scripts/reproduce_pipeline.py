#!/usr/bin/env python
"""One-command reproducible runner for ECG-ROBUST.

Usage:
    python scripts/reproduce_pipeline.py

Steps:
1. pip install -e . (installs the ecg_robust package + pinned deps from requirements.txt).
2. Runs the full pipeline: `python -m ecg_robust.pipeline`.
3. Runs the pytest suite.

Artifacts land in data/raw/, data/features/, reports/figures/,
reports/results/ and reports/TECHNICAL_REPORT.md.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": "src"},
                          check=False).returncode


def main() -> int:
    code = run([sys.executable, "-m", "pip", "install", "-e", "."])
    if code != 0:
        return code
    code = run([sys.executable, "-m", "ecg_robust.pipeline"])
    if code != 0:
        return code
    return run([sys.executable, "-m", "pytest", "-q"])


if __name__ == "__main__":
    raise SystemExit(main())
