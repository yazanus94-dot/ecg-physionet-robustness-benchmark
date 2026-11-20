"""Make `reference_solution`, `grader`, `generate_expected_output` importable
when running pytest from the repository root (tasks/* aren't packages)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
