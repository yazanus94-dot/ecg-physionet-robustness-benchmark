"""Grader for the ECG Arrhythmia Robustness Benchmark task (expert task).

Validates the reference solution against `expected_output.json`:

1. Predictions/truth match the expected output (determinism + correctness).
2. Metrics are within [0, 1] and are non-NaN.
3. One prediction per beat.

Usage: `python grader.py` (from the task directory).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from reference_solution import make_fixture, solve_fixture


def run_reference() -> dict:
    return solve_fixture(make_fixture())


def main() -> int:
    expected_path = Path(__file__).parent / "expected_output.json"
    if not expected_path.exists():
        print("expected_output.json not found -- run: python generate_expected_output.py")
        return 1
    expected = json.loads(expected_path.read_text())
    out = run_reference()

    checks: list[tuple[str, bool]] = []

    checks.append((
        "predictions match expected_output.json (deterministic reference)",
        out["predictions"] == expected["predictions"],
    ))
    checks.append((
        "truth matches expected_output.json",
        out["truth"] == expected["truth"],
    ))
    m = out["metrics"]
    metric_sanity = all(
        0.0 <= m[k] <= 1.0 for k in ("accuracy", "f1_macro", "f1_weighted", "balanced_accuracy")
    ) and not any(
        math.isnan(v) for v in (m["accuracy"], m["f1_macro"], m["f1_weighted"], m["balanced_accuracy"])
    )
    checks.append(("metrics within [0, 1] and finite", metric_sanity))
    n_pred, n_beats = len(out["predictions"]), len(out["beats"])
    checks.append((
        f"one prediction per beat (got {n_pred} predictions for {n_beats} beats)",
        n_pred == n_beats > 0,
    ))
    truth_ok = all(
        t == a for t, a in zip(out["truth"], [b["aami_class"] for b in out["beats"]])
    )
    checks.append(("predictions are strings from the AAMI class set",
                   all(p in {"N", "S", "V", "F", "Q"} for p in out["predictions"])))
    checks.append(("truth matches the beats' AAMI classes", truth_ok))

    ok = True
    for name, passed in checks:
        print(("PASS" if passed else "FAIL") + f": {name}")
        ok = ok and passed
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
