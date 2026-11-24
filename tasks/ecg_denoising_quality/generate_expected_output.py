"""Regenerate expected_output.json from the actual reference implementation.

These metrics describe the reference solution on the synthetic scenarios. They
are informational evidence of reproducibility, NOT universal acceptance
criteria: the grader evaluates signal quality independently of the algorithm.
"""

import json
from pathlib import Path

from grader import aggregate, evaluate_output
from reference_solution import SCENARIOS, denoise, make_fixture

HERE = Path(__file__).resolve().parent


def main() -> None:
    scenarios = {}
    for name in SCENARIOS:
        fixture = make_fixture(name)
        output = denoise(fixture["noisy"], fixture["fs"])
        metrics = evaluate_output(
            fixture["clean"],
            fixture["noisy"],
            output,
            fixture["fs"],
            fixture["r_peaks"],
        )
        metrics["n_samples"] = len(fixture["noisy"])
        metrics["requested_snr_db"] = fixture["snr_db"]
        scenarios[name] = metrics
    payload = {
        "task": "ecg_denoising_quality",
        "note": "Reference metrics only; alternative valid methods may differ.",
        "fixtures": {
            "generator": "reference_solution.make_fixture",
            "seed_base": 20261124,
            "scenarios": list(SCENARIOS),
        },
        "scenarios": scenarios,
        "aggregate": aggregate(scenarios),
    }
    (HERE / "expected_output.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
