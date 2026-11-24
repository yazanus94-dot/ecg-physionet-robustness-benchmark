"""Regenerate expected_output.json from the deterministic reference fixture."""

import json
from pathlib import Path

from grader import evaluate_output
from reference_solution import denoise, make_fixture

HERE = Path(__file__).resolve().parent


def main() -> None:
    fixture = make_fixture()
    output = denoise(fixture["noisy"], fixture["fs"])
    metrics = evaluate_output(
        fixture["clean"], fixture["noisy"], output, fixture["fs"], fixture["r_peaks"]
    )
    payload = {
        "task": "ecg_denoising_quality",
        "fixture": {
            "generator": "reference_solution.make_fixture",
            "seed": 20261124,
            "fs_hz": fixture["fs"],
            "seconds": 10,
            "requested_input_snr_db": fixture["snr_db"],
            "n_r_peaks": len(fixture["r_peaks"]),
        },
        "metrics": metrics,
    }
    (HERE / "expected_output.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
