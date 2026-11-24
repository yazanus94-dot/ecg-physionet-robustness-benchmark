"""Regenerate expected_output.json from the deterministic reference solution."""

import json
from pathlib import Path

from reference_solution import FEATURE_COLUMNS, make_fixture, patient_independent_oof

HERE = Path(__file__).resolve().parent


def main() -> None:
    frame = make_fixture()
    output = patient_independent_oof(
        frame[list(FEATURE_COLUMNS)],
        frame["label"].to_numpy(),
        frame["patient_id"].to_numpy(),
    )
    payload = {
        "task": "arrhythmia_classification",
        "fixture": {
            "generator": "reference_solution.make_fixture",
            "seed": 424242,
            "n_patients": 12,
            "beats_per_patient": 20,
            "classes": ["N", "S", "V", "F"],
        },
        "metrics": output["metrics"],
    }
    (HERE / "expected_output.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
