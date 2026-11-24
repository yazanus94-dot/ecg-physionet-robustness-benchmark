"""Automated grader for the ECG denoising benchmark task.

Usage:
    python grader.py [path/to/candidate.py]

The grader validates actual signal quality and correctness, not merely that
code executes. It exits 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from reference_solution import denoise as reference_denoise
from reference_solution import make_fixture
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent


def _load_candidate(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("candidate_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snr_db(clean: np.ndarray, estimated: np.ndarray) -> float:
    residual = clean - estimated
    power = float(np.mean(residual**2))
    if power == 0.0:
        return float("inf")
    return float(10 * np.log10(np.mean(clean**2) / power))


def detection_f1(
    signal: np.ndarray, fs: int, peaks: np.ndarray, tolerance_s: float = 0.15
) -> float:
    threshold = 0.6 * float(np.max(np.abs(signal)))
    detected, _ = find_peaks(signal, height=threshold, distance=int(0.25 * fs))
    match_window = int(tolerance_s * fs)
    matched = 0
    used: set[int] = set()
    for peak in detected:
        distances = np.abs(peaks - peak)
        order = np.argsort(distances)
        for idx in order:
            if idx in used:
                continue
            if distances[idx] <= match_window:
                used.add(int(idx))
                matched += 1
            break
    precision = matched / len(detected) if len(detected) else 0.0
    recall = matched / len(peaks) if len(peaks) else 0.0
    return (
        float(2 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0
    )


def evaluate_output(
    clean: np.ndarray, noisy: np.ndarray, output: np.ndarray, fs: int, peaks: np.ndarray
) -> dict:
    return {
        "snr_input_db": round(snr_db(clean, noisy), 4),
        "snr_output_db": round(snr_db(clean, output), 4),
        "snr_improvement_db": round(snr_db(clean, output) - snr_db(clean, noisy), 4),
        "correlation": round(float(np.corrcoef(clean, output)[0, 1]), 4),
        "peak_f1": round(detection_f1(output, fs, peaks), 4),
    }


def _check(name: str, ok: bool, detail: str, results: dict) -> None:
    results[name] = {"passed": bool(ok), "detail": detail}


def grade(candidate_path: Path) -> dict:
    candidate = _load_candidate(candidate_path)
    fixture = make_fixture()
    clean, noisy, fs, peaks = (
        fixture["clean"],
        fixture["noisy"],
        fixture["fs"],
        fixture["r_peaks"],
    )
    results: dict[str, dict] = {}

    try:
        output = np.asarray(candidate.denoise(noisy, fs), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - grader reports any candidate failure
        _check("runs", False, f"denoise raised {type(exc).__name__}: {exc}", results)
        return {"checks": results, "metrics": {}, "passed": False}

    _check("runs", True, "denoise returned an array", results)
    _check(
        "shape",
        output.shape == noisy.shape,
        f"expected {noisy.shape}, got {output.shape}",
        results,
    )
    _check("finite", bool(np.isfinite(output).all()), "all samples finite", results)

    metrics = evaluate_output(clean, noisy, output, fs, peaks)
    _check(
        "snr_improvement",
        metrics["snr_improvement_db"] >= 4.0,
        f"improvement {metrics['snr_improvement_db']} dB, need >= 4 dB",
        results,
    )
    _check(
        "morphology",
        metrics["correlation"] >= 0.90,
        f"correlation {metrics['correlation']}, need >= 0.90",
        results,
    )
    _check(
        "peak_detection",
        metrics["peak_f1"] >= 0.85,
        f"peak F1 {metrics['peak_f1']}, need >= 0.85",
        results,
    )

    reference = reference_denoise(noisy, fs)
    _check(
        "reference_consistency",
        bool(np.allclose(output, reference, rtol=1e-6, atol=1e-8)),
        "output matches the reference filter within tolerance",
        results,
    )

    first = candidate.denoise(noisy, fs)
    second = candidate.denoise(noisy, fs)
    _check(
        "determinism",
        bool(np.array_equal(np.asarray(first), np.asarray(second))),
        "two identical calls give bitwise-identical output",
        results,
    )

    bad_inputs = {
        "empty": (np.array([]), fs),
        "2d": (np.zeros((10, 2)), fs),
        "nan": (np.full(120, np.nan), fs),
        "bad_fs": (np.ones(120), 0),
        "short": (np.ones(10), fs),
    }
    for label, (sig, rate) in bad_inputs.items():
        try:
            candidate.denoise(sig, rate)
            _check(f"invalid_{label}", False, "expected ValueError", results)
        except ValueError:
            _check(f"invalid_{label}", True, "ValueError raised", results)
        except Exception as exc:  # noqa: BLE001
            _check(
                f"invalid_{label}",
                False,
                f"wrong exception {type(exc).__name__}",
                results,
            )

    scaled = candidate.denoise(noisy * 2.0, fs)
    _check(
        "scale_linearity",
        bool(np.allclose(np.asarray(scaled), 2.0 * output, rtol=1e-6, atol=1e-8)),
        "linear filter: doubling input doubles output",
        results,
    )

    expected_path = HERE / "expected_output.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    for key, target in expected["metrics"].items():
        _check(
            f"expected_{key}",
            abs(metrics[key] - target) <= 0.02,
            f"got {metrics[key]}, expected {target} +/- 0.02",
            results,
        )

    passed = all(item["passed"] for item in results.values())
    return {"checks": results, "metrics": metrics, "passed": passed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate", nargs="?", default=str(HERE / "starter_solution.py")
    )
    args = parser.parse_args()
    report = grade(Path(args.candidate))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"RESULT: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
