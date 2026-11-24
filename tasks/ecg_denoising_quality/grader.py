"""Algorithm-independent grader for the ECG denoising benchmark task.

Usage:
    python grader.py [path/to/candidate.py]

A candidate is accepted when it produces scientifically acceptable signal
quality across several deterministic corruption scenarios. The output is NOT
compared to the reference filter waveform; the reference is educational only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from reference_solution import make_fixture, make_scenarios
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent

# Documented aggregate acceptance thresholds. A single scenario may fall
# below a target, but the aggregate and the catastrophic-failure guards below
# must hold. These are quality criteria, not reference-waveform matching.
MIN_MEDIAN_SNR_GAIN_DB = 1.5
MIN_MEAN_CORRELATION = 0.85
MIN_WEIGHTED_PEAK_F1 = 0.80
MIN_SINGLE_CORRELATION = 0.55
MIN_SINGLE_PEAK_F1 = 0.55
PEAK_TOLERANCE_S = 0.15


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


def match_peaks_one_to_one(
    detected: np.ndarray, reference: np.ndarray, tolerance_samples: int
) -> tuple[int, np.ndarray]:
    """Greedy closest-first one-to-one matching; each detection/reference used once."""
    if len(detected) == 0 or len(reference) == 0:
        return 0, np.array([], dtype=int)
    pairs = []
    for d_idx, detected_peak in enumerate(detected):
        for r_idx, reference_peak in enumerate(reference):
            distance = abs(int(detected_peak) - int(reference_peak))
            if distance <= tolerance_samples:
                pairs.append((distance, d_idx, r_idx))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    used_d: set[int] = set()
    used_r: set[int] = set()
    matched: list[int] = []
    for _distance, d_idx, r_idx in pairs:
        if d_idx in used_d or r_idx in used_r:
            continue
        used_d.add(d_idx)
        used_r.add(r_idx)
        matched.append(r_idx)
    return len(matched), np.asarray(sorted(matched), dtype=int)


def detection_f1(signal: np.ndarray, fs: int, peaks: np.ndarray) -> float:
    threshold = 0.6 * float(np.max(np.abs(signal)))
    detected, _ = find_peaks(signal, height=threshold, distance=int(0.25 * fs))
    matched, _ = match_peaks_one_to_one(detected, peaks, int(PEAK_TOLERANCE_S * fs))
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
        "snr_gain_db": round(snr_db(clean, output) - snr_db(clean, noisy), 4),
        "correlation": round(float(np.corrcoef(clean, output)[0, 1]), 4),
        "peak_f1": round(detection_f1(output, fs, peaks), 4),
    }


def aggregate(scenario_metrics: dict[str, dict]) -> dict:
    """Aggregate over scenarios: median SNR gain, mean correlation,
    length-weighted peak F1 (a longer recording contributes more detections).
    """
    gains = [m["snr_gain_db"] for m in scenario_metrics.values()]
    corrs = [m["correlation"] for m in scenario_metrics.values()]
    weights = [m["n_samples"] for m in scenario_metrics.values()]
    f1s = [m["peak_f1"] for m in scenario_metrics.values()]
    return {
        "median_snr_gain_db": round(float(np.median(gains)), 4),
        "mean_correlation": round(float(np.mean(corrs)), 4),
        "weighted_peak_f1": round(float(np.average(f1s, weights=weights)), 4),
        "min_correlation": round(float(np.min(corrs)), 4),
        "min_peak_f1": round(float(np.min(f1s)), 4),
        "n_scenarios": len(scenario_metrics),
    }


def _check(name: str, ok: bool, detail: str, results: dict) -> None:
    results[name] = {"passed": bool(ok), "detail": detail}


def grade(candidate_path: Path) -> dict:
    candidate = _load_candidate(candidate_path)
    results: dict[str, dict] = {}
    per_scenario: dict[str, dict] = {}

    for fixture in make_scenarios():
        name = fixture["name"]
        clean, noisy, fs, peaks = (
            fixture["clean"],
            fixture["noisy"],
            fixture["fs"],
            fixture["r_peaks"],
        )
        try:
            output = np.asarray(candidate.denoise(noisy, fs), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 - grader reports any candidate failure
            _check(
                f"run_{name}",
                False,
                f"denoise raised {type(exc).__name__}: {exc}",
                results,
            )
            continue
        _check(f"run_{name}", True, "denoise returned an array", results)
        _check(
            f"shape_{name}",
            output.shape == noisy.shape,
            f"expected {noisy.shape}, got {output.shape}",
            results,
        )
        _check(
            f"finite_{name}",
            bool(np.isfinite(output).all()),
            "all samples finite",
            results,
        )
        metrics = evaluate_output(clean, noisy, output, fs, peaks)
        metrics["n_samples"] = len(noisy)
        per_scenario[name] = metrics

    if not per_scenario:
        return {"checks": results, "scenarios": {}, "aggregate": {}, "passed": False}

    agg = aggregate(per_scenario)
    _check(
        "median_snr_gain",
        agg["median_snr_gain_db"] >= MIN_MEDIAN_SNR_GAIN_DB,
        f"median gain {agg['median_snr_gain_db']} dB, need >= {MIN_MEDIAN_SNR_GAIN_DB}",
        results,
    )
    _check(
        "mean_correlation",
        agg["mean_correlation"] >= MIN_MEAN_CORRELATION,
        f"mean correlation {agg['mean_correlation']}, need >= {MIN_MEAN_CORRELATION}",
        results,
    )
    _check(
        "weighted_peak_f1",
        agg["weighted_peak_f1"] >= MIN_WEIGHTED_PEAK_F1,
        f"weighted peak F1 {agg['weighted_peak_f1']}, need >= {MIN_WEIGHTED_PEAK_F1}",
        results,
    )
    _check(
        "no_catastrophic_morphology",
        agg["min_correlation"] >= MIN_SINGLE_CORRELATION,
        f"worst correlation {agg['min_correlation']}, need >= {MIN_SINGLE_CORRELATION}",
        results,
    )
    _check(
        "no_catastrophic_peaks",
        agg["min_peak_f1"] >= MIN_SINGLE_PEAK_F1,
        f"worst peak F1 {agg['min_peak_f1']}, need >= {MIN_SINGLE_PEAK_F1}",
        results,
    )

    # Determinism, invalid inputs and numerical stability on the default fixture.
    fixture = make_fixture()
    noisy, fs = fixture["noisy"], fixture["fs"]
    try:
        first = np.asarray(candidate.denoise(noisy, fs))
        second = np.asarray(candidate.denoise(noisy, fs))
        _check(
            "determinism",
            bool(np.array_equal(first, second)),
            "two identical calls give bitwise-identical output",
            results,
        )
    except Exception as exc:  # noqa: BLE001
        _check("determinism", False, f"raised {type(exc).__name__}", results)

    bad_inputs = {
        "empty": (np.array([]), fs),
        "2d": (np.zeros((10, 2)), fs),
        "nan": (np.full(120, np.nan), fs),
        "inf": (np.full(120, np.inf), fs),
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

    extreme = noisy.copy()
    extreme[100] = 1e9
    try:
        out = np.asarray(candidate.denoise(extreme, fs))
        _check(
            "extreme_stability",
            bool(np.isfinite(out).all()),
            "finite output on 1e9 impulse",
            results,
        )
    except Exception as exc:  # noqa: BLE001
        _check("extreme_stability", False, f"raised {type(exc).__name__}", results)

    expected = json.loads((HERE / "expected_output.json").read_text(encoding="utf-8"))
    # Informational only: reference metrics are recorded for transparency and
    # are explicitly NOT acceptance criteria for alternative implementations.
    _check(
        "expected_output_present",
        bool(expected.get("scenarios")),
        "reference metrics recorded; not used as candidate acceptance criteria",
        results,
    )

    passed = all(item["passed"] for item in results.values())
    return {
        "checks": results,
        "scenarios": per_scenario,
        "aggregate": agg,
        "passed": passed,
    }


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
