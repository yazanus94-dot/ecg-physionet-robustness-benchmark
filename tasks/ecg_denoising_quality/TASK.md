# Task A — ECG Denoising and Signal-Quality Robustness

## Problem

Given a single-lead ECG corrupted by baseline wander, powerline interference and broadband
noise, return a denoised signal that preserves diagnostic morphology and R-peak timing.
Self-contained and **synthetic**; no PhysioNet data, hidden labels or network access is used.

The grader is **algorithm-independent**: it does not compare your waveform to the reference
filter. Any scientifically defensible method is accepted if it meets the quality criteria.

## Input / output

`denoise(signal: np.ndarray, fs: float) -> np.ndarray`
- non-empty 1-D finite samples, same output length as input;
- invalid inputs (empty, 2-D, NaN/Inf, non-positive `fs`, too short) must raise `ValueError`;
- deterministic, numerically stable on extreme impulses, no silent truncation.

## Deterministic scenarios

`make_fixture(scenario)` / `make_scenarios()` provide seven seeded fixtures:

| Scenario | Corruption / boundary |
|---|---|
| `baseline_wander` | strong 0.17 Hz wander + small broadband noise |
| `powerline_50hz` | wander + 50 Hz line + broadband noise (fs 360) |
| `broadband_low_snr` | low input SNR (1.5 dB), mostly in-band noise |
| `high_snr` | 12 dB input SNR; filtering can *reduce* measured SNR |
| `short_recording` | 3 s recording |
| `low_sampling_rate` | fs 250 Hz, line interference below Nyquist |
| `boundary_peaks` | R peaks close to recording boundaries |

Two ECG morphologies (A/B), different SNRs, short/long signals and near-filter-boundary cases
are included. Clean synthetic signals are the ground truth. Some scenarios do not improve every
individual metric — this is intentional and handled by the aggregate rule below.

## Aggregate evaluation (documented formula)

For each scenario the grader computes SNR gain (`SNR_out - SNR_in`), Pearson correlation with
clean, and one-to-one R-peak F1 (150 ms tolerance, closest-first greedy matching, each
detection/reference used once). It then requires:

- median SNR gain >= **1.5 dB**
- mean correlation >= **0.85**
- length-weighted peak F1 >= **0.80**
- no single scenario with correlation < **0.55** or peak F1 < **0.55**

Shape, finiteness, determinism, invalid-input handling and extreme-value stability are hard
checks. `expected_output.json` records the reference's metrics **for transparency only**; they
are not universal acceptance thresholds.

## Reproduction

```bash
python generate_expected_output.py
python grader.py reference_solution.py      # PASS
python grader.py starter_solution.py        # FAIL
python -m pytest -q tests                   # includes alternative + adversarial candidates
```

## Files

`TASK.md`, `RUBRIC.md`, `starter_solution.py`, `reference_solution.py`, `grader.py`,
`expected_output.json`, `generate_expected_output.py`, `tests/` (including
`tests/candidates/` with valid-alternative, passthrough and morphology-damaging solutions).

Synthetic benchmark evidence only — never clinical validation.
