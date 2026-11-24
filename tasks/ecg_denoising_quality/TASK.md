# Task A — ECG Denoising and Signal-Quality Robustness

## Problem

Given a single-lead ECG corrupted by baseline wander, 50 Hz powerline interference and
additive noise, produce a denoised signal that preserves diagnostic morphology and R-peak
timing. This is a **synthetic, self-contained** benchmark: no PhysioNet data, hidden labels or
network access is used by the grader.

## Input / output schema

- Input: `denoise(signal: np.ndarray, fs: float) -> np.ndarray`
  - `signal`: non-empty 1-D finite array of samples in mV-like units.
  - `fs`: positive finite sampling rate in Hz.
- Output: 1-D finite `np.ndarray` with the **same length** as the input.

Invalid inputs (empty, 2-D, non-finite, non-positive `fs`, too short for the filter) must raise
`ValueError`. No silent truncation, interpolation or sample invention is allowed.

## Deterministic fixture

`reference_solution.make_fixture()` generates a 10 s, 360 Hz synthetic ECG (seed 20261124)
with 11 annotated R peaks. The noisy mixture is scaled to 0 dB input SNR; its corruption is
mostly out-of-band (0.15 Hz wander + 50 Hz interference + small in-band white noise).

## Grading

`python grader.py path/to/candidate.py` checks, in order:

1. schema — output exists, 1-D, same length, finite;
2. quality — SNR improvement >= 4 dB, correlation with clean >= 0.90, R-peak F1 >= 0.85
   (150 ms tolerance) using a documented threshold detector;
3. correctness — agreement with the reference filter within `rtol=1e-6, atol=1e-8`;
4. determinism — identical calls return bitwise-identical output;
5. adversarial — empty, 2-D, NaN/Inf, bad `fs`, too-short, amplitude scaling, impulse stability;
6. expected metrics — all values in `expected_output.json` within +/-0.02.

The grader exits 0 only if every check passes. It tests signal correctness, not just execution.

## Reference approach (technical explanation)

A 4th-order Butterworth bandpass (0.5–40 Hz) is designed as second-order sections and applied
with `scipy.signal.sosfiltfilt`, giving zero phase so R peaks are not time-shifted. The padding
requirement `3 * (2 * n_sections + 1)` is checked explicitly before filtering; short signals
raise rather than relying on SciPy's internal error. SOS avoids the numerical instability of
transfer-function coefficients at low normalized cutoffs. The filter is linear, so doubling the
input doubles the output, and deterministic because it uses no randomness.

## Reproduction

```bash
python generate_expected_output.py     # regenerate expected_output.json
python grader.py reference_solution.py # PASS
python grader.py starter_solution.py   # FAIL (NotImplementedError)
python -m pytest -q tests
```

## Files

`TASK.md`, `RUBRIC.md`, `starter_solution.py`, `reference_solution.py`, `grader.py`,
`expected_output.json`, `generate_expected_output.py`, `tests/`.

Results are synthetic benchmark evidence only — never clinical validation.
