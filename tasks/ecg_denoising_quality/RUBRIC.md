# Rubric — ECG Denoising Benchmark (100 points)

| Criterion | Points | Full-credit evidence |
|---|---:|---|
| Input validation & boundary handling | 15 | ValueError for empty, 2-D, non-finite, bad fs and too-short signals |
| Output schema | 10 | 1-D finite array, exact input length |
| SNR improvement (aggregate) | 20 | median gain >= 1.5 dB across seven scenarios |
| Morphology preservation | 20 | mean correlation >= 0.85; no scenario < 0.55 |
| R-peak timing | 20 | length-weighted one-to-one F1 >= 0.80; no scenario < 0.55 |
| Determinism & numerical stability | 10 | bitwise-identical repeats; finite output on 1e9 impulse |
| Documentation | 5 | algorithm-independent design and aggregate rule explained |

Hard failures (schema, validation, instability, catastrophic morphology loss) fail the grader
regardless of score. The reference filter is an educational example; alternative methods are
expected to produce different waveforms and are accepted when the quality criteria hold.
