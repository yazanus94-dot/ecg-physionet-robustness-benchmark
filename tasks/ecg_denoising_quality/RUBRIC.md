# Rubric — ECG Denoising Benchmark (100 points)

| Criterion | Points | Full-credit evidence |
|---|---:|---|
| Input validation | 15 | ValueError for empty, 2-D, non-finite, non-positive fs and too-short signals |
| Output schema | 10 | 1-D finite array, exact input length |
| SNR improvement | 20 | >= 4 dB on the deterministic fixture |
| Morphology preservation | 15 | correlation with clean >= 0.90 |
| R-peak timing | 15 | detection F1 >= 0.85 at 150 ms tolerance |
| Numerical/determinism | 10 | finite on impulses/constants; bitwise-identical repeats; amplitude linearity |
| Reference agreement | 10 | output matches reference within 1e-6 relative tolerance |
| Documentation | 5 | filter design, edge handling and units explained |

A solution scoring below 70, or failing any hard schema/edge check, does not pass. The grader's
binary PASS/FAIL corresponds to all hard checks passing; the rubric is for human review of
partial solutions.
