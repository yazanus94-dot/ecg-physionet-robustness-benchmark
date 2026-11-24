# Rubric — Patient-Independent Arrhythmia Classification (100 points)

| Criterion | Points | Full-credit evidence |
|---|---:|---|
| Patient-level split integrity | 20 | every fold reports disjoint train/test patient sets |
| Leak-free preprocessing | 15 | imputation/scaling/model fit on training rows only; test labels unused |
| Input validation | 15 | ValueError for empty, mismatched, single-class, unsupported-label and bad-group inputs |
| Classification quality | 20 | balanced accuracy >= 0.80 and >= dummy + 0.15 |
| Metric correctness | 10 | OOF aggregation, explicit label universe, macro/balanced definitions |
| Determinism | 10 | repeated OOF runs give identical predictions |
| Documentation | 10 | split design, class imbalance handling and synthetic framing explained |

Hard failures (leakage, schema violation, invalid-input handling) fail the grader regardless of
score. The benchmark is synthetic and must not be reported as clinical validation.
