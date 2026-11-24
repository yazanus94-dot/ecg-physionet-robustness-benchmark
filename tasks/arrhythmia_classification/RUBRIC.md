# Rubric — Patient-Independent Arrhythmia Classification (100 points)

| Criterion | Points | Full-credit evidence |
|---|---:|---|
| Independently verified fold predictions | 25 | grader's per-fold `fit_predict` OOF equals submitted predictions |
| Canonical fold metadata | 10 | `fold_groups` equals expected patient assignments |
| Preprocessing boundaries | 15 | train-only fit; test rows cannot influence one another |
| Input validation | 10 | ValueError for empty, mismatched, single-class, bad groups, wrong n_splits |
| Classification quality | 20 | balanced accuracy >= 0.65, macro F1 >= 0.55, dummy margin >= 0.15 |
| Metric honesty & determinism | 10 | reported metrics match recomputation; deterministic OOF |
| Documentation | 10 | delegation contract, synthetic framing and evaluation protocol explained |

Hard failures (leakage, metadata fabrication, label leakage, incomplete OOF, fabricated metrics)
fail the grader regardless of score. The fixture is synthetic and must not be reported as
clinical validation or as a reproduce of MIT-BIH complexity.
