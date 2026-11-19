"""Patient-independent train/validation splits.

Uses ``sklearn.model_selection.StratifiedGroupKFold`` on recording IDs:
each MIT-BIH recording is a group, so individual heartbeats of the same
recording are never split between train and validation ("no patient
leakage"). Recordings 201 and 202 are treated as a single subject group
(the user instruction: "keep recordings 201 and 202 in the same subject
group").
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# MIT-BIH subject groups: recordings 201 and 202 belong to the same subject.
SUBJECT_GROUPS = {"201": "201+202", "202": "201+202"}


def make_groups(record_ids: list[str] | np.ndarray) -> np.ndarray:
    """Recording-ID groups; 201 and 202 share one subject group."""
    return np.array(
        [SUBJECT_GROUPS.get(str(r), str(r)) for r in record_ids], dtype=object
    )


def patient_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
):
    """Yield ``(train_idx, val_idx)`` folds via StratifiedGroupKFold.

    Stratification is on the AAMI class labels; grouping is on recording
    IDs, guaranteeing no patient overlap between folds.
    """
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=bool(shuffle), random_state=random_state
    )
    X_dummy = np.zeros((len(y), 1))
    yield from sgkf.split(X_dummy, np.asarray(y), groups=groups)
