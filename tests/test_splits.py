"""Tests for patient-independent splitting (no patient leakage)."""

import numpy as np
import pytest

from ecg_robust.splits import make_groups, patient_splits


def test_make_groups_201_202_share_group():
    groups = make_groups(["201", "202", "303"])
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]
    assert set(groups[:2]) == {"201+202"}


def test_patient_splits_five_folds():
    rng = np.random.default_rng(0)
    y = rng.choice(["N", "S", "V"], size=60)
    groups = make_groups([f"r{i}" for i in range(60)])
    folds = list(patient_splits(y, groups, n_splits=5, random_state=42))
    assert len(folds) == 5


def test_no_patient_overlap_between_folds():
    rng = np.random.default_rng(1)
    y = rng.choice(["N", "S", "V", "F", "Q"], size=50)
    # each beat belongs to a distinct recording (distinct group)
    groups = make_groups([f"r{i}" for i in range(50)])
    for tr, va in patient_splits(y, groups, n_splits=5, random_state=42):
        assert len(tr) > 0 and len(va) > 0
        assert len(set(groups[tr]) & set(groups[va])) == 0


def test_group_201_202_never_split():
    y = np.array(["N"] * 70)
    recs = (
        ["201"] * 10
        + ["202"] * 10
        + ["303"] * 20
        + ["304"] * 10
        + ["305"] * 10
        + ["306"] * 10
    )
    groups = make_groups(recs)
    group = "201+202"
    folds = list(patient_splits(y, groups, n_splits=5, random_state=42))
    assert len(folds) == 5
    for tr, va in folds:
        # the two recordings of 201/202 must not be separated:
        assert not (group in set(groups[tr]) and group in set(groups[va]))
    # the shared group is exercised in some fold
    assert any(
        group in set(groups[va]) or group in set(groups[tr]) for _tr, va in folds
    )


def test_pipeline_folds_no_group_leakage_on_real_records():
    # Phase 6: the actual pipeline design — patient-independent folds over
    # the real MITDB recording groups (201/202 grouped) — has no
    # train/validation group overlap (no patient leakage).
    import numpy as np

    from ecg_robust.dataset import MITDB_DIR, database_records
    from ecg_robust.splits import make_groups, patient_splits

    records = database_records(MITDB_DIR)
    if not records:
        pytest.skip("MITDB not downloaded")
    y = np.array(["N"] * len(records))
    groups = make_groups(records)
    for tr, va in patient_splits(y, groups, n_splits=5, random_state=42):
        assert len(set(groups[tr]) & set(groups[va])) == 0
    # 201/202 share one subject group
    assert make_groups(["201", "202"])[0] == make_groups(["201", "202"])[1]
