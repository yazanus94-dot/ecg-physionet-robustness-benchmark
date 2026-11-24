"""Regression tests for incomplete-record handling in the pipeline."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ecg_robust.pipeline import load_record_safely


def test_incomplete_record_is_skipped_with_reason(tmp_path):
    (tmp_path / "999.dat").touch()  # .dat present, .hea/.atr missing
    with patch(
        "ecg_robust.pipeline.load_ecg_record", side_effect=FileNotFoundError("no .hea")
    ):
        loaded, reason = load_record_safely("999", tmp_path)
    assert loaded is None
    assert reason and "no .hea" in reason


def test_misaligned_annotations_are_skipped():
    rec = SimpleNamespace(samples=np.zeros((100, 1)), fs=360)
    ann = SimpleNamespace(sample=np.array([1, 2]), symbol=["N"])
    with patch("ecg_robust.pipeline.load_ecg_record", return_value=(rec, ann)):
        loaded, reason = load_record_safely("100", "unused")
    assert loaded is None
    assert "misaligned" in reason


def test_complete_record_loads():
    rec = SimpleNamespace(samples=np.zeros((100, 1)), fs=360)
    ann = SimpleNamespace(sample=np.array([1, 2]), symbol=["N", "V"])
    with patch("ecg_robust.pipeline.load_ecg_record", return_value=(rec, ann)):
        loaded, reason = load_record_safely("100", "unused")
    assert loaded is not None
    assert reason is None
