"""Tests for the documented feature set."""

import numpy as np

from ecg_robust.features import extract_features


def _segment(n=100, fs=1000.0, amp=0.5):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * 5 * t)


def test_expected_feature_keys():
    f = extract_features(_segment(), 1000.0)
    for key in [
        "mean",
        "std",
        "min",
        "max",
        "energy",
        "peak_amplitude",
        "duration_s",
        "rise_time_s",
        "fall_time_s",
        "zero_crossings",
        "spectral_centroid",
        "spectral_flatness",
        "fft_peak_frequency",
        "fft_total_energy",
        "skew",
        "kurtosis",
        "p5",
        "p25",
        "p50",
        "p75",
        "p95",
    ]:
        assert key in f
    assert np.isfinite(list(f.values())).all()


def test_basic_stats_on_sine():
    seg = _segment(amp=0.5)
    f = extract_features(seg, 1000.0)
    assert abs(f["peak_amplitude"] - 0.5) < 0.05
    assert abs(f["duration_s"] - 0.1) < 1e-9
    assert f["duration_s"] == len(seg) / 1000.0
    assert abs(f["energy"] - 100 * 0.5**2 / 2) < 0.05  # E = sum(x^2)
    assert f["min"] <= 0 < f["max"]  # sine starts at 0; max at 0.5


def test_zero_crossings_nonnegative():
    assert extract_features(_segment(), 1000.0)["zero_crossings"] >= 0


def test_rr_and_ratio_included_when_provided():
    seg = _segment()
    f = extract_features(seg, 1000.0, rr=0.030, rr_ratio=1.1)
    assert f["rr_s"] == 0.030
    assert f["rr_ratio"] == 1.1


def test_segment_too_short_raises():
    with np.errstate(all="ignore"):
        import pytest

        with pytest.raises(ValueError):
            extract_features(np.array([0.0, 1.0]), 1000.0)


def test_missing_or_invalid_feature_values_raise():
    import pytest

    with pytest.raises(ValueError):
        extract_features(np.array([np.nan] * 4), 1000.0)
    with pytest.raises(ValueError):
        extract_features(np.array([np.inf, 0.0, 0.0, 0.0]), 1000.0)
    with pytest.raises(ValueError):
        extract_features(np.array([-np.inf, 0.0, 0.0, 0.0]), 1000.0)
