"""Tests for 4th-order Butterworth bandpass filtering and normalization."""

import numpy as np
from scipy.signal import sosfiltfilt

from ecg_robust.preprocessing import (
    bandpass_filter,
    butter_bandpass_sos,
    normalize_segment,
)

FS = 500.0
N = 5000


def _test_signal():
    t = np.arange(N) / FS
    return (
        np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 100 * t) + 5.0
    )  # DC + passband + noise


def test_filter_removes_dc():
    y = bandpass_filter(_test_signal(), FS)
    assert len(y) == N
    assert abs(np.mean(y[1000:-1000])) < 0.05


def test_filter_preserves_length():
    assert len(bandpass_filter(np.random.default_rng(0).normal(size=N), FS)) == N


def test_sos_matches_direct_call():
    x = np.random.default_rng(1).normal(size=N)
    sos = butter_bandpass_sos(FS)
    assert np.allclose(bandpass_filter(x, FS), sosfiltfilt(sos, x))


def test_bandpass_sos_is_4th_order():
    sos = butter_bandpass_sos(FS, 0.5, 40.0, order=4)
    assert sos.ndim == 2 and sos.shape[1] == 6
    assert sos.shape[0] >= 2  # >= two second-order sections
    assert np.all(sos[:, 3] == 1)  # a0 == 1 in each row


def test_normalize_segment_uses_given_stats():
    seg = np.linspace(0, 1, 100)
    out = normalize_segment(seg, mean=0.5, std=0.5)
    assert np.allclose(out, (seg - 0.5) / (0.5 + 1e-12))
    # training-fitted parameters: a different scale shifts the result
    out2 = normalize_segment(seg, mean=1.0, std=1.0)
    assert not np.allclose(out, out2)


def test_normalize_segment_length_preserved():
    assert len(normalize_segment(np.zeros(77), 0.0, 1.0)) == 77
