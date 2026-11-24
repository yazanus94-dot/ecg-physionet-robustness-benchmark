"""Edge-case and adversarial tests: invalid inputs, stability and boundaries."""

import numpy as np
import pytest
from reference_solution import denoise, make_fixture


@pytest.mark.parametrize(
    "signal,fs",
    [
        (np.array([]), 360),
        (np.ones((20, 2)), 360),
        (np.full(100, np.nan), 360),
        (np.full(100, np.inf), 360),
        (np.ones(100), 0),
        (np.ones(100), -360),
        (np.ones(10), 360),
    ],
)
def test_invalid_inputs_raise_value_error(signal, fs):
    with pytest.raises(ValueError):
        denoise(signal, fs)


def test_short_signal_boundary():
    """A signal exactly at the padding boundary must not raise an SciPy error."""
    from scipy.signal import butter

    sos = butter(4, [0.5, 40.0], btype="bandpass", fs=360, output="sos")
    padlen = 3 * (2 * sos.shape[0] + 1)
    with pytest.raises(ValueError, match="too short"):
        denoise(np.ones(padlen), 360)
    result = denoise(np.ones(padlen + 1), 360)
    assert result.shape == (padlen + 1,)


def test_constant_signal_is_numerically_stable():
    result = denoise(np.ones(2000), 360)
    assert np.isfinite(result).all()


def test_amplitude_scale_linearity():
    noisy = make_fixture()["noisy"]
    base = denoise(noisy, 360)
    doubled = denoise(noisy * 2.0, 360)
    np.testing.assert_allclose(doubled, 2.0 * base, rtol=1e-9, atol=1e-10)


def test_impulse_does_not_produce_non_finite_output():
    noisy = make_fixture()["noisy"].copy()
    noisy[1000] = 1e6
    result = denoise(noisy, 360)
    assert np.isfinite(result).all()
