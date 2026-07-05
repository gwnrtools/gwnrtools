# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnr/gwnr/blob/master/LICENSE>
"""Tests for gwnr.analysis.psd"""

import numpy as np
import pytest

pycbc_types = pytest.importorskip("pycbc.types")

from gwnr.analysis.psd import resample_and_extrapolate_psd


def test_resample_and_extrapolate_psd():
    # Synthetic PSD measured between 10 and 50 Hz
    freqs = np.linspace(10.0, 50.0, 401)
    psd_vals = 1e-46 * (freqs / 20.0) ** -4
    delta_f, f_max = 0.25, 64.0

    psd = resample_and_extrapolate_psd(freqs, psd_vals, delta_f, f_max)

    assert np.isclose(psd.delta_f, delta_f)
    assert len(psd) == int(round(f_max / delta_f))

    sample_f = np.array(psd.sample_frequencies)
    # Below the measured band: extrapolated with the lowest measured value
    below = sample_f < freqs[0]
    assert np.allclose(psd.data[below], psd_vals[0])
    # Above the measured band: extrapolated with the highest measured value
    above = sample_f > freqs[-1]
    assert np.allclose(psd.data[above], psd_vals[-1])
    # Inside the band: interpolation error is small
    inside = (sample_f >= freqs[0]) & (sample_f <= freqs[-1])
    expected = 1e-46 * (sample_f[inside] / 20.0) ** -4
    assert np.allclose(psd.data[inside], expected, rtol=1e-3)


def test_resample_and_extrapolate_psd_length_mismatch():
    with pytest.raises(AssertionError):
        resample_and_extrapolate_psd(np.arange(10), np.arange(9), 0.25, 64.0)
