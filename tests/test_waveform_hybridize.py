# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnrtools/gwnrtools/blob/master/LICENSE>
"""Tests for gwnr.waveform.hybridize helper functions"""

import numpy as np
import pytest

from gwnr.waveform.hybridize import (
    compute_amplitude,
    compute_frequency,
    compute_phase,
    find_first_value_location_in_series,
    find_last_value_location_in_series,
)


@pytest.fixture
def monochromatic_mode():
    """h(t) = A exp(-i 2 pi f t) sampled at 4096 Hz for 1 s"""
    delta_t = 1.0 / 4096
    t = np.arange(0, 1, delta_t)
    A, f = 2.5, 30.0
    h = A * np.exp(-1j * 2 * np.pi * f * t)
    return h, t, A, f, delta_t


def test_compute_amplitude(monochromatic_mode):
    h, t, A, f, delta_t = monochromatic_mode
    amp = compute_amplitude(h)
    assert np.allclose(amp, A)


def test_compute_phase_monotonic(monochromatic_mode):
    h, t, A, f, delta_t = monochromatic_mode
    phase = compute_phase(h)
    # phase = 2 pi f t (increasing, since compute_phase unwraps -angle)
    assert np.all(np.diff(phase) > 0)
    assert np.allclose(np.diff(phase), 2 * np.pi * f * delta_t, rtol=1e-6)


def test_compute_frequency(monochromatic_mode):
    h, t, A, f, delta_t = monochromatic_mode
    phase = compute_phase(h)
    freq = compute_frequency(phase, delta_t)
    assert np.allclose(freq, f, rtol=1e-5)


def test_find_value_locations():
    freq = np.linspace(10.0, 100.0, 91)  # monotonic frequency series
    idx = find_first_value_location_in_series(freq, 50.0)
    assert np.isclose(freq[idx], 50.0, atol=1.0)
    idx = find_last_value_location_in_series(freq, 50.0)
    assert np.isclose(freq[idx], 50.0, atol=1.0)


def test_find_value_location_out_of_bounds():
    freq = np.linspace(10.0, 100.0, 91)
    with pytest.raises(Exception):
        find_first_value_location_in_series(freq, 5.0)
    with pytest.raises(Exception):
        find_first_value_location_in_series(freq, 200.0)
