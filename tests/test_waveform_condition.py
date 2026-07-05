# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnrtools/gwnrtools/blob/master/LICENSE>
"""Tests for gwnr.waveform.condition"""

import numpy as np
import pytest

from gwnr.waveform.condition import moving_window_average, planck_window, smooth


def test_planck_window_shape_and_range():
    win = planck_window(N=100, eps=0.1)
    assert len(win) == 100
    assert np.all(win > 0)
    assert np.all(win <= 1.0)
    # One-sided window: turns on smoothly, ends fully open
    assert win[0] < 0.5
    assert np.isclose(win[-1], 1.0)
    assert np.all(np.diff(win) >= -1e-12)


def test_planck_window_requires_args():
    with pytest.raises(IOError):
        planck_window()


def test_planck_window_winstart_prepends_ones():
    win = planck_window(N=100, eps=0.1, winstart=10)
    assert np.allclose(win[:10], 1.0)


def test_smooth_preserves_constant_signal():
    x = np.ones(200) * 3.14
    y = smooth(x, window_len=11, window="flat")
    assert np.allclose(y, 3.14)


def test_smooth_reduces_noise_variance():
    rng = np.random.RandomState(0)
    x = rng.normal(0, 1, 2000)
    y = smooth(x, window_len=21, window="flat")
    assert np.var(y) < 0.5 * np.var(x)


def test_moving_window_average_constant():
    x = np.ones(50) * 7.0
    assert np.isclose(moving_window_average(x, 25, window_len=10), 7.0)
