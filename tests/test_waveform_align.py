# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnr/gwnr/blob/master/LICENSE>
"""Tests for gwnr.waveform.align"""

import numpy as np
import pytest

pytest.importorskip("pycbc")
from pycbc.types import TimeSeries
from pycbc.waveform import amplitude_from_polarizations

from gwnr.waveform.align import (
    align_waveforms_amplitude_peak,
    shift_waveform_phase,
    shift_waveform_time,
)

DT = 1.0 / 512


def _gaussian_chirplet(t_peak=2.0, f0=20.0, duration=4.0):
    """h = A(t) exp(-i Phi(t)) with a Gaussian amplitude peaked at t_peak"""
    t = np.arange(0, duration, DT)
    amp = np.exp(-0.5 * ((t - t_peak) / 0.3) ** 2)
    phase = 2 * np.pi * f0 * t
    hp = TimeSeries(amp * np.cos(phase), delta_t=DT)
    hc = TimeSeries(-amp * np.sin(phase), delta_t=DT)
    return hp, hc


def test_shift_waveform_phase_by_pi_flips_sign():
    hp, hc = _gaussian_chirplet()
    hp2, hc2 = shift_waveform_phase(hp, hc, np.pi, trim_trailing=False)
    assert np.allclose(hp2.data, -hp.data, atol=1e-10)
    assert np.allclose(hc2.data, -hc.data, atol=1e-10)


def test_shift_waveform_phase_preserves_amplitude():
    hp, hc = _gaussian_chirplet()
    hp2, hc2 = shift_waveform_phase(hp, hc, 0.7, trim_trailing=False)
    amp_before = amplitude_from_polarizations(hp, hc)
    amp_after = amplitude_from_polarizations(hp2, hc2)
    assert np.allclose(amp_before.data, amp_after.data, atol=1e-10)


def test_shift_waveform_time_epoch_only():
    hp, hc = _gaussian_chirplet()
    t_shift = 0.25
    hp2, hc2 = shift_waveform_time(
        hp, hc, t_shift, shift_epochs_only=True, trim_trailing=False
    )
    # Epoch-shift moves sample_times without touching the data
    assert np.allclose(hp2.data, hp.data)
    assert np.isclose(
        float(hp2.sample_times[0]) - float(hp.sample_times[0]), t_shift, atol=1e-9
    )


def test_align_waveforms_amplitude_peak():
    hp1, hc1 = _gaussian_chirplet(t_peak=1.5)
    hp2, hc2 = _gaussian_chirplet(t_peak=2.5)
    out = align_waveforms_amplitude_peak(
        hp1, hc1, hp2, hc2, shift_epochs_only=True, trim_trailing=False
    )
    ahp1, ahc1, ahp2, ahc2 = out[:4]
    amp1 = amplitude_from_polarizations(ahp1, ahc1)
    amp2 = amplitude_from_polarizations(ahp2, ahc2)
    t1 = float(amp1.sample_times[np.argmax(amp1.data)])
    t2 = float(amp2.sample_times[np.argmax(amp2.data)])
    assert abs(t1 - t2) < 5 * DT
