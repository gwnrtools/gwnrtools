# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnr/gwnr/blob/master/LICENSE>
"""Tests for gwnr.waveform.utils"""

import numpy as np
import pytest

lal = pytest.importorskip("lal")
pytest.importorskip("pycbc")

from gwnr.waveform.utils import f_ISCO_spin, get_detector_response


def test_f_isco_spin_nonspinning_close_to_schwarzschild():
    m1 = m2 = 10.0
    f_fit = f_ISCO_spin(m1, m2, 0.0, 0.0)
    # Schwarzschild ISCO for total mass 20 Msun (test-mass limit); the
    # fit includes comparable-mass corrections so allow a loose tolerance
    m_total_sec = (m1 + m2) * lal.MTSUN_SI
    f_schw = 1.0 / (6.0**1.5 * np.pi * m_total_sec)
    assert 0.7 * f_schw < f_fit < 1.5 * f_schw


def test_f_isco_spin_increases_with_aligned_spin():
    f_low = f_ISCO_spin(10.0, 10.0, 0.0, 0.0)
    f_high = f_ISCO_spin(10.0, 10.0, 0.9, 0.9)
    assert f_high > f_low


def test_f_isco_spin_scales_inversely_with_mass():
    f_20 = f_ISCO_spin(10.0, 10.0, 0.0, 0.0)
    f_40 = f_ISCO_spin(20.0, 20.0, 0.0, 0.0)
    assert np.isclose(f_20 / f_40, 2.0, rtol=1e-6)


def test_get_detector_response_bounds():
    fp, fc = get_detector_response(ra=1.0, dec=0.5, psi=0.3, detector_tag="H1")
    assert -1.0 <= fp <= 1.0
    assert -1.0 <= fc <= 1.0


def test_get_detector_response_known_detectors():
    for tag in ["H1", "L1", "V1", "G1"]:
        fp, fc = get_detector_response(0.0, 0.0, 0.0, tag)
        assert np.isfinite(fp) and np.isfinite(fc)
