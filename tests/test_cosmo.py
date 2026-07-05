# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnr/gwnr/blob/master/LICENSE>
"""Tests for gwnr.cosmo"""

import numpy as np
import pytest

lal = pytest.importorskip("lal")

from gwnr.cosmo import (
    calculate_redshift,
    detector_to_source_frame,
    source_to_detector_frame,
)


def test_calculate_redshift_inverts_lal_luminosity_distance():
    h, om, ol, w0 = 0.679, 0.3065, 0.6935, -1.0
    omega = lal.CreateCosmologicalParameters(h, om, ol, w0, 0.0, 0.0)
    for z_true in [0.05, 0.1, 0.5]:
        dl = lal.LuminosityDistance(omega, z_true)  # Mpc
        z = calculate_redshift(float(dl), h=h, om=om, ol=ol, w0=w0)
        assert np.allclose(z, z_true, rtol=1e-6)


def test_calculate_redshift_monotonic_in_distance():
    z1 = calculate_redshift(400.0)
    z2 = calculate_redshift(800.0)
    assert z2 > z1 > 0


def test_frame_conversions_roundtrip():
    m_src, z = 30.0, 0.2
    m_det = source_to_detector_frame(m_src, z)
    assert np.isclose(m_det, m_src * (1 + z))
    assert np.isclose(detector_to_source_frame(m_det, z), m_src)
