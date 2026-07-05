# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnrtools/gwnrtools/blob/master/LICENSE>
"""Tests for gwnr.nr.utils"""

import numpy as np

from gwnr.nr.utils import (
    InnerProductVectors,
    JOfOmegaNonSpinning,
    PhaseFromPlanarOrbit,
)


def test_inner_product_vectors_normalized():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert np.isclose(InnerProductVectors(v1, v2), 0.0)
    assert np.isclose(InnerProductVectors(v1, v1), 1.0)
    # Normalization makes it scale-invariant
    assert np.isclose(InnerProductVectors(3 * v1, 7 * v1), 1.0)


def test_phase_from_planar_orbit_circular():
    # Circular orbit: r(t) = (cos wt, sin wt); the unwrapped phase must be
    # linear in t and grow by 2 pi per orbit
    w = 2 * np.pi  # one orbit per unit time
    t = np.linspace(0, 3, 3000)
    rvec = np.column_stack([np.cos(w * t), np.sin(w * t)])
    phase = PhaseFromPlanarOrbit(rvec)
    assert np.allclose(np.diff(phase), w * (t[1] - t[0]), rtol=1e-6)
    assert np.isclose(phase[-1] - phase[0], 3 * 2 * np.pi, rtol=1e-6)


def test_j_of_omega_nonspinning_positive_and_decreasing_in_omega():
    # PN orbital angular momentum decreases as the binary tightens
    j_lo = JOfOmegaNonSpinning(1.0, 0.25, 0.01)
    j_hi = JOfOmegaNonSpinning(1.0, 0.25, 0.05)
    assert j_lo > 0 and j_hi > 0
    assert j_lo > j_hi
