# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnr/gwnr/blob/master/LICENSE>
"""Tests for gwnr.waveform.parameters"""

import numpy as np
import pytest

from gwnr.waveform.parameters import (
    chip_from_masses_spins,
    eta_to_q,
    q_to_eta,
    spins_to_2PNeffective_spin,
    spins_to_PNeffective_spin,
    spins_to_damoureffective_spin,
    spins_to_massweighted_spin,
)


def test_q_to_eta_known_values():
    assert np.isclose(q_to_eta(1.0), 0.25)
    assert np.isclose(q_to_eta(4.0), 4.0 / 25.0)


def test_eta_to_q_roundtrip():
    for q in [1.0, 1.5, 3.0, 10.0]:
        eta = q_to_eta(q)
        q_rec = eta_to_q(eta)
        # eta -> q is two-valued; the function returns one root: q or 1/q
        assert np.isclose(q_rec, q, rtol=1e-10) or np.isclose(
            q_rec, 1.0 / q, rtol=1e-10
        )


def test_eta_bounded_above_by_quarter():
    qs = np.linspace(1.0, 20.0, 50)
    etas = q_to_eta(qs)
    assert np.all(etas <= 0.25 + 1e-12)
    assert np.all(etas > 0)


def test_massweighted_spin_equal_masses_is_mean():
    assert np.isclose(spins_to_massweighted_spin(10, 10, 0.5, -0.1), 0.2)


def test_massweighted_spin_limits():
    # All weight on body 1 when m2 -> 0
    assert np.isclose(spins_to_massweighted_spin(10, 1e-9, 0.7, -0.5), 0.7, atol=1e-8)


def test_pn_effective_spin_equal_spins():
    # For chi1 = chi2 = chi, any mass combination must return chi times
    # (113 (m1^2+m2^2) + 150 m1 m2)/(113 (m1+m2)^2) -- check equal-mass case:
    # (113*2 + 150)/(113*4) = 376/452
    chi = 0.6
    expected = chi * (113.0 * 2 + 150.0) / (113.0 * 4)
    assert np.isclose(spins_to_PNeffective_spin(5, 5, chi, chi), expected)


def test_2pn_effective_spin_zero_spins():
    assert spins_to_2PNeffective_spin(5, 5, 0.0, 0.0) == 0.0


def test_damour_effective_spin_equal_mass_equal_spin():
    chi = 0.4
    # (4 m^2 chi + 4 m^2 chi + 6 m^2 chi) / (4 * 4 m^2) = 14/16 chi
    assert np.isclose(spins_to_damoureffective_spin(5, 5, chi, chi), 14.0 / 16.0 * chi)


def test_chip_aligned_spins_is_zero():
    assert chip_from_masses_spins(10, 5, 0, 0, 0.9, 0, 0, -0.3) == 0.0


def test_chip_inplane_primary_spin():
    # For in-plane spin only on the (heavier) primary, chi_p equals its magnitude
    s1x, s1y = 0.3, 0.4
    chip = chip_from_masses_spins(10, 5, s1x, s1y, 0.0, 0, 0, 0)
    assert np.isclose(chip, np.sqrt(s1x**2 + s1y**2))


def test_chip_bounded():
    rng = np.random.RandomState(7)
    for _ in range(20):
        s1 = rng.uniform(-0.5, 0.5, 3)
        s2 = rng.uniform(-0.5, 0.5, 3)
        chip = chip_from_masses_spins(12.0, 7.0, *s1, *s2)
        assert 0 <= chip < 1
