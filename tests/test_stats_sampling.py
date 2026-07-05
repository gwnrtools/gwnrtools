# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnr/gwnr/blob/master/LICENSE>
"""Tests for gwnr.stats.sampling"""

import numpy as np
import pandas as pd
import pytest

from gwnr.stats.sampling import (
    OneDRandom,
    cube_to_uniform_on_S2,
    idempotence,
    uniform_in_angle,
    uniform_in_cos_angle,
    uniform_in_totalmass_massratio_masses,
    uniform_in_volume_distance,
    uniform_mass,
    uniform_massratio,
    uniform_on_S2,
    uniform_spin_magnitude,
    zero_distribution,
)

N = 500


def test_uniform_samplers_respect_bounds():
    assert np.all((uniform_massratio(N, 1.0, 5.0) >= 1.0))
    assert np.all((uniform_massratio(N, 1.0, 5.0) <= 5.0))
    m = uniform_mass(N, 5.0, 50.0)
    assert np.all((m >= 5.0) & (m <= 50.0))
    a = uniform_spin_magnitude(N, 0.0, 0.99)
    assert np.all((a >= 0.0) & (a <= 0.99))


def test_zero_distribution():
    assert np.all(zero_distribution(N) == 0)
    assert len(zero_distribution(N)) == N


def test_uniform_in_angle_bounds():
    ang = uniform_in_angle(N)
    assert np.all((ang >= 0) & (ang <= 2 * np.pi))


def test_uniform_in_cos_angle_bounds():
    theta = uniform_in_cos_angle(N)
    assert np.all((theta >= 0) & (theta <= np.pi))


def test_cube_to_uniform_on_S2():
    u = np.linspace(0, 1, 101)
    v = np.linspace(0, 1, 101)
    phi, theta = cube_to_uniform_on_S2(u, v)
    assert np.all((phi >= 0) & (phi <= 2 * np.pi))
    assert np.all((theta >= 0) & (theta <= np.pi))


def test_cube_to_uniform_on_S2_rejects_out_of_range():
    with pytest.raises(IOError):
        cube_to_uniform_on_S2(np.array([1.5]), np.array([0.5]))


def test_uniform_on_S2():
    phi, theta = uniform_on_S2(N)
    assert len(phi) == N and len(theta) == N
    assert np.all((phi >= 0) & (phi <= 2 * np.pi))
    assert np.all((theta >= 0) & (theta <= np.pi))


def test_uniform_in_volume_distance_bounds_and_density():
    d = uniform_in_volume_distance(10000, 100.0, 1000.0)
    assert np.all((d >= 100.0) & (d <= 1000.0))
    # Uniform-in-volume means more samples at larger distance
    assert np.sum(d > 550.0) > np.sum(d < 550.0)


def test_uniform_in_totalmass_massratio_masses():
    m1, m2 = uniform_in_totalmass_massratio_masses(N, 20.0, 100.0, 1.0, 4.0)
    mtot = m1 + m2
    assert np.all((mtot >= 20.0 - 1e-9) & (mtot <= 100.0 + 1e-9))
    assert np.all(m1 > 0) and np.all(m2 > 0)


def test_idempotence():
    out = idempotence(5, 3.14)
    assert out.shape == (5,)
    assert np.all(out == 3.14)


def test_onedrandom_uniform_sampling():
    params = pd.DataFrame(
        {"mass1": pd.Series({"dist": "uniform", "range": (10.0, 20.0)})}
    )
    sampler = OneDRandom(params)
    assert "mass1" in sampler.available_parameters()
    assert "uniform" in sampler.available_distributions()
    samples = sampler.sample("mass1", size=100)
    assert len(samples) == 100
    assert np.all((samples >= 10.0) & (samples <= 20.0))
