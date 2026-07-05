# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnr/gwnr/blob/master/LICENSE>
"""Tests for gwnr.stats.distribution"""

import numpy as np
import pytest

pytest.importorskip("scipy.stats")

from gwnr.stats.distribution import OneDDistribution


@pytest.fixture
def normal_samples():
    rng = np.random.RandomState(42)
    return rng.normal(loc=5.0, scale=2.0, size=20000)


def test_oned_distribution_mean_median(normal_samples):
    dist = OneDDistribution(normal_samples)
    assert np.isclose(dist.mean(), 5.0, atol=0.1)
    assert np.isclose(dist.median(), 5.0, atol=0.1)


def test_oned_distribution_percentile(normal_samples):
    dist = OneDDistribution(normal_samples)
    # 84th percentile of N(5, 2) is ~ mean + sigma
    assert np.isclose(dist.percentile(84.1), 7.0, atol=0.2)
    assert np.isclose(dist.percentile(15.9), 3.0, atol=0.2)


def test_oned_distribution_xlimits(normal_samples):
    dist = OneDDistribution(normal_samples)
    lo, hi = dist.xlimits()
    assert lo == normal_samples.min()
    assert hi == normal_samples.max()
