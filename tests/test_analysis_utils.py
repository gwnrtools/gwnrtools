# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnr/gwnr/blob/master/LICENSE>
"""Tests for gwnr.analysis.utils"""

import numpy as np
import pytest

pytest.importorskip("pycbc")

from gwnr.analysis.utils import (
    get_uniform_mass_range,
    outside_mchirp_window,
    outside_tau0_window,
)


class _Point(object):
    def __init__(self, mass1, mass2):
        self.mass1 = mass1
        self.mass2 = mass2


def test_get_uniform_mass_range_endpoints():
    out = get_uniform_mass_range(5.0, 20.0, 2.0)
    assert out[0] == 5.0
    assert out[-1] == 20.0


def test_outside_mchirp_window_same_point():
    a = _Point(10.0, 10.0)
    b = _Point(10.0, 10.0)
    assert not outside_mchirp_window(a, b, 0.1)


def test_outside_mchirp_window_distant_points():
    a = _Point(10.0, 10.0)
    b = _Point(50.0, 50.0)
    assert outside_mchirp_window(a, b, 0.01)


def test_outside_tau0_window_same_point():
    a = _Point(10.0, 10.0)
    b = _Point(10.0, 10.0)
    assert not outside_tau0_window(a, b, 0.1, 20.0)
