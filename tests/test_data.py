# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnrtools/gwnrtools/blob/master/LICENSE>
"""Tests for gwnr.data"""

import os

import numpy as np

from gwnr.data import available_gw_noise_curves, gw_noise_curve_file


def test_available_gw_noise_curves_nonempty():
    curves = available_gw_noise_curves()
    assert len(curves) > 0
    for name in curves:
        assert "/" not in name


def test_gw_noise_curve_files_exist():
    for name in available_gw_noise_curves():
        path = gw_noise_curve_file(name)
        assert os.path.isabs(path)
        assert os.path.exists(path)


def test_noise_curves_are_loadable_ascii():
    for name in available_gw_noise_curves():
        if not name.endswith((".txt", ".dat")):
            continue
        data = np.loadtxt(gw_noise_curve_file(name))
        assert data.ndim == 2
        assert data.shape[0] > 10  # frequency rows
        assert data.shape[1] >= 2  # frequency + at least one PSD/ASD column
        # frequency column should be positive
        assert np.all(data[:, 0] >= 0)
