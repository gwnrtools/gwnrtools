# Copyright (C) 2018 Prayush Kumar
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
from __future__ import print_function

import numpy as np

########################################


def JOfOmegaNonSpinning(m, eta, om):
    """
    c.f. Eq.234 of Blanchet (2013) Living Review arxiv:1310.1528
    """
    x = om ** (2.0 / 3.0)
    mu = eta * m
    J0 = mu * m / x**0.5
    J1 = (1.5 + eta / 6.0) * x
    J2 = (27.0 / 8.0 - 19.0 * eta / 8.0 + eta**2 / 24.0) * x**2
    J3 = (
        135.0 / 16.0
        + (-6889.0 / 144.0 + 41.0 / 24.0 * np.pi**2) * eta
        + 31.0 * eta**2 / 24.0
        + 7.0 * eta**3 / 1296.0
    ) * x**3
    e4 = (
        -123671.0 / 5760.0
        + 9037.0 * np.pi**2 / 1536.0
        + 1792.0 * np.log(2) / 15.0
        + 896.0 * np.euler_gamma / 15.0
        + (-498449.0 / 3456.0 + 3157.0 * np.pi**2 / 576.0) * eta
        + 301.0 * eta**2 / 1728.0
        + 77.0 * eta**3 / 31104.0
    )
    j4 = -5.0 * e4 / 7.0 + 64.0 / 35.0
    J4 = (2835.0 / 128.0 + eta * j4 - 64.0 * eta * np.log(x) / 3.0) * x**4
    e5 = 0
    j5 = -2.0 * e5 / 3.0 - 4988.0 / 945.0 - 656.0 * eta / 135.0
    J5 = (
        15309.0 / 256.0
        + eta * j5
        + (9976.0 / 105.0 + 1312.0 * eta / 15.0) * eta * np.log(x)
    ) * x**5
    return J0 * (1.0 + J1 + J2 + J3 + J4 + J5 * 0)


def InnerProductVectors(v1, v2):
    sm = 0
    for tv1, tv2 in zip(v1, v2):
        sm += tv1 * tv2
    v1_norm = np.sum(v1**2) ** 0.5
    v2_norm = np.sum(v2**2) ** 0.5
    sm /= v1_norm
    sm /= v2_norm
    return sm


def PhaseFromPlanarOrbit(rVec):
    PhaseWrapped = np.arctan2(rVec[:, 1], rVec[:, 0])
    PhaseUnWrapped = np.unwrap(PhaseWrapped, discont=np.pi)
    return PhaseUnWrapped
