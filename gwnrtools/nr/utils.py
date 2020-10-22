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
    x = om**(2. / 3.)
    mu = eta * m
    J0 = mu * m / x**0.5
    J1 = (1.5 + eta / 6.0) * x
    J2 = (27. / 8. - 19. * eta / 8. + eta**2 / 24.) * x**2
    J3 = (135. / 16. + (-6889. / 144. + 41. / 24. * np.pi**2) * eta +
          31. * eta**2 / 24. + 7. * eta**3 / 1296.) * x**3
    e4 = -123671. / 5760. + 9037. * np.pi**2 / 1536. + 1792. * np.log(2) / 15. +\
        896.*np.euler_gamma/15. + (-498449./3456. + 3157. * np.pi**2 / 576.) * eta +\
        301. * eta**2 / 1728. + 77. * eta**3 / 31104.
    j4 = -5. * e4 / 7. + 64. / 35.
    J4 = (2835. / 128. + eta * j4 - 64. * eta * np.log(x) / 3.) * x**4
    e5 = 0
    j5 = -2. * e5 / 3. - 4988. / 945. - 656. * eta / 135.
    J5 = (15309. / 256. + eta * j5 +
          (9976. / 105. + 1312. * eta / 15.) * eta * np.log(x)) * x**5
    return J0 * (1. + J1 + J2 + J3 + J4 + J5 * 0)


def InnerProductVectors(v1, v2):
    sm = 0
    for tv1, tv2 in zip(v1, v2):
        sm += tv1 * tv2
    v1_norm = np.sum(v1**2)**0.5
    v2_norm = np.sum(v2**2)**0.5
    sm /= v1_norm
    sm /= v2_norm
    return sm


def PhaseFromPlanarOrbit(rVec):
    PhaseWrapped = np.arctan2(rVec[:, 1], rVec[:, 0])
    PhaseUnWrapped = np.unwrap(PhaseWrapped, discont=np.pi)
    return PhaseUnWrapped
