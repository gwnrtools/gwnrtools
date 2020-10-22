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
from __future__ import (absolute_import, print_function)

import numpy as np


def spins_to_PNeffective_spin(m1, m2, chi1, chi2):
    chieff = (113.*m1*m1*chi1 + 113.*m2*m2*chi2 + 75.*m1*m2*(chi1 + chi2)) /\
        (113. * (m1 + m2)**2)
    return chieff


def spins_to_2PNeffective_spin(m1, m2, chi1, chi2):
    q1, q2 = 1, 1
    num = (1. + 80.*q1) * m1**2 * chi1**2 + (1. + 80.*q2) * m2**2 * chi2**2 \
        + 158. * m1*m2*chi1*chi2
    den = 16. * (m1 + m2)**2
    return num / den


def spins_to_massweighted_spin(m1, m2, chi1, chi2):
    chiwt = m1 * chi1 + m2 * chi2
    return chiwt / (m1 + m2)


def spins_to_damoureffective_spin(m1, m2, chi1, chi2):
    chiwt = 4 * m1**2 * chi1 + 4 * m2**2 * chi2 + 3. * m1 * m2 * (chi1 + chi2)
    return chiwt / 4. / (m1 + m2)**2


def chip_from_masses_spins(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z):
    """
    Compute the IMRPhenomPv2 chi-precessing "chi_p" parameter, given
    component masses and spins for a binary.

    NOTE: Assumes convention m1 > m2
    """
    m1_2, m2_2 = m1**2, m2**2
    # Magnitude of the spin projections in the orbital plane */
    S1_perp = m1_2 * np.sqrt(s1x * s1x + s1y * s1y)
    S2_perp = m2_2 * np.sqrt(s2x * s2x + s2y * s2y)
    # /* From this we can compute chip*/
    A1 = 2. + (3. * m2) / (2 * m1)
    A2 = 2. + (3. * m1) / (2 * m2)
    ASp1 = A1 * S1_perp
    ASp2 = A2 * S2_perp
    if type(m1) != float and type(m1) != int:
        num = np.zeros(len(m1))
        den = np.zeros(len(m1))
        #
        mask = ASp2 > ASp1
        num[mask] = ASp2[mask]
        mask = ASp2 <= ASp1
        num[mask] = ASp1[mask]
        #
        den = A1 * m1_2
    else:
        if ASp2 > ASp1:
            num = ASp2
        else:
            num = ASp1
        if m2 > m1:
            den = A2 * m2_2
        else:
            den = A1 * m1_2
    # chip = max(A1 Sp1, A2 Sp2) / (A_i m_i^2) for i index of larger BH (See Eqn. 32 in technical document)
    chip = num / den
    return chip


def q_to_eta(q):
    return q / (1. + q)**2


def eta_to_q(eta):
    a = c = 1.
    b = 2. - 1. / np.array(eta)
    D = (b**2 - 4. * a * c)**0.5
    # print type(a), type(b), type(c), type(D)
    roots = [(-b + D) / (2. * a), (-b - D) / (2. * a)]
    return roots[0]
