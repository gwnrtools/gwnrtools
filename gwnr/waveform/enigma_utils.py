# Copyright (C) 2020 Prayush Kumar
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
"""Fitting functions specific to ENIGMA"""

from __future__ import absolute_import

import logging


class FitMOmegaIMRAttachmentNonSpinning():
    called_once = False

    def __init__(self):
        self.called_once = False
        return

    @classmethod
    def fit_quadratic_poly(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_quadratic_poly")
            cls.called_once = True
        assert (len(coeffs) == 2), "{} coeffs passed!".format(len(coeffs))
        a1, a2 = coeffs
        return (1. / 6**1.5) * (1. + a1 * eta + a2 * eta**2)

    @classmethod
    def fit_cubic_poly(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_cubic_poly")
            cls.called_once = True
        assert (len(coeffs) == 3), "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3 = coeffs
        return (1. / 6**1.5) * (1. + a1 * eta + a2 * eta**2 + a3 * eta**3)

    @classmethod
    def fit_ratio_poly_44(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_poly_44")
            cls.called_once = True
        assert (len(coeffs) == 6), "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2, b3 = coeffs
        return (1. / 6**1.5) * (1. + a1 * eta + a2 * eta**2 + a3 * eta**3) / (
            1. + b1 * eta + b2 * eta**2 + b3 * eta**3)

    @classmethod
    def fit_ratio_sqrt_poly_44(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_sqrt_poly_44")
            cls.called_once = True
        assert (len(coeffs) == 6), "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2, b3 = coeffs
        s_eta = eta**0.5
        return (1. /
                6**1.5) * (1. + a1 * s_eta + a2 * s_eta**2 + a3 * s_eta**3) / (
                    1. + b1 * s_eta + b2 * s_eta**2 + b3 * s_eta**3)

    @classmethod
    def fit_ratio_sqrt_hyb1_poly_44(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_sqrt_hyb1_poly_44")
            cls.called_once = True
        assert (len(coeffs) == 6), "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2, b3 = coeffs
        s_eta = eta**0.5
        return (1. / 6**1.5) * (1. + a1 * eta + a2 * eta**2 + a3 * eta**3) / (
            1. + b1 * eta + b2 * eta**2 + b3 * eta**3)

    @classmethod
    def fit_ratio_poly_43(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_poly_43")
            cls.called_once = True
        assert (len(coeffs) == 5), "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2 = coeffs
        return (1. / 6**1.5) * (1. + a1 * eta + a2 * eta**2 +
                                a3 * eta**3) / (1. + b1 * eta + b2 * eta**2)

    @classmethod
    def fit_ratio_sqrt_poly_43(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_sqrt_poly_43")
            cls.called_once = True
        assert (len(coeffs) == 5), "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2 = coeffs
        s_eta = eta**0.5
        return (1. / 6**1.5) * (1. + a1 * s_eta + a2 * s_eta**2 + a3 *
                                s_eta**3) / (1. + b1 * s_eta + b2 * s_eta**2)

    @classmethod
    def fit_ratio_sqrt_hyb1_poly_43(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_sqrt_hyb1_poly_43")
            cls.called_once = True
        assert (len(coeffs) == 5), "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2 = coeffs
        s_eta = eta**0.5
        return (1. / 6**1.5) * (1. + a1 * eta * s_eta + a2 * eta**2 * s_eta +
                                a3 * eta**3 * s_eta) / (1. + b1 * eta +
                                                        b2 * eta**2)

    @classmethod
    def fit_ratio_poly_34(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_poly_34")
            cls.called_once = True
        assert (len(coeffs) == 5), "{} coeffs passed!".format(len(coeffs))
        a1, a2, b1, b2, b3 = coeffs
        return (1. / 6**1.5) * (1. + a1 * eta + a2 * eta**2) / (
            1. + b1 * eta + b2 * eta**2 + b3 * eta**3)
