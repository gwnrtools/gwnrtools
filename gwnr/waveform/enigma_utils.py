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
import numpy as np
import time
import gwnr
import lal
import lalsimulation as ls
import pycbc.types as pt


def find_x_for_y(x, y, y0):
    return x[abs(y - y0).argmin()]


def eccentricity_at_extremum_frequency(
    mass1,
    mass2,
    spin1z,
    spin2z,
    e0,
    l0,
    f_lower,
    sample_rate,
    f_extremum,
    extremum="periastron",
    show_figures=False,
    verbose=False,
):
    """ """
    if extremum.lower() not in ["periastron", "apastron"]:
        raise IOError("Allowed values for extremum: periastron, apastron")

    itime = time.time()
    retval = ls.SimInspiralENIGMADynamics(
        mass1, mass2, spin1z, spin2z, e0, f_lower, l0, 1e-12, sample_rate, False
    )
    t, x, e, l, phi, phidot, r, rdot = retval[:8]
    t.data.data *= lal.MTSUN_SI

    time_delta = time.time() - itime
    if verbose:
        print("Orbital evolution took: {} seconds".format(time_delta))

    omega = pt.TimeSeries(phidot.data.data, delta_t=t.data.data[1] - t.data.data[0])
    if extremum == "periastron":
        extremum_frequencies_times, extremum_frequencies = gwnr.waveform.get_peak_freqs(
            omega
        )
    else:
        extremum_frequencies_times, extremum_frequencies = gwnr.waveform.get_peak_freqs(
            -1 * omega
        )
        extremum_frequencies *= -1

    piMf_extremum = f_extremum * lal.PI * (mass1 + mass2) * lal.MTSUN_SI
    time_at_sensitive_freq = find_x_for_y(
        extremum_frequencies_times, extremum_frequencies, piMf_extremum
    )
    idx_e0 = abs(t.data.data - time_at_sensitive_freq).argmin()
    e0 = e.data.data[idx_e0]

    if show_figures:
        import matplotlib.pyplot as plt

        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
        ax2 = ax1
        ax1.plot(
            extremum_frequencies_times,
            extremum_frequencies,
            "o",
            markersize=1,
            label="extrema",
        )
        ax1.plot(t.data.data, x.data.data**1.5, "--", label="x ** {3/2}")
        ax1.plot(t.data.data, phidot.data.data, lw=0.5, label="omega")

        ax1.axhline(piMf_extremum, c="c", lw=1)
        ax1.axvline(time_at_sensitive_freq, c="r", lw=1)

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Orbital angular velocity")

        ax = plt.twinx(ax2)
        ax.plot(t.data.data, e.data.data, label="eccentricity", lw=1)
        ax.axhline(e0, c="k", lw=1)
        ax.set_xlim(ax1.get_xlim())

        ax1.legend(loc="center left")
        ax.legend(loc="upper center")

        return e0, fig

    return e0


def eccentricity_at_reference_frequency(
    mass1,
    mass2,
    spin1z,
    spin2z,
    e0,
    l0,
    f_lower,
    sample_rate,
    f_reference,
    show_figures=False,
    verbose=False,
):
    """ """
    itime = time.time()
    retval = ls.SimInspiralENIGMADynamics(
        mass1, mass2, spin1z, spin2z, e0, f_lower, l0, 1e-12, sample_rate, False
    )
    t, x, e, l, phi, phidot, r, rdot = retval[:8]
    t.data.data *= lal.MTSUN_SI

    time_delta = time.time() - itime
    if verbose:
        print("Orbital evolution took: {} seconds".format(time_delta))

    x_reference = (np.pi * (mass1 + mass2) * lal.MTSUN_SI * f_reference) ** (2.0 / 3.0)

    idx_e0 = abs(x.data.data - x_reference).argmin()
    e0 = e.data.data[idx_e0]

    if show_figures:
        import matplotlib.pyplot as plt

        fig, (ax) = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
        ax.plot(t.data.data, x.data.data**1.5, "--", label="x ** {3/2}")
        ax.plot(t.data.data, phidot.data.data, lw=0.5, label="omega")
        ax.axhline(x_reference**1.5, color="c", lw=1)
        ax.axvline(t.data.data[idx_e0], color="r", lw=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Orbital angular velocity")

        ax2 = plt.twinx(ax)
        ax2.plot(t.data.data, e.data.data, label="eccentricity", lw=1)
        ax2.axhline(e0, c="k", lw=1)
        ax2.set_xlim(ax.get_xlim())

        ax.legend(loc="center left")
        ax2.legend(loc="upper center")
        return e0, fig

    return e0


def get_imr_enigma_modes(
    mass1,
    mass2,
    spin1z,
    spin2z,
    eccentricity,
    mean_anomaly,
    distance,
    f_lower,
    sample_rate,
    modes_to_use=[(2, 2), (3, 3), (4, 4)],
    include_conjugate_modes=True,
    f_mr_transition=None,
    f_window_mr_transition=None,
    return_hybridization_info=False,
    verbose=False,
):
    if f_mr_transition is None:
        f_mr_transition = 6.0**-1.5 / (mass1 + mass2) / lal.MTSUN_SI / lal.PI
        # get_isco_frequency(m1, m2, s1z, s2z) / 2 / 2,
    if f_window_mr_transition is None:
        f_window_mr_transition = 10.0

    if distance < lal.PC_SI:
        distance = distance * lal.PC_SI

    itime = time.time()
    retval = ls.SimInspiralENIGMADynamics(
        mass1,
        mass2,
        spin1z,
        spin2z,
        eccentricity,
        f_lower,
        mean_anomaly,
        1e-12,
        sample_rate,
        False,
    )
    t, x, e, l, phi, phidot, r, rdot = retval[:8]
    t.data.data *= lal.MTSUN_SI
    time_delta = time.time() - itime
    if verbose:
        print("Orbital evolution took: {} seconds".format(time_delta))

    # Include conjugate modes in the mode list
    if include_conjugate_modes:
        for el, em in modes_to_use:
            if (el, -em) not in modes_to_use:
                modes_to_use.append((el, -em))

    itime = time.time()
    modes = {}
    for el, em in modes_to_use:
        modes[(el, em)] = ls.SimInspiralENIGMAModeFromDynamics(
            el,
            em,
            t.data,
            x.data,
            phi.data,
            phidot.data,
            r.data,
            rdot.data,
            mass1,
            mass2,
            spin1z,
            spin2z,
            distance,
        )

    modes_numpy = {k: np.asarray(modes[k].data.data) for k in modes}
    time_delta = time.time() - itime
    if verbose:
        print("Modes generation took: {} seconds".format(time_delta))

    # SimInspiralChooseTDModes(
    # REAL8 phiRef, REAL8 deltaT,
    # REAL8 m1, REAL8 m2,
    # REAL8 S1x, REAL8 S1y, REAL8 S1z, REAL8 S2x, REAL8 S2y, REAL8 S2z,
    # REAL8 f_min, REAL8 f_ref, REAL8 r, Dict LALpars, int lmax,
    # Approximant approximant) -> SphHarmTimeSeries
    hlm_mr = ls.SimInspiralChooseTDModes(
        0,
        1.0 / sample_rate,
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        0,
        0,
        spin1z,
        0,
        0,
        spin2z,
        (f_mr_transition - f_window_mr_transition) * 0.9,
        (f_mr_transition - f_window_mr_transition) * 0.9,
        distance,
        None,
        4,
        ls.NRSur7dq4,
    )

    modes_mr = {}
    while hlm_mr is not None:
        modes_mr[(hlm_mr.l, hlm_mr.m)] = hlm_mr.mode
        hlm_mr = hlm_mr.next

    modes_mr_numpy = {k: np.asarray(modes_mr[k].data.data) for k in modes_mr}

    retval = gwnr.waveform.hybridize.hybridize_modes(
        modes_numpy,
        modes_mr_numpy,
        f_mr_transition,
        f_window_mr_transition,
        1.0 / sample_rate,
        modes_to_hybridize=modes_to_use,
        verbose=verbose,
    )
    modes_imr_numpy = retval[0]

    # Align modes at peak of (2, 2) mode
    mode_to_align_by = (2, 2)
    if mode_to_align_by not in modes_imr_numpy:
        mode_to_align_by = list(modes_imr_numpy.keys())[0]
    idx_peak = abs(modes_imr_numpy[mode_to_align_by]).argmax()
    t_peak = idx_peak / sample_rate

    itime = time.time()
    modes_imr = {}
    for el, em in modes_imr_numpy:
        modes_imr[(el, em)] = pt.TimeSeries(
            modes_imr_numpy[(el, em)], delta_t=1.0 / sample_rate, epoch=-1 * t_peak
        )
    if verbose:
        print(
            "Time taken to store in pycbc.TimeSeries is {} secs".format(
                time.time() - itime
            )
        )

    if verbose:
        print("hybridized.")
    if return_hybridization_info:
        return modes_imr, retval
    return modes_imr


def get_imr_enigma_waveform(
    mass1,
    mass2,
    spin1z,
    spin2z,
    eccentricity,
    mean_anomaly,
    inclination,
    coa_phase,
    distance,
    f_lower,
    sample_rate,
    modes_to_use=[(2, 2), (3, 3), (4, 4)],
    f_mr_transition=None,
    f_window_mr_transition=None,
    return_hybridization_info=False,
    verbose=False,
):
    if distance < lal.PC_SI:
        distance = distance * lal.PC_SI

    retval = get_imr_enigma_modes(
        mass1,
        mass2,
        spin1z,
        spin2z,
        eccentricity,
        mean_anomaly,
        distance,
        f_lower,
        sample_rate,
        modes_to_use=modes_to_use,
        include_conjugate_modes=True,
        f_mr_transition=f_mr_transition,
        f_window_mr_transition=f_window_mr_transition,
        return_hybridization_info=return_hybridization_info,
        verbose=verbose,
    )
    if return_hybridization_info:
        modes_imr, retval = retval
    else:
        modes_imr = retval

    hp_ihc = modes_imr[(2, 2)] * 0  # Initialize with zeros
    if verbose:
        print("Shape of hp_ihc: {}".format(np.shape(hp_ihc)), flush=True)

    for el, em in modes_imr:
        ylm = lal.SpinWeightedSphericalHarmonic(inclination, coa_phase, -2, el, em)
        hp_ihc = hp_ihc + modes_imr[(el, em)] * ylm
        if verbose:
            print(f"Adding mode {el}, {em} with ylm = {ylm}", flush=True)
            print(
                "... adding {}, {}".format(
                    modes_imr[(el, em)], modes_imr[(el, em)] * ylm
                ),
                flush=True,
            )
            print(f"hp after adding: {hp_ihc.data}", flush=True)

    hp = hp_ihc.real()
    hc = -1 * hp_ihc.imag()

    if return_hybridization_info:
        return hp, hc, retval

    return hp, hc


class FitMOmegaIMRAttachmentNonSpinning:
    called_once = False

    def __init__(self):
        self.called_once = False
        return

    @classmethod
    def fit_quadratic_poly(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_quadratic_poly")
            cls.called_once = True
        assert len(coeffs) == 2, "{} coeffs passed!".format(len(coeffs))
        a1, a2 = coeffs
        return (1.0 / 6**1.5) * (1.0 + a1 * eta + a2 * eta**2)

    @classmethod
    def fit_cubic_poly(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_cubic_poly")
            cls.called_once = True
        assert len(coeffs) == 3, "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3 = coeffs
        return (1.0 / 6**1.5) * (1.0 + a1 * eta + a2 * eta**2 + a3 * eta**3)

    @classmethod
    def fit_ratio_poly_44(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_poly_44")
            cls.called_once = True
        assert len(coeffs) == 6, "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2, b3 = coeffs
        return (
            (1.0 / 6**1.5)
            * (1.0 + a1 * eta + a2 * eta**2 + a3 * eta**3)
            / (1.0 + b1 * eta + b2 * eta**2 + b3 * eta**3)
        )

    @classmethod
    def fit_ratio_sqrt_poly_44(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_sqrt_poly_44")
            cls.called_once = True
        assert len(coeffs) == 6, "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2, b3 = coeffs
        s_eta = eta**0.5
        return (
            (1.0 / 6**1.5)
            * (1.0 + a1 * s_eta + a2 * s_eta**2 + a3 * s_eta**3)
            / (1.0 + b1 * s_eta + b2 * s_eta**2 + b3 * s_eta**3)
        )

    @classmethod
    def fit_ratio_sqrt_hyb1_poly_44(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_sqrt_hyb1_poly_44")
            cls.called_once = True
        assert len(coeffs) == 6, "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2, b3 = coeffs
        s_eta = eta**0.5
        return (
            (1.0 / 6**1.5)
            * (1.0 + a1 * eta + a2 * eta**2 + a3 * eta**3)
            / (1.0 + b1 * eta + b2 * eta**2 + b3 * eta**3)
        )

    @classmethod
    def fit_ratio_poly_43(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_poly_43")
            cls.called_once = True
        assert len(coeffs) == 5, "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2 = coeffs
        return (
            (1.0 / 6**1.5)
            * (1.0 + a1 * eta + a2 * eta**2 + a3 * eta**3)
            / (1.0 + b1 * eta + b2 * eta**2)
        )

    @classmethod
    def fit_ratio_sqrt_poly_43(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_sqrt_poly_43")
            cls.called_once = True
        assert len(coeffs) == 5, "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2 = coeffs
        s_eta = eta**0.5
        return (
            (1.0 / 6**1.5)
            * (1.0 + a1 * s_eta + a2 * s_eta**2 + a3 * s_eta**3)
            / (1.0 + b1 * s_eta + b2 * s_eta**2)
        )

    @classmethod
    def fit_ratio_sqrt_hyb1_poly_43(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_sqrt_hyb1_poly_43")
            cls.called_once = True
        assert len(coeffs) == 5, "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2 = coeffs
        s_eta = eta**0.5
        return (
            (1.0 / 6**1.5)
            * (1.0 + a1 * eta * s_eta + a2 * eta**2 * s_eta + a3 * eta**3 * s_eta)
            / (1.0 + b1 * eta + b2 * eta**2)
        )

    @classmethod
    def fit_ratio_poly_34(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_poly_34")
            cls.called_once = True
        assert len(coeffs) == 5, "{} coeffs passed!".format(len(coeffs))
        a1, a2, b1, b2, b3 = coeffs
        return (
            (1.0 / 6**1.5)
            * (1.0 + a1 * eta + a2 * eta**2)
            / (1.0 + b1 * eta + b2 * eta**2 + b3 * eta**3)
        )
