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
from __future__ import absolute_import, print_function

import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize_scalar

import lal
from pycbc.types import FrequencySeries
from pycbc.waveform import (
    amplitude_from_polarizations,
    frequency_from_polarizations,
    phase_from_polarizations,
)
from pycbc.detector import overhead_antenna_pattern as generate_fplus_fcross
from pycbc.pnutils import *


def get_detector_response(ra, dec, psi, detector_tag, gmst=0):
    detMap = {
        "H1": lal.LALDetectorIndexLHODIFF,
        "H2": lal.LALDetectorIndexLHODIFF,
        "L1": lal.LALDetectorIndexLLODIFF,
        "G1": lal.LALDetectorIndexGEO600DIFF,
        "V1": lal.LALDetectorIndexVIRGODIFF,
        "T1": lal.LALDetectorIndexTAMA300DIFF,
        "AL1": lal.LALDetectorIndexLLODIFF,
        "AH1": lal.LALDetectorIndexLHODIFF,
        "AV1": lal.LALDetectorIndexVIRGODIFF,
    }
    detector = detMap[detector_tag]
    # get detector
    detval = lal.CachedDetectors[detector]
    # get its response Tensor
    response = detval.response
    # get plus and cross polarization response
    return lal.ComputeDetAMResponse(response, ra, dec, psi, gmst)


def generate_detector_strain(template_params, h_plus, h_cross):
    # {{{
    latitude = 0
    longitude = 0
    polarization = 0

    if hasattr(template_params, "latitude"):
        latitude = template_params.latitude
    else:
        latitude = template_params["latitude"]
    if hasattr(template_params, "longitude"):
        longitude = template_params.longitude
    else:
        longitude = template_params["longitude"]
    if hasattr(template_params, "polarization"):
        polarization = template_params.polarization
    else:
        polarization = template_params["polarization"]

    f_plus, f_cross = generate_fplus_fcross(longitude, latitude, polarization)

    return h_plus * f_plus + h_cross * f_cross
    # }}}


def get_ncycles_to_merger(hp, hc):
    if type(hp) == FrequencySeries:
        return -1
    a = amplitude_from_polarizations(hp, hc)
    p = phase_from_polarizations(hp, hc)
    idx = a.abs_max_loc()[-1]
    ncyc = np.abs(p[idx] - p[0]) / np.pi / 2.0
    return ncyc


def get_time_at_frequency_from_polarizations(hp, hc, fvalue):
    fr = frequency_from_polarizations(hp, hc)
    obj_func = np.abs(np.abs(fr) - fvalue)
    id_start = np.where(obj_func == np.min(obj_func))[0][0]
    for idx in range(id_start, len(fr)):
        if fr[idx] > 2 * fvalue and fr[idx + 1] > 2 * fvalue:
            break
    frI = InterpolatedUnivariateSpline(fr.sample_times, obj_func)
    tmp = minimize_scalar(
        frI,
        fr.sample_times[id_start],
        method="bounded",
        bounds=(fr.sample_times[id_start], fr.sample_times[idx]),
    )
    return tmp["x"]


def get_time_at_frequency(fr, fvalue):
    return get_time_at_y(fr, fvalue)


def get_freq_crossings(freq, f0, df_threshold=0.4):
    """
    Inputs
    ------
    freq: Array of similar iterable of frequency values
    f0:   Frequency value that one needs the crossing times for

    Output
    ------
    crossing_times: numpy.array
        Array of crossing times
    crossing_freqs: numpy.array
        Array of precise crossing frequencies. These may be slightly different
        from f0 given that the `freq` is discretely sampled
    """
    f0_crossing_times, f0_crossing_freqs = [], []
    for idx, finst in enumerate(freq):
        if idx == 0 or idx == len(freq) - 1:
            continue
        if (
            (np.abs(freq[idx - 1] - f0) > np.abs(finst - f0))
            and (np.abs(freq[idx + 1] - f0) > np.abs(finst - f0))
            and (np.abs(finst - f0) < df_threshold)
        ):
            f0_crossing_freqs.append(finst)
            f0_crossing_times.append(freq.sample_times[idx])
    return (np.array(f0_crossing_times), np.array(f0_crossing_freqs))


def get_time_at_y(fr, fvalue):
    """
    Finds the closest match to `fvalue` in a TimeSeries.
    Input a TimeSeries with epoch set correctly.
    """
    # Define time interval to be searched
    idx_first = int(len(fr) * 0.2)  # 20% margin for junk - TOO MUCH?
    idx_end = np.where(
        np.abs(fr.sample_times.data) == np.abs(fr.sample_times.data).min()
    )[0][
        0
    ]  # Assume a properly aligned TimeSeries
    # Starting guess
    obj_func = np.abs(np.abs(fr) - fvalue)[idx_first:idx_end]
    id_start = np.where(obj_func == np.min(obj_func))[0][
        int(np.ceil(len(np.where(obj_func == np.min(obj_func))[0]) / 2))
    ]
    # Interpolate and find
    frI = InterpolatedUnivariateSpline(fr.sample_times[idx_first:idx_end], obj_func)
    tmp = minimize_scalar(
        frI,
        fr.sample_times[id_start],
        method="bounded",
        bounds=(fr.sample_times[idx_first], fr.sample_times[idx_end]),
    )
    # Return time value
    return tmp["x"]


def get_isco_x(mass1, mass2, spin1z, spin2z, show_figure=False):
    def isco_eqn(
        x,
        mass1,
        mass2,
        spin1z,
        spin2z,
    ):
        """ """
        try:
            if x <= 0:
                return 1e99
        except:
            pass

        dm = mass1 - mass2
        total_mass = mass1 + mass2
        dm_over_m = dm / total_mass
        eta = mass1 * mass2 / total_mass / total_mass

        # S^c_{1,2} = s{1,2}z * m{1,2} * m{1,2}
        s_c_1 = spin1z * mass1 * mass1
        s_c_2 = spin2z * mass2 * mass2

        # s_c_l = ell.S^c = Z.S^c, with
        # S^c = S^c_1 + S^c_2
        s_c_l = s_c_1 + s_c_2

        # sigma_c_l = ell.sigma^c = Z.sigma_c, with
        # sigma^c = (M/m2) S^c_2 - (M/m1) S^c_1
        sigma_c_l = (total_mass / mass2) * s_c_2 - (total_mass / mass1) * s_c_1

        # s_c_0l = ell.S^c_0 = Z.S^c_0, with
        # S^c_0 = (1 + m2/m1) S^c_1 + (1 + m1/m2) S^_2
        s_c_0l = (1.0 + mass2 / mass1) * s_c_1 + (1.0 + mass1 / mass2) * s_c_2

        # Now, normalize all spin combinations with total_mass^2
        s_c_l /= total_mass**2
        sigma_c_l /= total_mass**2
        s_c_0l /= total_mass**2

        print(f"Spin combos: {s_c_l}, {sigma_c_l}, {s_c_0l} for ({spin1z}, {spin2z})")

        pn1p5 = 14 * s_c_l + 6.0 * dm_over_m * sigma_c_l
        pn2 = 14.0 * eta - 3 * s_c_0l**2
        pn2p5 = -(
            (22.0 + 32.0 * eta) * s_c_l + dm_over_m * sigma_c_l * (18.0 + 15.0 * eta)
        )
        pn3 = (397.0 / 2.0 - 123.0 * np.pi * np.pi / 16.0) * eta - 14.0 * eta**2
        return (
            1
            - 6.0 * x
            + pn1p5 * x**1.5
            + pn2 * x**2
            + pn2p5 * x**2.5
            + pn3 * x**3
        )

    res = minimize_scalar(
        isco_eqn, method="brent", bracket=[0.01, 1], args=(mass1, mass2, spin1z, spin2z)
    )
    print(f"Value of x at ISCO: {res.x}")
    if show_figure:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(
            np.arange(0, 1, 0.01),
            isco_eqn(np.arange(0, 1, 0.01), mass1, mass2, spin1z, spin2z),
        )
        plt.axvline(res.x, color="g", label="ISCO x")
        plt.axhline(res.fun, color="r", label="min potential")
        plt.axvline(1 / 6, color="k", label="r=6M")
        plt.legend()
        plt.ylim(0, 6)
        plt.xlabel("x")
        plt.ylabel("effective binary potential")
    return res.x


def get_isco_frequency(mass1, mass2, spin1z, spin2z):
    x_isco = get_isco_x(mass1, mass2, spin1z, spin2z)
    m_omg_isco = x_isco**1.5
    f_isco = m_omg_isco / (lal.PI * (mass1 + mass2) * lal.MTSUN_SI)
    print(f"Value of f at ISCO: {f_isco}Hz")
    return f_isco
