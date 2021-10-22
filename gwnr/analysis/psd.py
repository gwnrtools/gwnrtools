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
"""Utilities for various actions on PSD measured from data"""

import numpy
import scipy
from pycbc.types import FrequencySeries


def resample_and_extrapolate_psd(
        freq_vals,
        psd_vals,
        delta_f,
        f_max,
        precision=None,
        interpolation_func=scipy.interpolate.interp1d):
    """Resamples given psd(f) data to a uniform grid with spacing
    equal to delta_f provided. Also extrapolates the same to
    f = 0 if data is not given below some physically measurable
    value of frequency.

    Parameters
        ----------
        freq_vals, psd_vals: array
            Frequency values and corresponding PSD values
        delta_f: float
            Desired sampling interval in frequency

        Returns
        -------
        psd: pycbc.types.FrequencySeries
            Resampled PSD.
    """
    assert len(freq_vals) == len(psd_vals),\
        "Length of frequency and PSD arrays provided are different: {0} and {1}".format(
        len(freq_vals), len(psd_vals)
    )

    psd_interp = interpolation_func(freq_vals, psd_vals)

    n = int(numpy.round(f_max / delta_f))
    interpolated_psd = FrequencySeries(numpy.zeros(n),
                                       delta_f=delta_f,
                                       dtype=precision)
    interpolated_freq_vals = numpy.array(interpolated_psd.sample_frequencies)

    # Assume monotonic freq_vals
    data_f_min = freq_vals[0]
    data_f_max = freq_vals[-1]
    data_f_mask = (interpolated_freq_vals >=
                   data_f_min) & (interpolated_freq_vals <= data_f_max)
    interpolated_psd.data[data_f_mask] = psd_interp(
        interpolated_freq_vals[data_f_mask])

    data_f_min_mask = (interpolated_freq_vals < data_f_min)
    interpolated_psd.data[data_f_min_mask] = psd_vals[0]

    data_f_max_mask = (interpolated_freq_vals > data_f_max)
    interpolated_psd.data[data_f_max_mask] = psd_vals[-1]
    return interpolated_psd
