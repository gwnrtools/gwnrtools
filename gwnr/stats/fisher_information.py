# Copyright (C) 2017 Prayush Kumar
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
"""WAVEFORM DERIVATIVES & FISHER MATRIX CALCULATIONS"""

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

from numpy import *
import numpy as np
import scipy as sp
import time

from pycbc.detector import *
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.types import FrequencySeries, TimeSeries

_itime = time.time()
verbose = True


#############################
# Function to compute derivatives dh/dth
def get_waveform_derivatives_wrt_params(
        approximant='SEOBNRv2',
        mass1=None,
        mass2=None,
        spin1x=0,
        spin2x=0,
        spin1y=0,
        spin2y=0,
        spin1z=0,
        spin2z=0,
        distance=1,
        coa_phase=0,
        inclination=0,
        latitude=0,
        longitude=0,
        polarization=0,
        f_lower=None,
        sample_rate=None,
        time_length=4,
        deriv_params=['mass1', 'mass2', 'spin1z', 'spin2z'],
        delta_m1=1e-3,
        delta_m2=1e-3,
        delta_s1=1e-4,
        delta_s2=1e-4,
        abs_tolerance=1e-7,
        verbose=True):
    """
This function computes dh/dtheta , where theta = {m1, m2, spin1z, spin2z, ..., }.

mass1 : {None, float}
    The mass of the first component object in the binary (in solar masses).
mass2 : {None, float}
    The mass of the second component object in the binary (in solar masses).
spin1x : {0.0, float}
    The x component of the first binary component's dimensionless spin.
spin1y : {0.0, float}
    The y component of the first binary component's dimensionless spin.
spin1z : {0.0, float}
    The z component of the first binary component's dimensionless spin.
spin2x : {0.0, float}
    The x component of the second binary component's dimensionless spin.
spin2y : {0.0, float}
    The y component of the second binary component's dimensionless spin.
spin2z : {0.0, float}
    The z component of the second binary component's dimensionless spin.
distance : {1.0, float}
    Luminosity distance to the binary (in Mpc).
coa_phase : {0.0, float}
    Coalesence phase of the binary (in rad).
inclination : {0.0, float}
    Inclination (rad), defined as the angle between the total angular momentum J and the line-of-sight.

h(t) = A(t - tc) * Exp(-i * Phi(t - tc) ), where tc \equiv 0 is the time of merger.
    """
    param_list = {}
    param_list['mass1'] = mass1
    param_list['mass2'] = mass2
    param_list['spin1x'] = spin1x
    param_list['spin1y'] = spin1y
    param_list['spin1z'] = spin1z
    param_list['spin2x'] = spin2x
    param_list['spin2y'] = spin2y
    param_list['spin2z'] = spin2z
    param_list['distance'] = distance
    param_list['coa_phase'] = coa_phase
    param_list['inclination'] = inclination
    param_list['latitude'] = latitude
    param_list['longitude'] = longitude
    param_list['polarization'] = polarization

    delta_t = 1. / sample_rate
    delta_f = 1. / time_length
    N = sample_rate * time_length
    n = N / 2 + 1

    def FUNC(x):
        # Substitute the right parameter
        deriv_param = param_list['deriv_param']
        param_list[deriv_param] = x
        # Compute the waveform
        if approximant in td_approximants():
            test_hp, test_hc = get_td_waveform(
                approximant=approximant,
                mass1=param_list['mass1'],
                mass2=param_list['mass2'],
                spin1x=param_list['spin1x'],
                spin1y=param_list['spin1y'],
                spin1z=param_list['spin1z'],
                spin2x=param_list['spin2x'],
                spin2y=param_list['spin2y'],
                spin2z=param_list['spin2z'],
                distance=param_list['distance'],
                inclination=param_list['inclination'],
                coa_phase=param_list['coa_phase'],
                f_lower=f_lower,
                delta_t=delta_t)
            test_wav = generate_detector_strain(param_list, test_hp, test_hc)
            test_wav = extend_waveform_TimeSeries(test_wav, N)
        elif approximant in fd_approximants():
            test_hp, test_hc = get_fd_waveform(
                approximant=approximant,
                mass1=param_list['mass1'],
                mass2=param_list['mass2'],
                spin1x=param_list['spin1x'],
                spin1y=param_list['spin1y'],
                spin1z=param_list['spin1z'],
                spin2x=param_list['spin2x'],
                spin2y=param_list['spin2y'],
                spin2z=param_list['spin2z'],
                distance=param_list['distance'],
                inclination=param_list['inclination'],
                coa_phase=param_list['coa_phase'],
                f_lower=f_lower,
                delta_f=delta_f)
            test_wav = generate_detector_strain(param_list, test_hp, test_hc)
            test_wav = extend_waveform_FrequencySeries(test_wav, n)
        # Return waveform
        return np.array(test_wav.data)

    derivatives = {}
    for idx, deriv_param in enumerate(deriv_params):
        if verbose:
            print("Computing derivative w.r.t. %s (%d / %d)" %
                  (deriv_param, idx + 1, len(deriv_params)))
        param_list['deriv_param'] = deriv_param
        deriv_param_value = param_list[deriv_param]
        dx = 1e-4 * deriv_param_value
        if dx < abs_tolerance:
            dx = abs_tolerance
        dhdparam = sp.misc.derivative(FUNC,
                                      deriv_param_value,
                                      dx=dx,
                                      n=1,
                                      order=5)
        derivatives[deriv_param] = dhdparam

    return derivatives


#############################


def get_correlation_fisher_matrices(
        approximant='SEOBNRv2',
        mass1=None,
        mass2=None,
        spin1x=0,
        spin2x=0,
        spin1y=0,
        spin2y=0,
        spin1z=0,
        spin2z=0,
        distance=1,
        coa_phase=0,
        inclination=0,
        latitude=0,
        longitude=0,
        polarization=0,
        f_lower=None,
        f_upper=None,
        sample_rate=None,
        time_length=4,
        deriv_params=['mass1', 'mass2', 'spin1z', 'spin2z'],
        psd='aLIGOZeroDetHighPower',
        delta_m1=1e-3,
        delta_m2=1e-3,
        delta_s1=1e-4,
        delta_s2=1e-4,
        abs_tolerance=1e-7,
        return_derivs=False,
        verbose=True):
    #
    # 0) Precompute parameters
    delta_t = 1. / sample_rate
    delta_f = 1. / time_length
    N = sample_rate * time_length
    n = N / 2 + 1

    #
    # 1) Get derivatives of waveform first.
    if approximant in td_approximants():
        out_type = TimeSeries
    elif approximant in fd_approximants():
        out_type = FrequencySeries
    derivs = get_waveform_derivatives_wrt_params(approximant=approximant,
                                                 mass1=mass1,
                                                 mass2=mass2,
                                                 spin1x=spin1x,
                                                 spin2x=spin2x,
                                                 spin1y=spin1y,
                                                 spin2y=spin2y,
                                                 spin1z=spin1z,
                                                 spin2z=spin2z,
                                                 distance=distance,
                                                 coa_phase=coa_phase,
                                                 inclination=inclination,
                                                 latitude=latitude,
                                                 longitude=longitude,
                                                 polarization=polarization,
                                                 f_lower=f_lower,
                                                 sample_rate=sample_rate,
                                                 time_length=time_length,
                                                 deriv_params=deriv_params,
                                                 delta_m1=1e-3,
                                                 delta_m2=1e-3,
                                                 delta_s1=1e-4,
                                                 delta_s2=1e-4,
                                                 abs_tolerance=abs_tolerance,
                                                 verbose=verbose)
    #
    # 2) Compute noise PSD
    if FrequencySeries == type(psd):
        pass
    elif type(psd) == str:
        psd_name = psd
        psd = from_string(psd_name, n, delta_f, f_lower)
    else:
        raise IOError(
            "Either provide a psd compatible with waveforms (difficult) or a string"
        )

    #
    # 3) Compute Correlations
    nrows = ncols = len(deriv_params)
    correlation_matrix = np.zeros((nrows, ncols))

    # Loop over outer indices
    for idx, outer_param in enumerate(deriv_params):
        if type(derivs[outer_param]) == out_type:
            outer_vec = derivs[outer_param]
        else:
            derivs[outer_param] = convert_numpy_to_pycbc_type(
                derivs[outer_param],
                out_type,
                sample_rate=sample_rate,
                time_length=time_length)
            outer_vec = derivs[outer_param]
        # Loop over inner indices
        for jdx, inner_param in enumerate(deriv_params):
            if type(derivs[inner_param]) == out_type:
                inner_vec = derivs[inner_param]
            else:
                derivs[inner_param] = convert_numpy_to_pycbc_type(
                    derivs[inner_param],
                    out_type,
                    sample_rate=sample_rate,
                    time_length=time_length)
                inner_vec = derivs[inner_param]
            # Compute inner products
            olap = overlap_cplx(inner_vec,
                                outer_vec,
                                psd=psd,
                                low_frequency_cutoff=f_lower,
                                high_frequency_cutoff=f_upper,
                                normalized=False)
            # np.abs(olap) # FIXME ABS OR REAL ?
            correlation_matrix[jdx, idx] = olap.real
    #
    # 4) Compute Inverse of Correlation matrix
    try:
        fisher_matrix = np.linalg.inv(correlation_matrix)
    except:
        fisher_matrix = None
        print(
            "Warning: Could not invert correlation matrix, Fisher matrix uncomputable."
        )

    #
    # 5) RETURN
    if return_derivs:
        return correlation_matrix, fisher_matrix, derivs
    return correlation_matrix, fisher_matrix
