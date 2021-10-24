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
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
from __future__ import print_function

import os
import sys
import glob
import subprocess
import numpy as np
from scipy.optimize import minimize

from pycbc.waveform import frequency_from_polarizations
from gwnr.waveform.align import align_curves


def get_peak_freqs(freq):
    '''
    Inputs
    ------
    freq: Array of similar iterable of frequency values

    Outputs
    -------
    peak_times: Times at which local maxima of frequency are attained
    peak_freqs: Frequency at these times
    '''
    peaks, peak_times = [], []
    fvals = freq.data

    for idx, finst in enumerate(fvals):
        if idx == 0 or idx == len(fvals) - 1:
            continue

        if ((fvals[idx - 1]) < (finst)) and ((fvals[idx + 1]) < (finst)):
            peaks.append(finst)
            peak_times.append(freq.sample_times[idx])

    return np.array(peak_times), np.array(peaks)


def get_periastron_frequencies(hp, hc):
    '''
    Input
    -----
    hp, hc: pycbc.types.timeseries
            Plus (+) and Cross (x) polarizations

    Output:
    -------
    periastron_frequencies: numpy.array
        Frequency of GW emission at periastron at increasing values of time
    periastron_times: numpy.array
        Times at which periastron passages occur
    '''
    freq = frequency_from_polarizations(hp, hc)
    ft, ff = get_peak_freqs(freq)
    return (ft, ff)


def get_apastron_frequencies(hp, hc):
    '''
    Input
    -----
    hp, hc: pycbc.types.timeseries
            Plus (+) and Cross (x) polarizations

    Output:
    -------
    apastron_frequencies: numpy.array
        Frequency of GW emission at periastron at increasing values of time
    apastron_times: numpy.array
        Times at which apastron passages occur
    '''
    freq = frequency_from_polarizations(hp, hc)
    ft, ff = get_peak_freqs(-1 * freq)
    return (ft, -1 * ff)


os.environ['LD_LIBRARY_PATH'] =\
    '/home/prayush/research/Eccentric_IMRGPR/Code/MergerRingdownModel/C_implementation/bin/'


def get_eccentric_waveform_and_dynamics(
        m1_min,
        m1_max,
        m1_nbins,
        m2_min,
        m2_max,
        m2_nbins,
        e_min,
        e_max,
        e_nbins,
        f_lower,
        delta_t,
        EXE='export LD_LIBRARY_PATH=/home/prayush/src/EccIMR/code/MergerRingdownModel/C_implementation/bin/:${LD_LIBRARY_PATH} && /home/prayush/src/EccIMR/code/map_link_codes/bbhall -d',
        mean_anomaly=0,
        inclination=0,
        init_phase=0,
        tolerance=1e-12,
        verbose=False):
    """
    This function computes an eccentric inspiral, and returns both the coordinate
    trajectory information as well as the GW polarizations.
    """
    # {{{
    if m1_nbins != 1 or m2_nbins != 1 or e_nbins != 1:
        raise IOError(
            "Function does not support generating multiple waveforms at present. Call `generate_eccentric_waveforms` instead."
        )

    from gwnr.nr.spec import ParseHeaderForSpECTabularOutputASCII

    cmd_string = "%s -m %.18e -M %.18e -x %d -n %.18e -N %.18e -y %d -e %.18e -E %.18e -z %d" %\
        (EXE, m1_min, m1_max, m1_nbins, m2_min,
         m2_max, m2_nbins, e_min, e_max, e_nbins)
    cmd_string += " -a %.18e -i %.18e -b %.18e -t %.18e -f %.18e -s %.18e" %\
        (mean_anomaly, inclination, init_phase, tolerance, f_lower, 1./delta_t)
    output_file_tag = 'tmp_%06d' % int(np.random.random() * 1e7)
    cmd_string += " -o %s -v" % output_file_tag
    if verbose:
        print("Command being run: %s" % cmd_string, file=sys.stdout)
        sys.stdout.flush()
    cmd_output = subprocess.getoutput(cmd_string)
    if verbose:
        print(cmd_output, file=sys.stdout)
        sys.stdout.flush()
    ##
    dynamics_filename = sorted(glob.glob(output_file_tag + "*" +
                                         "Dynamics*"))[-1]
    if verbose:
        print("Reading dynamics from: %s" % dynamics_filename, file=sys.stdout)
        sys.stdout.flush()
    ##
    dynamics_headers, _ = ParseHeaderForSpECTabularOutputASCII(
        dynamics_filename, separate_quantities_from_subdomains=False)
    dynamics_data = np.loadtxt(dynamics_filename)
    dynamics = {}
    for idx in dynamics_headers:
        label = dynamics_headers[idx].split()[0]
        if verbose:
            print("reading %s from col %d" % (label, idx))
        dynamics[label] = dynamics_data[:, idx]
    ##
    wave_filename = sorted(glob.glob(output_file_tag + "*"))[0]
    if verbose:
        print("Reading waveform from: %s" % wave_filename, file=sys.stdout)
        sys.stdout.flush()
    ##
    wave_data = np.loadtxt(wave_filename)
    if verbose:
        print("Reading polarization timeseries from OBJ of shape: ",
              np.shape(wave_data),
              file=sys.stdout)
    dynamics['h_Time'] = wave_data[:, 0]
    dynamics['hp'] = wave_data[:, 1]
    dynamics['hc'] = wave_data[:, 2]
    ##
    cmd_string = 'rm -f %s %s' % (wave_filename, dynamics_filename)
    subprocess.getoutput(cmd_string)
    return dynamics
    # }}}


def optimize_eccentricity(x1,
                          y1,
                          q,
                          use_var="r",
                          EXE=None,
                          x_low_lim=None,
                          x_high_lim=None,
                          ecc_low=0,
                          ecc_high=0.1,
                          ecc_init=0,
                          anom_low=0,
                          anom_high=6.28318,
                          anom_init=0,
                          sample_rate=4096,
                          f_lower=30.0,
                          method='Nelder-Mead',
                          objective_scaling_fac=None,
                          num_retries=1,
                          verbose=True,
                          debug=False):
    """
    Given a trajectory, this function computes the eccentricity and initial
    mean anomaly for a binary (at f_low Hz) that optimizes the agreement
    between its radial evolution and the trajectory.

    [Goal]
    To minimize :
        f(x_offset,
            e0,
            anom0) := \int_{x_low_lim}^{x_high_lim}
                        |y2(x + x_offset; e0, anom0) - y1(x)| dx

    over {x_offset, e0, anom0). This is done in two steps:-

    1) f(x_offset, e0, anom0) is minimized over x_offset using align_curves,
    to get fOpt(x0, anom0)

    2) fOpt(e0, anom0) is minimized over {e0, anom0} using scipy.minimize here.


    [Notes]

    1) [x_low_lim, x_high_lim] are with respect to the (x1, y1) pair.
    The other pair (x1, y2) is the one effectively shifted.

    2) Not specifying [x_low_lim, x_high_lim] is equivalent to integrating
    the mean-square difference over the complete (x2) vector.
    """
    # {{{
    if use_var != "r" and use_var != "omega":
        raise RuntimeError(
            "Which variable to use for eccentricity optimization")
    if objective_scaling_fac is None:
        if use_var == "r":
            objective_scaling_fac = 1.0
        else:
            objective_scaling_fac = 1e5
    #

    def objective_function_eccentricity(x, *args):
        anom0, e0 = x
        x1, y1, q = args
        lbl = '%.12f,%.12f' % (e0, anom0)
        if lbl not in list(objective_function_eccentricity.waves.keys()):
            try:
                if EXE is None:
                    retval = get_eccentric_waveform_and_dynamics(
                        20 * q,
                        20 * q,
                        1,
                        20,
                        20,
                        1,
                        e0,
                        e0,
                        1,
                        f_lower,
                        1. / sample_rate,
                        mean_anomaly=anom0,
                        verbose=debug)
                else:
                    retval = get_eccentric_waveform_and_dynamics(
                        20 * q,
                        20 * q,
                        1,
                        20,
                        20,
                        1,
                        e0,
                        e0,
                        1,
                        f_lower,
                        1. / sample_rate,
                        mean_anomaly=anom0,
                        EXE=EXE,
                        verbose=debug)
                objective_function_eccentricity.waves[lbl] = retval
            except:
                return 1e99
        else:
            retval = objective_function_eccentricity.waves[lbl]
        if use_var == "omega":
            used_var = retval['phidot']
        elif use_var == "r":
            used_var = retval['r']
        shift, res = align_curves(x1,
                                  y1,
                                  retval['Time'],
                                  used_var,
                                  x_low_lim=x_low_lim,
                                  x_high_lim=x_high_lim,
                                  offset_low_lim=-400,
                                  offset_high_lim=-100)
        if objective_function_eccentricity.counter % 1 == 0:
            print("trying out: e0 = %.12f, anom0 = %.12f, objective = %.12f" %
                  (e0, anom0, objective_scaling_fac * res.fun))
        objective_function_eccentricity.counter += 1
        return res.fun * objective_scaling_fac

    objective_function_eccentricity.counter = 0
    objective_function_eccentricity.waves = {}
    ###
    # CALL THE scipy.optimize.minimize TO COMPUTE OPTIMAL ECCENTRICITY & INIT MEAN ANOMALY
    opt_args = (x1, y1, q)

    if debug:
        print("Testing objective function")
        print("Offset 0: ", objective_function_eccentricity([0, 0], *opt_args))
        print("Offset 550: ",
              objective_function_eccentricity([np.pi, 0.01], *opt_args))

    for idx in range(num_retries):
        if verbose:
            print("\nTry %d to compute optimal eccentricity" % idx,
                  file=sys.stdout)
            sys.stdout.flush()
        ##
        bnds = ((anom_low, anom_high), (ecc_low, ecc_high))
        retval = minimize(objective_function_eccentricity,
                          [anom_init, ecc_init],
                          bounds=bnds,
                          method=method,
                          args=opt_args)
        anom_init, ecc_init = retval.x
    ##
    if verbose:
        print("optimization took %d objective func evals" %
              objective_function_eccentricity.counter,
              file=sys.stdout)
        sys.stdout.flush()
    # 7) RETURN OPTIMIZED PARAMETERS
    return [retval.x, retval]
    # }}}
