# Copyright (C) 2015 Prayush Kumar
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

from gwnr.utils.types import extend_waveform_FrequencySeries, extend_waveform_TimeSeries
from gwnr.waveform.waveform import get_waveform

import os
import sys
import numpy as np
import pyswarm

import pycbc
from pycbc.psd import from_string
from pycbc.filter import match, make_frequency_series
import pycbc.pnutils as pnutils
import pycbc.waveform.generator as pywfg
import pycbc.waveform as pywf

from glue.ligolw import lsctables
from glue.ligolw import ligolw


@lsctables.use_in
class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
    pass


def calculate_faithfulness(
    m1,
    m2,
    s1x=0,
    s1y=0,
    s1z=0,
    s2x=0,
    s2y=0,
    s2z=0,
    tc=0,
    phic=0,
    inclination=0,
    ra=0,
    dec=0,
    polarization=0,
    signal_approx="IMRPhenomD",
    signal_file="",
    signal_h=None,
    tmplt_approx="IMRPhenomC",
    tmplt_file="",
    tmplt_h=None,
    aligned_spin_tmplt_only=True,
    non_spin_tmplt_only=False,
    f_lower=15.0,
    sample_rate=4096,
    signal_duration=32,
    psd_string="aLIGOZeroDetHighPower",
    verbose=True,
    debug=False,
):
    """
    Calculates the match for a signal of given physical
    parameters, as modelled by a given signal approximant, against
    templates of another approximant.

    This function allows turning off x,y components of
    spin for templates.

    IN PROGRESS: Adding facility to use "FromDataFile" waveforms
    Inputs
    ------

    m1: float
        Mass of more massive black hole in binary .
    m2: float
        Mass of less massive black hole in binary.
    s1x: {0, float}
        Spin component along x-axis of more massive black hole.
    s1y: {0, float}
        Spin component along y-axis of more massive black hole.
    s1z: {0, float}
        Spin component along z-axis of more massive black hole. This
        is also the axis along which the orbital angular momentum points.
    s1x: {0, float}
        Spin component along x-axis of less massive black hole.
    s1y: {0, float}
        Spin component along y-axis of lessmassive black hole.
    s1z: {0, float}
        Spin component along z-axis of less massive black hole. This
        is also the axis along which the orbital angular momentum points.
    tc: {0, float, lal.LIGOTimeGPS}
        Time of coalescence of the binary.
    phic: {0, float}
        Orbital phase at coalescence of the binary.
    inclination: {0, float}
        Inclination angle of binary with respect to line of sight joining it
        with the GW detector
    ra: {0, float}
        Right ascension of the binary in sky (longitude).
    dec: {0, float}
        Declination of the binary in sky (latitude).
    polarization: {0, float}
        Polarization angle between the binary's source frame, and the radiation
        frame that has its z-axis along the line of sight to the source from
        Earth-based detector.

    signal_approx: {"", string}
        Waveform approximant to model the simulated signal.
    signal_file: {None, string}
        Path to file containing simulated signal written in ASCII.
    Note: One and only out of signal_approx, signal_file must be specified

    tmplt_approx: string
        Waveform approximant to model the simulated signal.
    tmplt_file: {None, string}
        Path to file containing simulated signal written in ASCII.
    Note: One and only out of tmplt_approx, tmplt_file must be specified

    aligned_spin_tmplt_only: {True, bool}
        Restrict to aligned-spin templates. Set x-y components of black hole
        spins to zero.
    non_spin_tmplt_only: {False, bool}
        Restrict to non-spinning templates. Set spins of black holes to zero.

    f_lower: {15.0, float}
        Lower frequency cutoff to integrate overlaps.
    sample_rate: {4096, int}
        Sample rate at which to sample all time series.
    signal_duration: {32, int}
        Maximum allowed length (seconds) of signal and templates.
    psd_string: {"aLIGOZeroDetHighPower", string}
        PSD name, as cataloged in `pycbc.psd`

    verbose: {True, bool}
        Enable verbose logging.
    debug: {False, bool}
        Enable debugging level logging.
    

    Returns
    -------
    overlap: float
        Match between signal and tmplts with identical parameters.

    """
    # 0) OPTION CHECKING
    if aligned_spin_tmplt_only:
        print(
            "WARNING: Spin components parallel to L allowed, others set to 0 in templates."
        )

    # 1) GENERATE FILTERING META-PARAMETERS
    filter_N = signal_duration * sample_rate
    filter_n = filter_N // 2 + 1
    delta_t = 1.0 / sample_rate
    delta_f = 1.0 / signal_duration
    # LIGO Noise PSD
    psd = from_string(psd_string, filter_n, delta_f, f_lower)

    # 2) GENERATE THE TARGET SIGNAL
    # Get the signal waveform first
    if signal_h is None:
        if (
            signal_approx in pywf.fd_approximants()
            or signal_approx in pywf.td_approximants()
        ):
            signal_generator = pywfg.FDomainDetFrameGenerator(
                pywfg.select_waveform_generator(signal_approx),
                0,
                variable_args=[
                    "mass1",
                    "mass2",
                    "spin1x",
                    "spin1y",
                    "spin1z",
                    "spin2x",
                    "spin2y",
                    "spin2z",
                    "coa_phase",
                    "tc",
                    "inclination",
                    "ra",
                    "dec",
                    "polarization",
                ],
                detectors=["H1"],
                delta_t=delta_t,
                delta_f=delta_f,
                f_lower=f_lower,
                approximant=signal_approx,
            )
        elif "FromDataFile" in signal_approx:
            if os.path.getsize(signal_file) == 0:
                raise RuntimeError(
                    " ERROR:...OOPS. Waveform file %s empty!!" % signal_file
                )
            try:
                _ = np.loadtxt(signal_file)
            except BaseException:
                raise RuntimeError(
                    " WARNING: FAILURE READING DATA FROM %s.." % signal_file
                )

            waveform_params = lsctables.SimInspiral()
            waveform_params.latitude = 0
            waveform_params.longitude = 0
            waveform_params.polarization = 0
            waveform_params.spin1x = 0
            waveform_params.spin1y = 0
            waveform_params.spin1z = 0
            waveform_params.spin2x = 0
            waveform_params.spin2y = 0
            waveform_params.spin2z = 0
            # try:
            if True:
                if verbose:
                    print(".. generating signal waveform ")
                signal_htilde, _params = get_waveform(
                    signal_approx,
                    -1,
                    -1,
                    -1,
                    waveform_params,
                    f_lower,
                    sample_rate,
                    filter_N,
                    datafile=signal_file,
                )
                print(".. generated signal waveform ")
                m1, m2, w_value, _ = _params
                waveform_params.mass1 = m1
                waveform_params.mass2 = m2
                signal_h = signal_htilde
            # except: raise IOError("Approximant %s not found.." % signal_approx)
        else:
            raise IOError("Signal Approximant %s not found.." % signal_approx)

        if verbose:
            print(
                "..Generating signal with masses = %3f, %.3f, spin1 = (%.3f, %.3f, %.3f), and  spin2 = (%.3f, %.3f, %.3f)"
                % (m1, m2, s1x, s1y, s1z, s2x, s2y, s2z)
            )
            sys.stdout.flush()

    if signal_h is None:
        if (
            signal_approx in pywf.fd_approximants()
            or signal_approx in pywf.td_approximants()
        ):
            signal = signal_generator.generate(
                mass1=m1,
                mass2=m2,
                spin1x=s1x,
                spin1y=s1y,
                spin1z=s1z,
                spin2x=s2x,
                spin2y=s2y,
                spin2z=s2z,
                coa_phase=phic,
                tc=tc,
                inclination=inclination,
                ra=ra,
                dec=dec,
                polarization=polarization,
            )
            # NOTE: SEOBNRv4 has extra high frequency content, it seems..
            signal_h = signal["H1"]

    if type(signal_h) == pycbc.types.TimeSeries:
        signal_h = extend_waveform_TimeSeries(signal_h, filter_N)
    signal_h = make_frequency_series(signal_h)
    signal_h = extend_waveform_FrequencySeries(signal_h, filter_n, force_fit=True)

    # 3) GENERATE THE TARGET TEMPLATE
    # Get the signal waveform first
    if tmplt_h is None:
        if (
            tmplt_approx in pywf.fd_approximants()
            or tmplt_approx in pywf.td_approximants()
        ):
            tmplt_generator = pywfg.FDomainDetFrameGenerator(
                pywfg.select_waveform_generator(tmplt_approx),
                0,
                variable_args=[
                    "mass1",
                    "mass2",
                    "spin1x",
                    "spin1y",
                    "spin1z",
                    "spin2x",
                    "spin2y",
                    "spin2z",
                    "coa_phase",
                    "tc",
                    "inclination",
                    "ra",
                    "dec",
                    "polarization",
                ],
                detectors=["H1"],
                delta_f=delta_f,
                delta_t=delta_t,
                f_lower=f_lower,
                approximant=tmplt_approx,
            )
        elif "FromDataFile" in tmplt_approx:
            if os.path.getsize(tmplt_file) == 0:
                raise RuntimeError(
                    " ERROR:...OOPS. Waveform file %s empty!!" % tmplt_file
                )
            try:
                _ = np.loadtxt(tmplt_file)
            except BaseException:
                raise RuntimeError(
                    " WARNING: FAILURE READING DATA FROM %s.." % tmplt_file
                )

            waveform_params = lsctables.SimInspiral()
            waveform_params.latitude = 0
            waveform_params.longitude = 0
            waveform_params.polarization = 0
            waveform_params.spin1x = 0
            waveform_params.spin1y = 0
            waveform_params.spin1z = 0
            waveform_params.spin2x = 0
            waveform_params.spin2y = 0
            waveform_params.spin2z = 0
            # try:
            if True:
                if verbose:
                    print(".. generating signal waveform ")
                tmplt_htilde, _params = get_waveform(
                    tmplt_approx,
                    -1,
                    -1,
                    -1,
                    waveform_params,
                    f_lower,
                    1.0 / delta_t,
                    filter_N,
                    datafile=tmplt_file,
                )
                print(".. generated signal waveform ")
                m1, m2, w_value, _ = _params
                waveform_params.mass1 = m1
                waveform_params.mass2 = m2
                tmplt_h = make_frequency_series(tmplt_htilde)
                tmplt_h = extend_waveform_FrequencySeries(tmplt_h, filter_n)
            # except: raise IOError("Approximant %s not found.." % tmplt_approx)
        else:
            raise IOError("Template Approximant %s not found.." % tmplt_approx)

    if aligned_spin_tmplt_only:
        _m1, _m2, _s1x, _s1y, _s1z, _s2x, _s2y, _s2z = m1, m2, 0, 0, s1z, 0, 0, s2z
    elif non_spin_tmplt_only:
        _m1, _m2, _s1x, _s1y, _s1z, _s2x, _s2y, _s2z = m1, m2, 0, 0, 0, 0, 0, 0
    else:
        _m1, _m2, _s1x, _s1y, _s1z, _s2x, _s2y, _s2z = (
            m1,
            m2,
            s1x,
            s1y,
            s1z,
            s2x,
            s2y,
            s2z,
        )

    if verbose:
        print(
            "..Generating template with masses = %3f, %.3f, spin1 = (%.3f, %.3f, %.3f), and  spin2 = (%.3f, %.3f, %.3f)"
            % (_m1, _m2, _s1x, _s1y, _s1z, _s2x, _s2y, _s2z)
        )
        sys.stdout.flush()

    if tmplt_h is None:
        if (
            tmplt_approx in pywf.fd_approximants()
            or tmplt_approx in pywf.td_approximants()
        ):
            try:
                template = tmplt_generator.generate(
                    mass1=_m1,
                    mass2=_m2,
                    spin1x=_s1x,
                    spin1y=_s1y,
                    spin1z=_s1z,
                    spin2x=_s2x,
                    spin2y=_s2y,
                    spin2z=_s2z,
                    coa_phase=phic,
                    tc=tc,
                    inclination=inclination,
                    ra=ra,
                    dec=dec,
                    polarization=polarization,
                )
            except RuntimeError as rerr:
                print(
                    """FAILED TO GENERATE %s waveform for
                masses = %.3f, %.3f
                spins = (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f)from gwnr.analysis.filter import calculate_faithfulness

                phic, tc, ra, dec, pol = (%.3f, %.3f, %.3f, %.3f, %.3f)"""
                    % (
                        tmplt_approx,
                        _m1,
                        _m2,
                        _s1x,
                        _s1y,
                        _s1z,
                        _s2x,
                        _s2y,
                        _s2z,
                        phic,
                        tc,
                        ra,
                        dec,
                        polarization,
                    )
                )
                raise RuntimeError(rerr)
            tmplt_h = template["H1"]

    if type(tmplt_h) == pycbc.types.TimeSeries:
        tmplt_h = extend_waveform_TimeSeries(tmplt_h, filter_N)
    tmplt_h = make_frequency_series(tmplt_h)
    tmplt_h = extend_waveform_FrequencySeries(tmplt_h, filter_n, force_fit=True)

    # 4) COMPUTE MATCH
    m, idx = match(signal_h, tmplt_h, psd=psd, low_frequency_cutoff=f_lower)

    if debug:
        print(
            "MATCH IS %.6f for parameters" % m,
            m1,
            m2,
            _s1x,
            _s1y,
            _s1z,
            _s2x,
            _s2y,
            _s2z,
            inclination,
        )
        sys.stderr.flush()
    #
    # 5) RETURN OPTIMIZED MATCH
    return m, idx


def _constraint_function_fitting_factor_(x, *args):
    """
    This function implements constraints on the optimization of fitting
    factors:
    1) spin magnitudes on both holes should be <= 1

    """
    (
        signal_h,
        tmplt_generator,
        vary_masses_only,
        vary_masses_and_aligned_spin_only,
        [s_min, s_max],
        [s_eff_min, s_eff_max],
        [s1x, s1y, s1z],
        [s2x, s2y, s2z],
        [inclination],
        [psd, f_lower, filter_n],
        [verbose, debug],
    ) = args

    if len(x) == 2:
        m1, m2 = x
        if vary_masses_only:
            _s1x = _s1y = _s1z = _s2x = _s2y = _s2z = 0
        else:
            _s1x, _s1y, _s1z = s1x, s1y, s1z
            _s2x, _s2y, _s2z = s2x, s2y, s2z
    elif len(x) == 4:
        m1, m2, _s1z, _s2z = x
        if vary_masses_and_aligned_spin_only:
            _s1x = _s1y = _s2x = _s2y = 0
        else:
            _s1x, _s1y = s1x, s1y
            _s2x, _s2y = s2x, s2y
    elif len(x) == 9:
        m1, m2, _s1x, _s1y, _s1z, _s2x, _s2y, _s2z, _inclination = x
    else:
        raise IOError("No of vars %d not supported (should be 2 or 4 or 9)" % len(x))
    # Constraint on spin magnitudes
    s1_mag = (_s1x ** 2 + _s1y ** 2 + _s1z ** 2) ** 0.5
    s2_mag = (_s2x ** 2 + _s2y ** 2 + _s2z ** 2) ** 0.5
    if (s1_mag > s_max) or (s2_mag > s_max):
        return -1

    # Constraint on effective spin
    s_eff = (_s1z * m1 + _s2z * m2) / (m1 + m2)
    if (s_eff > s_eff_max) or (s_eff < s_eff_min):
        return -1

    # Default
    return 1


def _objective_function_fitting_factor_(x, *args):
    """
    This function is to be minimized if the fitting factor is to be found

    Inputs
    ------
    x: list
        Such that: m1, m2, _s1x, _s1y, _s1z, _s2x, _s2y, _s2z = x
    args: tuple
        Such that: signal_h, tmplt_generator = args
    """
    _objective_function_fitting_factor_.counter += 1
    (
        signal_h,
        tmplt_generator,
        vary_masses_only,
        vary_masses_and_aligned_spin_only,
        [s_min, s_max],
        [s_eff_min, s_eff_max],
        [s1x, s1y, s1z],
        [s2x, s2y, s2z],
        [inclination],
        [psd, f_lower, filter_n],
        [verbose, debug],
    ) = args

    # 1) OBTAIN THE TEMPLATE PARAMETERS FROM X. ASSUME THAT ONLY
    # THOSE ARE PASSED THAT ARE NEEDED BY THE GENERATOR
    if len(x) == 2:
        m1, m2 = x
        if vary_masses_only:
            _s1x = _s1y = _s1z = _s2x = _s2y = _s2z = 0
        else:
            _s1x, _s1y, _s1z = s1x, s1y, s1z
            _s2x, _s2y, _s2z = s2x, s2y, s2z
        _inclination = inclination
    elif len(x) == 4:
        m1, m2, _s1z, _s2z = x
        if vary_masses_and_aligned_spin_only:
            _s1x = _s1y = _s2x = _s2y = 0
        else:
            _s1x, _s1y = s1x, s1y
            _s2x, _s2y = s2x, s2y
        _inclination = inclination
    elif len(x) == 9:
        m1, m2, _s1x, _s1y, _s1z, _s2x, _s2y, _s2z, _inclination = x
    else:
        raise IOError("No of vars %d not supported (should be 2 or 4 or 9)" % len(x))

    # 2) ASSUME THAT
    tmplt = tmplt_generator.generate(
        mass1=m1,
        mass2=m2,
        spin1x=_s1x,
        spin1y=_s1y,
        spin1z=_s1z,
        spin2x=_s2x,
        spin2y=_s2y,
        spin2z=_s2z,
        inclination=_inclination,
    )
    if debug:
        print(
            "IN FF Objective-> for parameters:",
            m1,
            m2,
            _s1x,
            _s1y,
            _s1z,
            _s2x,
            _s2y,
            _s2z,
            _inclination,
        )
        print(
            "IN FF Objective-> Length(tmplt) = {}, making it {}".format(
                len(tmplt["H1"]), filter_n
            )
        )
    tmplt_h = make_frequency_series(tmplt["H1"])
    tmplt_h = extend_waveform_FrequencySeries(tmplt_h, filter_n, force_fit=True)

    # 3) COMPUTE MATCH
    m, _ = match(signal_h, tmplt_h, psd=psd, low_frequency_cutoff=f_lower)

    if debug:
        print(
            "MATCH IS %e for parameters:" % m,
            m1,
            m2,
            _s1x,
            _s1y,
            _s1z,
            _s2x,
            _s2y,
            _s2z,
            _inclination,
        )

    retval = np.log10(1.0 - m)

    # We do not want PSO to go berserk, so we stop when FF = 0.999999
    if retval <= -6.0:
        retval = -6.0
    return retval


def calculate_fitting_factor(
    m1,
    m2,
    tmplt_approx,
    s1x=0,
    s1y=0,
    s1z=0,
    s2x=0,
    s2y=0,
    s2z=0,
    tc=0,
    phic=0,
    inclination=0,
    ra=0,
    dec=0,
    polarization=0,
    signal_approx="",
    signal_file="",
    signal_h=None,
    vary_masses_only=True,
    vary_masses_and_aligned_spin_only=False,
    chirp_mass_window=0.2,
    effective_spin_window=0.75,
    f_lower=15.0,
    sample_rate=4096,
    signal_duration=16,
    psd_string="aLIGOZeroDetHighPower",
    ff_max=0.99999,
    pso_swarm_size=100,
    pso_omega=0.5,
    pso_phip=0.5,
    pso_phig=0.25,
    pso_minfunc=1e-3,
    pso_n_processes=1,
    num_retries=5,
    verbose=True,
    debug=False,
):
    """Calculates the fitting factor for a signal, against templates of
    another approximant.

    This function uses a particle swarm optimization to maximize
    the overlaps between signal and templates. Algorithm parameters
    are tunable, depending on how many dimensions we are optimizing
    over.

    Inputs
    ------

    m1: float
        Mass of more massive black hole in binary .
    m2: float
        Mass of less massive black hole in binary.
    tmplt_approx: string
        Waveform approximant to model the simulated signal.
    s1x: {0, float}
        Spin component along x-axis of more massive black hole.
    s1y: {0, float}
        Spin component along y-axis of more massive black hole.
    s1z: {0, float}
        Spin component along z-axis of more massive black hole. This
        is also the axis along which the orbital angular momentum points.
    s1x: {0, float}
        Spin component along x-axis of less massive black hole.
    s1y: {0, float}
        Spin component along y-axis of lessmassive black hole.
    s1z: {0, float}
        Spin component along z-axis of less massive black hole. This
        is also the axis along which the orbital angular momentum points.
    tc: {0, float, lal.LIGOTimeGPS}
        Time of coalescence of the binary.
    phic: {0, float}
        Orbital phase at coalescence of the binary.
    inclination: {0, float}
        Inclination angle of binary with respect to line of sight joining it
        with the GW detector
    ra: {0, float}
        Right ascension of the binary in sky (longitude).
    dec: {0, float}
        Declination of the binary in sky (latitude).
    polarization: {0, float}
        Polarization angle between the binary's source frame, and the radiation
        frame that has its z-axis along the line of sight to the source from
        Earth-based detector.

    signal_approx: {"", string}
        Waveform approximant to model the simulated signal.
    signal_file: {"", string}
        Path to file containing simulated signal written in ASCII.
    signal_h: {None, pycbc.types.FrequencySeries`}
        Simulated signal in frequency domain
    Note: One and only out of signal_approx, signal_file, signal_h must be specified
    
    vary_masses_only: {True, bool}
        Enable variation of masses only in templates. All other parameters are
        fixed to their signal values.
    vary_masses_and_aligned_spin_only: {False, bool}
        Enable variation of masses and z-component of black hole spins only in
        templates. All other parameters are fixed to their signal values.
    chirp_mass_window: {0.1, float}
        (fractional) Window around true value of chirp mass within which to
        optimize fitting factor
    effective_spin_window: {0.5, float}
        Window around true value of effective spin within which to optimize
        fitting factor

    f_lower: {15.0, float}
        Lower frequency cutoff to integrate overlaps.
    sample_rate: {4096, int}
        Sample rate at which to sample all time series.
    signal_duration: {32, int}
        Maximum allowed length (seconds) of signal and templates.
    psd_string: {"aLIGOZeroDetHighPower", string}
        PSD name, as cataloged in `pycbc.psd`

    pso_swarm_size: {500, int}
        Size of particle swarm to use when invoking PSO.
    pso_omega: {0.5, float}
        Particle velocity scaling factor.
    pso_phip: {0.5, float}
        Scaling factor to search away from the particle’s best known position.
    pso_phig: {0.25, float}
        Scaling factor to search away from the particle’s best known position.
    pso_minfunc: {1e-3, float}
        The minimum change of swarm’s best objective value before the search terminates.
    pso_n_processes: {1, int}
        Number of CPU cores to engage for calculating objective functions by PSO algorithm.
        NOTE: this option is under construction.
    num_retries: {4, int}
        Number of times we retune configurations of PSO before declaring a
        globally optimized fitting factor.

    verbose: {True, bool}
        Enable verbose logging.
    debug: {False, bool}
        Enable debugging level logging.
    

    Returns
    -------
    params: list
        List of parameters that were varied and correspond to the fitting factor
    overlap: float
        Match between signal and tmplts with identical parameters. This is to
        be used as a reference to compare the fitting factor against.
    fitting_factor: float
        Value of fitting factor

    """
    # 0) Verify input
    if (
        tmplt_approx not in pywf.td_approximants()
        and tmplt_approx not in pywf.fd_approximants()
    ):
        raise RuntimeError(
            "We do not recognize template approximant: {}.".format(tmplt_approx)
        )
    if (
        (
            signal_approx not in pywf.td_approximants()
            and signal_approx not in pywf.fd_approximants()
        )
        and not os.path.exists(signal_file)
        and signal_h is None
    ):
        raise RuntimeError(
            "Please provide either a signal approximant, a waveform, or a data file storing waveform"
        )
    if vary_masses_only:
        print(
            "WARNING: Only component masses are allowed to be varied in templates."
            " Setting the rest to signal values."
        )
    if vary_masses_and_aligned_spin_only:
        print(
            "WARNING: Only component masses and spin components parallel to L "
            "allowed to be varied in templates. Setting the rest to signal values."
        )
        # Explicitly supercede the `vary_masses_only` option flag.
        print(
            "WARNING: Inconsistent options: vary_masses_only and vary_masses_and_aligned_spin_only both specified. Choosing the more liberal option."
        )
        vary_masses_only = False
    if (not vary_masses_only) and (not vary_masses_and_aligned_spin_only):
        print(
            "WARNING: All mass and spin components, and source inclination being"
            " varied in templates. Sky parameters still fixed to signal values."
        )

    # 1) Filtering parameters
    signal_duration = int(signal_duration)
    sample_rate = int(sample_rate)
    filter_N = signal_duration * sample_rate
    filter_n = filter_N // 2 + 1
    delta_t = 1.0 / sample_rate
    delta_f = 1.0 / signal_duration
    psd = from_string(psd_string, filter_n, delta_f, f_lower)
    if verbose:
        print(
            "signal_duration = %d, sample_rate = %d, filter_N = %d, filter_n = %d"
            % (signal_duration, sample_rate, filter_N, filter_n)
        )
        print("deltaT = %f, deltaF = %f" % (delta_t, delta_f))

    # 2) Generate signal waveform
    if signal_h is None and "FromDataFile" in signal_approx:
        if os.path.getsize(signal_file) == 0:
            raise RuntimeError(
                " ERROR:...OOPS. Waveform file %s is empty!!" % signal_file
            )
        try:
            _ = np.loadtxt(signal_file)
        except BaseException:
            raise RuntimeError(" WARNING: FAILURE READING DATA FROM %s.." % signal_file)

        waveform_params = lsctables.SimInspiral()
        waveform_params.latitude = (
            waveform_params.longitude
        ) = waveform_params.polarization = 0
        waveform_params.spin1x = waveform_params.spin1y = waveform_params.spin1z = 0
        waveform_params.spin2x = waveform_params.spin2y = waveform_params.spin2z = 0
        # try:
        if True:
            if verbose:
                print(".. generating signal waveform ")
            signal_htilde, _params = get_waveform(
                signal_approx,
                -1,
                -1,
                -1,
                waveform_params,
                f_lower,
                sample_rate,
                filter_N,
                datafile=signal_file,
            )
            print(".. generated signal waveform ")
            m1, m2, w_value, _ = _params
            signal_h = make_frequency_series(signal_htilde)
            signal_h = extend_waveform_FrequencySeries(
                signal_h, filter_n, force_fit=True
            )
        # except: raise IOError("Approximant %s not found.." % signal_approx)

    if signal_h is None:
        if (
            signal_approx not in pywf.fd_approximants()
            and signal_approx not in pywf.td_approximants()
        ):
            raise IOError("Approximant %s not found.." % signal_approx)
        signal_generator = pywfg.FDomainDetFrameGenerator(
            pywfg.select_waveform_generator(signal_approx),
            0,
            variable_args=[
                "mass1",
                "mass2",
                "spin1x",
                "spin1y",
                "spin1z",
                "spin2x",
                "spin2y",
                "spin2z",
                "coa_phase",
                "tc",
                "inclination",
                "ra",
                "dec",
                "polarization",
            ],
            detectors=["H1"],
            delta_f=delta_f,
            delta_t=delta_t,
            f_lower=f_lower,
            approximant=signal_approx,
        )
        if verbose:
            print(
                "\nGenerating signal with masses = {:3f}, {:.3f},"
                " spin1 = ({:.3f}, {:.3f}, {:.3f}), and "
                " spin2 = ({:.3f}, {:.3f}, {:.3f})".format(
                    m1, m2, s1x, s1y, s1z, s2x, s2y, s2z
                )
            )
            sys.stdout.flush()

        signal = signal_generator.generate(
            mass1=m1,
            mass2=m2,
            spin1x=s1x,
            spin1y=s1y,
            spin1z=s1z,
            spin2x=s2x,
            spin2y=s2y,
            spin2z=s2z,
            coa_phase=phic,
            tc=tc,
            inclination=inclination,
            ra=ra,
            dec=dec,
            polarization=polarization,
        )
        signal_h = signal["H1"]

    if type(signal_h) == pycbc.types.TimeSeries:
        signal_h = extend_waveform_TimeSeries(signal_h, filter_N)
    signal_h = make_frequency_series(signal_h)
    signal_h = extend_waveform_FrequencySeries(signal_h, filter_n, force_fit=True)

    # 3) INITIALIZE THE WAVEFORM GENERATOR FOR TEMPLATES
    # We allow all intrinsic parameters to vary, and fix them to the signal
    # values, in case only masses or only mass+aligned-spin components are
    # requested to be varied. This fixing is done inside the objective
    # function.
    tmplt_generator = pywfg.FDomainDetFrameGenerator(
        pywfg.select_waveform_generator(tmplt_approx),
        0,
        variable_args=[
            "mass1",
            "mass2",
            "spin1x",
            "spin1y",
            "spin1z",
            "spin2x",
            "spin2y",
            "spin2z",
            "inclination",
        ],
        detectors=["H1"],
        coa_phase=phic,
        tc=tc,
        ra=ra,
        dec=dec,
        polarization=polarization,
        delta_f=delta_f,
        delta_t=delta_t,
        f_lower=f_lower,
        approximant=tmplt_approx,
    )

    # 6b) NOW SET THE RANGE OF PARAMETERS TO BE PROBED
    mt = m1 + m2 * 1.0
    et = m1 * m2 / mt / mt
    mc = mt * et ** 0.6
    mc_min = mc * (1.0 - chirp_mass_window)
    mc_max = mc * (1.0 + chirp_mass_window)
    et_max = 0.25
    et_min = 10.0 / 121.0  # Lets say we trust waveform models up to q = 10
    m1_max, _ = pnutils.mchirp_eta_to_mass1_mass2(mc_max, et_min)
    m1_min, _ = pnutils.mchirp_eta_to_mass1_mass2(mc_min, et_max)
    _, m2_max = pnutils.mchirp_eta_to_mass1_mass2(mc_max, et_max)
    _, m2_min = pnutils.mchirp_eta_to_mass1_mass2(mc_min, et_min)
    s_min = -0.99
    s_max = +0.99
    s_eff = (s1z * m1 + s2z * m2) / (m1 + m2)
    s_eff_min = s_eff - effective_spin_window
    s_eff_max = s_eff + effective_spin_window
    incl_min = 0
    incl_max = np.pi

    if verbose:
        print(
            "SEARCH LIMITS-> ",
            m1,
            m2,
            mt,
            et,
            mc,
            mc_min,
            mc_max,
            et_min,
            et_max,
            m1_min,
            m1_max,
            m2_min,
            m2_max,
        )

    if vary_masses_and_aligned_spin_only:
        low_lim = [m1_min, m2_min, s_min, s_min]
        high_lim = [m1_max, m2_max, s_max, s_max]
    elif vary_masses_only:
        low_lim = [m1_min, m2_min]
        high_lim = [m1_max, m2_max]
    else:
        low_lim = [m1_min, m2_min, s_min, s_min, s_min, s_min, s_min, s_min, incl_min]
        high_lim = [m1_max, m2_max, s_max, s_max, s_max, s_max, s_max, s_max, incl_max]

    if verbose:
        print("\nSearching within limits:\n", low_lim, " and \n", high_lim)
        print("\nCalculating overlap now..")
        sys.stdout.flush()
    olap, idx = calculate_faithfulness(
        m1,
        m2,
        s1x,
        s1y,
        s1z,
        s2x,
        s2y,
        s2z,
        tc=tc,
        phic=phic,
        inclination=inclination,
        ra=ra,
        dec=dec,
        polarization=polarization,
        signal_approx=signal_approx,
        signal_file=signal_file,
        signal_h=signal_h,
        tmplt_approx=tmplt_approx,
        tmplt_file="",
        aligned_spin_tmplt_only=vary_masses_and_aligned_spin_only,
        non_spin_tmplt_only=vary_masses_only,
        f_lower=f_lower,
        sample_rate=sample_rate,
        signal_duration=signal_duration,
        verbose=verbose,
        debug=debug,
    )

    # IF overlap is already high enough, skip FF computation.
    if olap >= ff_max:
        if verbose:
            print(
                "Skipping FF computation as olap is high enough at: {:.6f}".format(olap)
            )
        if vary_masses_and_aligned_spin_only:
            return [np.array([m1, m2, s1z, s2z]), olap, olap]
        if vary_masses_only:
            return [np.array([m1, m2]), olap, olap]
        return [
            np.array([m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, inclination]),
            olap,
            olap,
        ]
    if verbose:
        print(
            "Overlap with aligned_spin_tmplt_only = ",
            vary_masses_and_aligned_spin_only,
            " and non_spin_tmplt_only = ",
            vary_masses_only,
            ": ",
            olap,
            np.log10(1.0 - olap),
        )
        sys.stdout.flush()

    # 6a) FIRST CONSTRUCT THE FIXED ARGUMENTS FOR THE PSO's OBJECTIVE FUNCTION
    pso_args = (
        signal_h,
        tmplt_generator,
        vary_masses_only,
        vary_masses_and_aligned_spin_only,
        [s_min, s_max],
        [s_eff_min, s_eff_max],
        [s1x, s1y, s1z],
        [s2x, s2y, s2z],
        [inclination],
        [psd, f_lower, filter_n],
        [verbose, debug],
    )
    idx = 1
    ff = 0.0
    _objective_function_fitting_factor_.counter = 0

    # 6) Use PSO to compute fitting factor
    while ff <= olap and ff < ff_max:
        if idx and idx % 2 == 0:
            pso_minfunc *= 0.1
            pso_phig *= 1.1

        if idx > num_retries:
            print(
                "WARNING: Failed to improve on overlap in {} iterations. Set ff = olap now".format(
                    num_retries
                )
            )
            ff = olap
            break

        if verbose:
            print("\nTry %d to compute fitting factor" % idx)
            sys.stdout.flush()
        params, ff = pyswarm.pso(
            _objective_function_fitting_factor_,
            low_lim,
            high_lim,
            f_ieqcons=_constraint_function_fitting_factor_,
            args=pso_args,
            swarmsize=pso_swarm_size,
            omega=pso_omega,
            phip=pso_phip,
            phig=pso_phig,
            minfunc=pso_minfunc,
            maxiter=10,
            processes=pso_n_processes,
            debug=verbose,
        )
        # Restore fitting factor from 1-ff
        ff = 1.0 - 10 ** ff
        if verbose:
            print("\nLoop will continue till %.12f < %.12f" % (ff, olap))
            sys.stdout.flush()
        idx += 1

    if verbose:
        print(
            "optimization took %d objective func evals"
            % _objective_function_fitting_factor_.counter
        )
        sys.stdout.flush()
    #
    # 7) RETURN OPTIMIZED PARAMETERS
    return [params, olap, ff]


def overlap_between_waveforms(wav1, wav2, psd, f_lower=15.0):
    """
    Return overlap between two waveforms:

    TODO: Add resampling, padding capability.
    """
    len1, len2, lenp = len(wav1), len(wav2), len(psd)
    if len1 != len2:
        raise IOError("Length of waveforms not equal: %d,%d" % (len1, len2))
    if wav1.delta_t != wav2.delta_t:
        raise IOError("Mismatched wave sample rate")
    if len1 != 2 * lenp - 2:
        raise IOError("PSD length inconsistent with waveforms")
    #
    return match(wav1, wav2, psd=psd, low_frequency_cutoff=f_lower)[0]


def compute_snr_vs_time(wave, psd, time_step=1e-2, f_lower=15.0):
    # wave should be longer than dt
    assert time_step < len(wave) * wave.delta_t

    from numpy import round, arange
    from pycbc.filter import sigma
    from pycbc.types import TimeSeries

    integration_stop_times = arange(time_step, len(wave) * wave.delta_t, time_step)

    def truncate_wave_at_time(wave, end_time):
        wave_c = TimeSeries(wave, delta_t=wave.delta_t, copy=True)
        idx = int(round(float(end_time) / wave.delta_t))
        wave_c[idx:] = 0
        return wave_c

    wave_snr = []
    for integration_stop_time in integration_stop_times:
        wave_copy = truncate_wave_at_time(wave, integration_stop_time)
        wave_snr.append(sigma(wave_copy, psd, low_frequency_cutoff=f_lower))

    return TimeSeries(wave_snr, delta_t=time_step)
