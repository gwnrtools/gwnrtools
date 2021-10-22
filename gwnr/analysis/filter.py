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

from gwnr.utils.support import *
import os
import sys
import numpy as np

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


######################################################################
#      Overlap and Fitting Factor


def calculate_faithfulness(m1,
                           m2,
                           s1x=0,
                           s1y=0,
                           s1z=0,
                           s2x=0,
                           s2y=0,
                           s2z=0,
                           tc=0,
                           phic=0,
                           ra=0,
                           dec=0,
                           polarization=0,
                           signal_approx='IMRPhenomD',
                           signal_file=None,
                           tmplt_approx='IMRPhenomC',
                           tmplt_file=None,
                           aligned_spin_tmplt_only=True,
                           non_spin_tmplt_only=False,
                           f_lower=15.0,
                           sample_rate=4096,
                           signal_duration=256,
                           psd_string='aLIGOZeroDetHighPower',
                           verbose=True,
                           debug=False):
    """
Calculates the match for a signal of given physical
parameters, as modelled by a given signal approximant, against
templates of another approximant.

This function allows turning off x,y components of
spin for templates.

IN PROGRESS: Adding facility to use "FromDataFile" waveforms
    """
    # {{{
    # 0) OPTION CHECKING
    if aligned_spin_tmplt_only:
        print(
            "WARNING: Spin components parallel to L allowed, others set to 0 in templates."
        )

    # 1) GENERATE FILTERING META-PARAMETERS
    filter_N = signal_duration * sample_rate
    filter_n = filter_N / 2 + 1
    delta_t = 1. / sample_rate
    delta_f = 1. / signal_duration
    # LIGO Noise PSD
    psd = from_string(psd_string, filter_n, delta_f, f_lower)

    # 2) GENERATE THE TARGET SIGNAL
    # Get the signal waveform first
    if signal_approx in pywf.fd_approximants():
        generator = pywfg.FDomainDetFrameGenerator(
            pywfg.FDomainCBCGenerator,
            0,
            variable_args=[
                'mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z', 'spin2x',
                'spin2y', 'spin2z', 'coa_phase', 'tc', 'ra', 'dec',
                'polarization'
            ],
            detectors=['H1'],
            delta_f=delta_f,
            f_lower=f_lower,
            approximant=signal_approx)
    elif signal_approx in pywf.td_approximants():
        generator = pywfg.TDomainDetFrameGenerator(
            pywfg.TDomainCBCGenerator,
            0,
            variable_args=[
                'mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z', 'spin2x',
                'spin2y', 'spin2z', 'coa_phase', 'tc', 'ra', 'dec',
                'polarization'
            ],
            detectors=['H1'],
            delta_t=delta_t,
            f_lower=f_lower,
            approximant=signal_approx)
    elif 'FromDataFile' in signal_approx:
        if os.path.getsize(signal_file) == 0:
            raise RuntimeError(" ERROR:...OOPS. Waveform file %s empty!!" %
                               signal_file)
        try:
            _ = np.loadtxt(signal_file)
        except BaseException:
            raise RuntimeError(" WARNING: FAILURE READING DATA FROM %s.." %
                               signal_file)

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
            signal_htilde, _params = get_waveform(signal_approx,
                                                  -1,
                                                  -1,
                                                  -1,
                                                  waveform_params,
                                                  f_lower,
                                                  sample_rate,
                                                  filter_N,
                                                  datafile=signal_file)
            print(".. generated signal waveform ")
            m1, m2, w_value, _ = _params
            waveform_params.mass1 = m1
            waveform_params.mass2 = m2
            signal_h = make_frequency_series(signal_htilde)
            signal_h = extend_waveform_FrequencySeries(signal_h, filter_n)
        # except: raise IOError("Approximant %s not found.." % signal_approx)
    else:
        raise IOError("Signal Approximant %s not found.." % signal_approx)
    if verbose:
        print(
            "..Generating signal with masses = %3f, %.3f, spin1 = (%.3f, %.3f, %.3f), and  spin2 = (%.3f, %.3f, %.3f)"
            % (m1, m2, s1x, s1y, s1z, s2x, s2y, s2z))
        sys.stdout.flush()

    if signal_approx in pywf.fd_approximants():
        signal = generator.generate_from_args(m1, m2, s1x, s1y, s1z, s2x, s2y,
                                              s2z, phic, tc, ra, dec,
                                              polarization)
        # NOTE: SEOBNRv4 has extra high frequency content, it seems..
        if 'SEOBNRv4_ROM' in signal_approx or 'SEOBNRv2_ROM' in signal_approx:
            signal_h = extend_waveform_FrequencySeries(signal['H1'],
                                                       filter_n,
                                                       force_fit=True)
        else:
            signal_h = extend_waveform_FrequencySeries(signal['H1'], filter_n)
    elif signal_approx in pywf.td_approximants():
        signal = generator.generate_from_args(m1, m2, s1x, s1y, s1z, s2x, s2y,
                                              s2z, phic, tc, ra, dec,
                                              polarization)
        signal_h = make_frequency_series(signal['H1'])
        signal_h = extend_waveform_FrequencySeries(signal_h, filter_n)
    elif 'FromDataFile' in signal_approx:
        pass
    else:
        raise IOError("Signal Approximant %s not found.." % signal_approx)

    # 3) GENERATE THE TARGET TEMPLATE
    # Get the signal waveform first
    if tmplt_approx in pywf.fd_approximants():
        generator = pywfg.FDomainDetFrameGenerator(
            pywfg.FDomainCBCGenerator,
            0,
            variable_args=[
                'mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z', 'spin2x',
                'spin2y', 'spin2z', 'coa_phase', 'tc', 'ra', 'dec',
                'polarization'
            ],
            detectors=['H1'],
            delta_f=delta_f,
            f_lower=f_lower,
            approximant=tmplt_approx)
    elif tmplt_approx in pywf.td_approximants():
        generator = pywfg.TDomainDetFrameGenerator(
            pywfg.TDomainCBCGenerator,
            0,
            variable_args=[
                'mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z', 'spin2x',
                'spin2y', 'spin2z', 'coa_phase', 'tc', 'ra', 'dec',
                'polarization'
            ],
            detectors=['H1'],
            delta_t=delta_t,
            f_lower=f_lower,
            approximant=tmplt_approx)
    elif 'FromDataFile' in tmplt_approx:
        if os.path.getsize(tmplt_file) == 0:
            raise RuntimeError(" ERROR:...OOPS. Waveform file %s empty!!" %
                               tmplt_file)
        try:
            _ = np.loadtxt(tmplt_file)
        except BaseException:
            raise RuntimeError(" WARNING: FAILURE READING DATA FROM %s.." %
                               tmplt_file)

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
            tmplt_htilde, _params = get_waveform(tmplt_approx,
                                                 -1,
                                                 -1,
                                                 -1,
                                                 waveform_params,
                                                 f_lower,
                                                 1. / delta_t,
                                                 filter_N,
                                                 datafile=tmplt_file)
            print(".. generated signal waveform ")
            m1, m2, w_value, _ = _params
            waveform_params.mass1 = m1
            waveform_params.mass2 = m2
            tmplt_h = make_frequency_series(tmplt_htilde)
            tmplt_h = extend_waveform_FrequencySeries(tmplt_h, filter_n)
        # except: raise IOError("Approximant %s not found.." % tmplt_approx)
    else:
        raise IOError("Template Approximant %s not found.." % tmplt_approx)
    #
    if aligned_spin_tmplt_only:
        _m1, _m2, _s1x, _s1y, _s1z, _s2x, _s2y, _s2z = m1, m2, 0, 0, s1z, 0, 0, s2z
    elif non_spin_tmplt_only:
        _m1, _m2, _s1x, _s1y, _s1z, _s2x, _s2y, _s2z = m1, m2, 0, 0, 0, 0, 0, 0
    else:
        _m1, _m2, _s1x, _s1y, _s1z, _s2x, _s2y, _s2z = m1, m2, s1x, s1y, s1z, s2x, s2y, s2z
    #
    # template = generator.generate_from_args(_m1, _m2, _s1x, _s1y, _s1z, _s2x, _s2y, _s2z,\
    #                              phic, tc, ra, dec, polarization)
    #
    if verbose:
        print(
            "..Generating template with masses = %3f, %.3f, spin1 = (%.3f, %.3f, %.3f), and  spin2 = (%.3f, %.3f, %.3f)"
            % (_m1, _m2, _s1x, _s1y, _s1z, _s2x, _s2y, _s2z))
        sys.stdout.flush()

    if tmplt_approx in pywf.fd_approximants():
        try:
            template = generator.generate_from_args(_m1, _m2, _s1x, _s1y, _s1z,
                                                    _s2x, _s2y, _s2z, phic, tc,
                                                    ra, dec, polarization)
        except RuntimeError as rerr:
            print("""FAILED TO GENERATE %s waveform for
              masses = %.3f, %.3f
              spins = (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f)
              phic, tc, ra, dec, pol = (%.3f, %.3f, %.3f, %.3f, %.3f)""" %
                  (tmplt_approx, _m1, _m2, _s1x, _s1y, _s1z, _s2x, _s2y, _s2z,
                   phic, tc, ra, dec, polarization))
            raise RuntimeError(rerr)
        # NOTE: SEOBNRv4 has extra high frequency content, it seems..
        if 'SEOBNRv4_ROM' in tmplt_approx or 'SEOBNRv2_ROM' in tmplt_approx:
            template_h = extend_waveform_FrequencySeries(template['H1'],
                                                         filter_n,
                                                         force_fit=True)
        else:
            template_h = extend_waveform_FrequencySeries(
                template['H1'], filter_n)
    elif tmplt_approx in pywf.td_approximants():
        try:
            template = generator.generate_from_args(_m1, _m2, _s1x, _s1y, _s1z,
                                                    _s2x, _s2y, _s2z, phic, tc,
                                                    ra, dec, polarization)
        except RuntimeError as rerr:
            print("""FAILED TO GENERATE %s waveform for
              masses = %.3f, %.3f
              spins = (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f)
              phic, tc, ra, dec, pol = (%.3f, %.3f, %.3f, %.3f, %.3f)""" %
                  (tmplt_approx, _m1, _m2, _s1x, _s1y, _s1z, _s2x, _s2y, _s2z,
                   phic, tc, ra, dec, polarization))
            raise RuntimeError(rerr)
        template_h = make_frequency_series(template['H1'])
        template_h = extend_waveform_FrequencySeries(template_h, filter_n)
    elif 'FromDataFile' in tmplt_approx:
        pass
    else:
        raise IOError("Template Approximant %s not found.." % tmplt_approx)

    # 4) COMPUTE MATCH
    m, idx = match(signal_h, template_h, psd=psd, low_frequency_cutoff=f_lower)

    if debug:
        print("MATCH IS %.6f for parameters" % m, m1, m2, _s1x, _s1y, _s1z,
              _s2x, _s2y, _s2z)
        sys.stderr.flush()
    #
    # 5) RETURN OPTIMIZED MATCH
    return m, idx
    # }}}


#############################
def calculate_fitting_factor(m1,
                             m2,
                             s1x=0,
                             s1y=0,
                             s1z=0,
                             s2x=0,
                             s2y=0,
                             s2z=0,
                             tc=0,
                             phic=0,
                             ra=0,
                             dec=0,
                             polarization=0,
                             signal_approx='IMRPhenomD',
                             signal_file=None,
                             tmplt_approx='IMRPhenomC',
                             vary_masses_only=True,
                             vary_masses_and_aligned_spin_only=False,
                             chirp_mass_window=0.1,
                             effective_spin_window=0.5,
                             num_retries=4,
                             f_lower=15.0,
                             sample_rate=4096,
                             signal_duration=256,
                             psd_string='aLIGOZeroDetHighPower',
                             pso_swarm_size=500,
                             pso_omega=0.5,
                             pso_phip=0.5,
                             pso_phig=0.25,
                             pso_minfunc=1e-8,
                             verbose=True,
                             debug=False):
    """
Calculates the fitting factor for a signal of given physical
parameters, as modelled by a given signal approximant, against
templates of another approximant.

This function uses a particle swarm optimization to maximize
the overlaps between signal and templates. Algorithm parameters
are tunable, depending on how many dimensions we are optimizing
over.

IN PROGRESS: Adding facility to use "FromDataFile" waveforms
    """
    # {{{
    # 0) OPTION CHECKING
    if vary_masses_only:
        print(
            "WARNING: Only component masses are allowed to be varied in templates. Setting the rest to signal values."
        )
    if vary_masses_and_aligned_spin_only:
        print(
            "WARNING: Only component masses and spin components parallel to L allowed to be varied in templates. Setting the rest to signal values."
        )
    if vary_masses_only and vary_masses_and_aligned_spin_only:
        raise IOError(
            "Inconsistent options: vary_masses_only and vary_masses_and_aligned_spin_only"
        )
    if (not vary_masses_only) and (not vary_masses_and_aligned_spin_only):
        print(
            "WARNING: All mass and spin components varied in templates. Sky parameters still fixed to signal values."
        )

    # 1) GENERATE FILTERING META-PARAMETERS
    signal_duration = int(signal_duration)
    sample_rate = int(sample_rate)
    filter_N = signal_duration * sample_rate
    filter_n = filter_N / 2 + 1
    delta_t = 1. / sample_rate
    delta_f = 1. / signal_duration
    if verbose:
        print(
            "signal_duration = %d, sample_rate = %d, filter_N = %d, filter_n = %d"
            % (signal_duration, sample_rate, filter_N, filter_n))
        print("deltaT = %f, deltaF = %f" % (delta_t, delta_f))
    # LIGO Noise PSD
    psd = from_string(psd_string, filter_n, delta_f, f_lower)

    # 2) GENERATE THE TARGET SIGNAL
    # PREPARATORY: Get the signal generator
    if signal_approx in pywf.fd_approximants():
        generator = pywfg.FDomainDetFrameGenerator(
            pywfg.FDomainCBCGenerator,
            0,
            variable_args=[
                'mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z', 'spin2x',
                'spin2y', 'spin2z', 'coa_phase', 'tc', 'ra', 'dec',
                'polarization'
            ],
            detectors=['H1'],
            delta_f=delta_f,
            f_lower=f_lower,
            approximant=signal_approx)
    elif signal_approx in pywf.td_approximants():
        generator = pywfg.TDomainDetFrameGenerator(
            pywfg.TDomainCBCGenerator,
            0,
            variable_args=[
                'mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z', 'spin2x',
                'spin2y', 'spin2z', 'coa_phase', 'tc', 'ra', 'dec',
                'polarization'
            ],
            detectors=['H1'],
            delta_t=delta_t,
            f_lower=f_lower,
            approximant=signal_approx)
    elif 'FromDataFile' in signal_approx:
        if os.path.getsize(signal_file) == 0:
            raise RuntimeError(" ERROR:...OOPS. Waveform file %s empty!!" %
                               signal_file)
        try:
            _ = np.loadtxt(signal_file)
        except BaseException:
            raise RuntimeError(" WARNING: FAILURE READING DATA FROM %s.." %
                               signal_file)

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
            signal_htilde, _params = get_waveform(signal_approx,
                                                  -1,
                                                  -1,
                                                  -1,
                                                  waveform_params,
                                                  f_lower,
                                                  sample_rate,
                                                  filter_N,
                                                  datafile=signal_file)
            print(".. generated signal waveform ")
            m1, m2, w_value, _ = _params
            waveform_params.mass1 = m1
            waveform_params.mass2 = m2
            signal_h = make_frequency_series(signal_htilde)
            signal_h = extend_waveform_FrequencySeries(signal_h, filter_n)
        # except: raise IOError("Approximant %s not found.." % signal_approx)
    else:
        raise IOError("Approximant %s not found.." % signal_approx)

    if verbose:
        print(
            "\nGenerating signal with masses = %3f, %.3f, spin1 = (%.3f, %.3f, %.3f), and  spin2 = (%.3f, %.3f, %.3f)"
            % (m1, m2, s1x, s1y, s1z, s2x, s2y, s2z))
        sys.stdout.flush()

    # Actually GENERATE THE SIGNAL
    if signal_approx in pywf.fd_approximants():
        signal = generator.generate_from_args(m1, m2, s1x, s1y, s1z, s2x, s2y,
                                              s2z, phic, tc, ra, dec,
                                              polarization)
        signal_h = extend_waveform_FrequencySeries(signal['H1'], filter_n)
    elif signal_approx in pywf.td_approximants():
        signal = generator.generate_from_args(m1, m2, s1x, s1y, s1z, s2x, s2y,
                                              s2z, phic, tc, ra, dec,
                                              polarization)
        signal_h = make_frequency_series(signal['H1'])
        signal_h = extend_waveform_FrequencySeries(signal_h, filter_n)
    elif 'FromDataFile' in signal_approx:
        pass
    else:
        raise IOError("Approximant %s not found.." % signal_approx)

    ###
    # NOW : Set up PSO calculation of the optimal overlap parameter set, i.e. \theta(FF)
    ###

    # 3) INITIALIZE THE WAVEFORM GENERATOR FOR TEMPLATES
    # We allow all intrinsic parameters to vary, and fix them to the signal
    # values, in case only masses or only mass+aligned-spin components are
    # requested to be varied. This fixing is done inside the objective
    # function.
    if tmplt_approx in pywf.fd_approximants():
        generator_tmplt = pywfg.FDomainDetFrameGenerator(
            pywfg.FDomainCBCGenerator,
            0,
            variable_args=[
                'mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z', 'spin2x',
                'spin2y', 'spin2z'
            ],
            detectors=['H1'],
            coa_phase=phic,
            tc=tc,
            ra=ra,
            dec=dec,
            polarization=polarization,
            delta_f=delta_f,
            f_lower=f_lower,
            approximant=tmplt_approx)
    elif tmplt_approx in pywf.td_approximants():
        raise IOError(
            "Time-domain templates not supported yet (TDomainDetFrameGenerator doesn't exist)"
        )
        generator_tmplt = pywfg.TDomainDetFrameGenerator(
            pywfg.TDomainCBCGenerator,
            0,
            variable_args=[
                'mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z', 'spin2x',
                'spin2y', 'spin2z'
            ],
            detectors=['H1'],
            coa_phase=phic,
            tc=tc,
            ra=ra,
            dec=dec,
            polarization=polarization,
            delta_t=delta_t,
            f_lower=f_lower,
            approximant=tmplt_approx)
    elif 'FromDataFile' in tmplt_approx:
        raise RuntimeError(
            "Using **templates** from data files is not implemented yet")
    else:
        raise IOError("Approximant %s not found.." % tmplt_approx)

    # 4) DEFINE AN OBJECTIVE FUNCTION FOR PSO TO MINIMIZE
    def objective_function_fitting_factor(x, *args):
        """
        This function is to be minimized if the fitting factor is to be found
        """
        objective_function_fitting_factor.counter += 1
        # 1) OBTAIN THE TEMPLATE PARAMETERS FROM X. ASSUME THAT ONLY
        # THOSE ARE PASSED THAT ARE NEEDED BY THE GENERATOR
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
        elif len(x) == 8:
            m1, m2, _s1x, _s1y, _s1z, _s2x, _s2y, _s2z = x
        else:
            raise IOError(
                "No of vars %d not supported (should be 2 or 4 or 8)" % len(x))

        # 2) CHECK FOR CONSISTENCY
        if (_s1x**2 + _s1y**2 + _s1z**2) > s_max or (_s2x**2 + _s2y**2 +
                                                     _s2z**2) > s_max:
            return 1e99

        # 2) ASSUME THAT
        signal_h, tmplt_generator = args
        tmplt = tmplt_generator.generate_from_args(m1, m2, _s1x, _s1y, _s1z,
                                                   _s2x, _s2y, _s2z)
        tmplt_h = make_frequency_series(tmplt['H1'])

        if debug:
            print("IN FF Objective-> for parameters:", m1, m2, _s1x, _s1y,
                  _s1z, _s2x, _s2y, _s2z)
        if debug:
            print("IN FF Objective-> Length(tmplt) = %d, making it %d" %
                  (len(tmplt['H1']), filter_n))
        # NOTE: SEOBNRv4 has extra high frequency content, it seems..
        if 'SEOBNRv4_ROM' in tmplt_approx or 'SEOBNRv2_ROM' in tmplt_approx:
            tmplt_h = extend_waveform_FrequencySeries(tmplt_h,
                                                      filter_n,
                                                      force_fit=True)
        else:
            tmplt_h = extend_waveform_FrequencySeries(tmplt_h, filter_n)

        # 3) COMPUTE MATCH
        m, _ = match(signal_h, tmplt_h, psd=psd, low_frequency_cutoff=f_lower)

        if debug:
            print("MATCH IS %.6f for parameters:" % m, m1, m2, _s1x, _s1y,
                  _s1z, _s2x, _s2y, _s2z)

        retval = np.log10(1. - m)

        # We do not want PSO to go berserk, so we stop when FF = 0.999999
        if retval <= -6.0:
            retval = -6.0
        return retval

    objective_function_fitting_factor.counter = 0

    # 5) DEFINE A CONSTRAINT FUNCTION FOR PSO TO RESPECT
    def constraint_function_fitting_factor(x, *args):
        """
        This function implements constraints on the optimization of fitting
        factors:
        1) spin magnitudes on both holes should be <= 1

        """
        if len(x) == 2:
            m1, m2 = x
            s1x = s1y = s1z = s2x = s2y = s2z = 0
        elif len(x) == 4:
            m1, m2, s1z, s2z = x
            s1x = s1y = s2x = s2y = 0
        elif len(x) == 8:
            m1, m2, s1x, s1y, s1z, s2x, s2y, s2z = x
        # 1) Constraint on spin magnitudes
        s1_mag = (s1x**2 + s1y**2 + s1z**2)**0.5
        s2_mag = (s2x**2 + s2y**2 + s2z**2)**0.5
        ##
        if (s1_mag > s_max) or (s2_mag > s_max):
            return -1
        # 2) Constraint on effective spin
        s_eff = (s1z * m1 + s2z * m2) / (m1 + m2)
        ##
        if (s_eff > s_eff_max) or (s_eff < s_eff_min):
            return -1
        # FINALLY) DEFAULT
        return 1

    # 6) FINALLY, CALL THE PSO TO COMPUTE THE FITTING FACTOR
    # 6a) FIRST CONSTRUCT THE FIXED ARGUMENTS FOR THE PSO's OBJECTIVE FUNCTION
    pso_args = (signal_h, generator_tmplt)

    # 6b) NOW SET THE RANGE OF PARAMETERS TO BE PROBED
    mt = m1 + m2 * 1.0
    et = m1 * m2 / mt / mt
    mc = mt * et**0.6
    mc_min = mc * (1.0 - chirp_mass_window)
    mc_max = mc * (1.0 + chirp_mass_window)
    et_max = 0.25
    et_min = 10. / 121.  # Lets say we trust waveform models up to q = 10
    m1_max, _ = pnutils.mchirp_eta_to_mass1_mass2(mc_max, et_min)
    m1_min, _ = pnutils.mchirp_eta_to_mass1_mass2(mc_min, et_max)
    _, m2_max = pnutils.mchirp_eta_to_mass1_mass2(mc_max, et_max)
    _, m2_min = pnutils.mchirp_eta_to_mass1_mass2(mc_min, et_min)
    s_min = -0.99
    s_max = +0.99
    s_eff = (s1z * m1 + s2z * m2) / (m1 + m2)
    s_eff_min = s_eff - effective_spin_window
    s_eff_max = s_eff + effective_spin_window

    if verbose:
        print(m1, m2, mt, et, mc, mc_min, mc_max, et_min, et_max, m1_min,
              m1_max, m2_min, m2_max)

    if vary_masses_only:
        low_lim = [m1_min, m2_min]
        high_lim = [m1_max, m2_max]
    elif vary_masses_and_aligned_spin_only:
        low_lim = [m1_min, m2_min, s_min, s_min]
        high_lim = [m1_max, m2_max, s_max, s_max]
    else:
        low_lim = [m1_min, m2_min, s_min, s_min, s_min, s_min, s_min, s_min]
        high_lim = [m1_max, m2_max, s_max, s_max, s_max, s_max, s_max, s_max]
    #
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
        ra=ra,
        dec=dec,
        polarization=polarization,
        signal_approx=signal_approx,
        signal_file=signal_file,
        tmplt_approx=tmplt_approx,
        tmplt_file=None,
        aligned_spin_tmplt_only=vary_masses_and_aligned_spin_only,
        non_spin_tmplt_only=vary_masses_only,
        f_lower=f_lower,
        sample_rate=sample_rate,
        signal_duration=signal_duration,
        verbose=verbose,
        debug=debug)
    #
    if verbose:
        print("Overlap with aligned_spin_tmplt_only = ",
              vary_masses_and_aligned_spin_only, " and non_spin_tmplt_only = ",
              vary_masses_only, ": ", olap, np.log10(1. - olap))
        sys.stdout.flush()
    #
    idx = 1
    ff = 0.0
    while ff < olap:
        if idx and idx % 2 == 0:
            pso_minfunc *= 0.1
            pso_phig *= 1.1

        if idx > num_retries:
            print(
                "WARNING: Failed to improve on overlap in %d iterations. Set ff = olap now"
                % num_retries)
            ff = olap
            break

        if verbose:
            print("\nTry %d to compute fitting factor" % idx)
            sys.stdout.flush()
        params, ff = pso(objective_function_fitting_factor,
                         low_lim,
                         high_lim,
                         f_ieqcons=constraint_function_fitting_factor,
                         args=pso_args,
                         swarmsize=pso_swarm_size,
                         omega=pso_omega,
                         phip=pso_phip,
                         phig=pso_phig,
                         minfunc=pso_minfunc,
                         maxiter=500,
                         debug=verbose)
        # Restore fitting factor from 1-ff
        ff = 1.0 - 10**ff
        if verbose:
            print("\nLoop will continue till %.12f < %.12f" % (ff, olap))
            sys.stdout.flush()
        idx += 1

    if verbose:
        print("optimization took %d objective func evals" %
              objective_function_fitting_factor.counter)
        sys.stdout.flush()
    #
    # 7) RETURN OPTIMIZED PARAMETERS
    return [params, olap, ff]
    # }}}


######################################################################
######################################################################
#
#     OTHER FUNCTIONS
#
######################################################################
######################################################################


#############################
def overlap_between_waveforms(wav1, wav2, psd=None, f_lower=15.):
    '''
    Return overlap between two waveforms:

    TODO: Add resampling, padding capability.
    '''
    # {{{
    try:
        if psd is None:
            psd = self.psd
    except BaseException:
        raise IOError("Please compute and store PSD")
    #
    len1, len2, lenp = len(wav1), len(wav2), len(psd)
    if len1 != len2:
        raise IOError("Length of waveforms not equal: %d,%d" % (len1, len2))
    if wav1.delta_t != wav2.delta_t:
        raise IOError("Mismatched wave sample rate")
    if len1 != 2 * lenp - 2:
        raise IOError("PSD length inconsistent with waveforms")
    #
    return match(wav1, wav2, psd=psd, low_frequency_cutoff=f_lower)[0]
    # }}}


######################################################################
#      SNR CALCULATIONS


def compute_snr_vs_time(wave, psd, time_step=1e-2, f_lower=15.0):
    # wave should be longer than dt
    assert (time_step < len(wave) * wave.delta_t)

    from numpy import round, arange
    from pycbc.filter import sigma
    from pycbc.types import TimeSeries

    integration_stop_times = arange(time_step,
                                    len(wave) * wave.delta_t, time_step)

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
