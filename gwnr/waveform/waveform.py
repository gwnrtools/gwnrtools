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

from __future__ import (absolute_import, print_function)

import os
import numpy as np
from numpy import any, isinf, isnan

import lal
from pycbc.filter import make_frequency_series
from pycbc import (inject, types, waveform)
from pycbc import DYN_RANGE_FAC
from pycbc.pnutils import *
from glue.ligolw import ligolw, lsctables

from gwnr.utils.types import (extend_waveform_TimeSeries,
                              extend_waveform_FrequencySeries)

os.environ['LD_LIBRARY_PATH'] =\
    '/home/prayush/research/Eccentric_IMRGPR/Code/MergerRingdownModel/C_implementation/bin/'


class ContentHandler(ligolw.LIGOLWContentHandler):
    pass


lsctables.use_in(ContentHandler)


def get_waveform(approximant,
                 phase_order,
                 amplitude_order,
                 spin_order,
                 template_params,
                 start_frequency,
                 sample_rate,
                 length,
                 datafile=None,
                 verbose=False):
    delta_t = 1. / sample_rate
    delta_f = 1. / length
    filter_N = int(length)
    filter_n = filter_N / 2 + 1
    if approximant in waveform.fd_approximants(
    ) and 'Eccentric' not in approximant:
        print("NORMAL FD WAVEFORM for", approximant)
        delta_f = sample_rate / length
        hplus, hcross = waveform.get_fd_waveform(
            template_params,
            approximant=approximant,
            spin_order=spin_order,
            phase_order=phase_order,
            delta_f=delta_f,
            f_lower=start_frequency,
            amplitude_order=amplitude_order)
    elif approximant in waveform.td_approximants(
    ) and 'Eccentric' not in approximant:
        print("NORMAL TD WAVEFORM for", approximant)
        hplus, hcross = waveform.get_td_waveform(
            template_params,
            approximant=approximant,
            spin_order=spin_order,
            phase_order=phase_order,
            delta_t=1.0 / sample_rate,
            f_lower=start_frequency,
            amplitude_order=amplitude_order)
    elif 'EccentricIMR' in approximant:
        # {{{
        # Legacy support
        import sys
        sys.path.append('/home/kuma/grav/kuma/src/Eccentric_IMR/Codes/Python/')
        import EccentricIMR as Ecc
        try:
            mass1 = getattr(template_params, 'mass1')
            mass2 = getattr(template_params, 'mass2')
        except:
            raise RuntimeError("template_params does not have mass1 or mass2!")
        try:
            ecc = getattr(template_params, 'alpha1')
            if 'E0' in approximant:
                ecc = 0
            anom = getattr(template_params, 'alpha2')
            inc = getattr(template_params, 'inclination')
            rtrans = getattr(template_params, 'alpha')
            beta = 0
        except:
            raise RuntimeError(
                "template_params does not have alpha{,1,2} or inclination")
        tol = 1.e-16
        fmin = start_frequency
        sample_rate = sample_rate
        #
        print(" Using phase order: %d" % phase_order, file=sys.stdout)
        sys.stdout.flush()
        hplus, hcross = Ecc.generate_eccentric_waveform(
            mass1,
            mass2,
            ecc,
            anom,
            inc,
            beta,
            tol,
            r_transition=rtrans,
            phase_order=phase_order,
            fmin=fmin,
            sample_rate=sample_rate,
            inspiral_only=False)
        # }}}
    elif 'EccentricInspiral' in approximant:
        # {{{
        # Legacy support
        import sys
        sys.path.append('/home/kuma/grav/kuma/src/Eccentric_IMR/Codes/Python/')
        import EccentricIMR as Ecc
        try:
            mass1 = getattr(template_params, 'mass1')
            mass2 = getattr(template_params, 'mass2')
        except:
            raise RuntimeError("template_params does not have mass1 or mass2!")
        try:
            ecc = getattr(template_params, 'alpha1')
            if 'E0' in approximant:
                ecc = 0
            anom = getattr(template_params, 'alpha2')
            inc = getattr(template_params, 'inclination')
            beta = getattr(template_params, 'alpha')
        except:
            raise RuntimeError(
                "template_params does not have alpha{,1,2} or inclination")
        tol = 1.e-16
        fmin = start_frequency
        sample_rate = sample_rate
        #
        hplus, hcross = Ecc.generate_eccentric_waveform(
            mass1,
            mass2,
            ecc,
            anom,
            inc,
            beta,
            tol,
            phase_order=phase_order,
            fmin=fmin,
            sample_rate=sample_rate,
            inspiral_only=True)
        # }}}
    elif 'EccentricFD' in approximant:
        # {{{
        # Legacy support
        import lalsimulation as ls
        import lal
        delta_f = sample_rate / length
        try:
            mass1 = getattr(template_params, 'mass1')
            mass2 = getattr(template_params, 'mass2')
        except:
            raise RuntimeError("template_params does not have mass1 or mass2!")
        try:
            ecc = getattr(template_params, 'alpha1')
            if 'E0' in approximant:
                ecc = 0
            anom = getattr(template_params, 'alpha2')
            inc = getattr(template_params, 'inclination')
        except:
            raise RuntimeError(
                "template_params does not have alpha{1,2} or inclination")
        eccPar = ls.SimInspiralCreateTestGRParam("inclination_azimuth", inc)
        ls.SimInspiralAddTestGRParam(eccPar, "e_min", ecc)
        fmin = start_frequency
        fmax = sample_rate / 2
        #
        thp, thc = ls.SimInspiralChooseFDWaveform(
            0, delta_f, mass1 * lal.MSUN_SI, mass2 * lal.MSUN_SI, 0, 0, 0, 0,
            0, 0, fmin, fmax, 0, 1.e6 * lal.PC_SI, inc, 0, 0, None, eccPar, -1,
            7, ls.EccentricFD)
        hplus = types.FrequencySeries(thp.data.data[:],
                                      delta_f=thp.deltaF,
                                      epoch=thp.epoch)
        hcross = types.FrequencySeries(thc.data.data[:],
                                       delta_f=thc.deltaF,
                                       epoch=thc.epoch)
        # }}}
    elif 'FromDataFile' in approximant:
        # {{{
        # Legacy support
        if not os.path.exists(datafile):
            raise IOError("File %s not found!" % datafile)
        if verbose:
            print("Reading from data file %s" % datafile)

        # Figure out waveform parameters from filename
        #q_value, M_value, w_value, _, _ = EA.get_q_m_e_pn_o_from_filename(datafile)
        q_value, M_value, w_value = EA.get_q_m_e_from_filename(datafile)

        # Read data, down-sample (assume data file is more finely sampled than
        # needed, i.e. interpolation is NOT supported, nor will be)
        data = np.loadtxt(datafile)
        dt = data[1, 0] - data[0, 0]
        delta_t = 1. / sample_rate
        downsample_ratio = delta_t / dt
        if not approx_equal(downsample_ratio, np.int(downsample_ratio)):
            raise RuntimeError(
                "Cannot handling resampling at a fractional factor = %e" %
                downsample_ratio)
        elif verbose:
            print("Downsampling by a factor of %d" % int(downsample_ratio))
        h_real = types.TimeSeries(data[::int(downsample_ratio), 1] /
                                  DYN_RANGE_FAC,
                                  delta_t=delta_t)
        h_imag = types.TimeSeries(data[::int(downsample_ratio), 2] /
                                  DYN_RANGE_FAC,
                                  delta_t=delta_t)

        if verbose:
            print("max, min,len of h_real = ", max(h_real.data),
                  min(h_real.data), len(h_real.data))

        # Compute Strain
        tmplt_pars = template_params
        wav = generate_detector_strain(tmplt_pars, h_real, h_imag)
        wav = extend_waveform_TimeSeries(wav, filter_N)

        # Return TimeSeries with (m1, m2, w_value)
        m1, m2 = mtotal_eta_to_mass1_mass2(M_value,
                                           q_value / (1. + q_value)**2)
        htilde = make_frequency_series(wav)
        htilde = extend_waveform_FrequencySeries(htilde, filter_n)

        if verbose:
            print("ISNAN(htilde from file) = ", np.any(np.isnan(htilde.data)))
        return htilde, [m1, m2, w_value, dt]
        # }}}
    else:
        raise IOError(".. APPROXIMANT %s not found.." % approximant)
    ##
    hvec = hplus
    htilde = make_frequency_series(hvec)
    htilde = extend_waveform_FrequencySeries(htilde, filter_n)

    if any(isnan(hplus.data)) or any(isnan(hcross.data)):
        print("..### %s hplus or hcross have NANS!!" % approximant)
    if any(isinf(hplus.data)) or any(isinf(hcross.data)):
        print("..### %s hplus or hcross have INFS!!" % approximant)
    if any(isnan(htilde.data)):
        print("..### %s Fourier transform htilde has NANS!!" % approximant)
    if any(isinf(htilde.data)):
        print("..### %s Fourier transform htilde has INFS!!" % approximant)
    return htilde


def project_polarizations_onto_detector(ifo_name,
                                        hp,
                                        hc,
                                        ra,
                                        dec,
                                        pol,
                                        tc,
                                        taper_mode='TAPER_NONE',
                                        amp_scaler=1.0):
    '''
    Inputs
    ------
    ifo_name: str
        Abbreviated detector name, e.g. H1, L1, V1, G1.

    hp: TimeSeries
        Plus (+) polarization time series

    hc: TimeSeries:
        Cross (x) polarization time series

    ra: float
        right ascension angle in radians

    dec: float
        declination angle in radians

    pol: float
        polarization angle in radians

    tc: float
        coalescences time

    taper_mode : string
        Should be one of ('TAPER_NONE', 'TAPER_START', 'TAPER_END',
        'TAPER_STARTEND', 'start', 'end', 'startend') - NB 'TAPER_NONE' will
        not change the series!

    amp_scaler: float
        Scaling factor by which the polarizations are divided, before
        projecting them onto the detector
    '''

    class MyInj(object):

        def __init__(self,
                     ra,
                     dec,
                     pol,
                     tc,
                     taper_mode='TAPER_START',
                     proj_method='lal'):
            self.tc = self.time_geocent = lal.LIGOTimeGPS(tc)
            self.ra = self.longitude = ra
            self.dec = self.latitude = dec
            self.polarization = pol
            self.taper = taper_mode
            self.detector_projection_method = proj_method

    inj = MyInj(ra, dec, pol, tc, taper_mode=taper_mode)
    strain = inject.projector(ifo_name, inj, hp, hc, distance_scale=amp_scaler)
    return strain
