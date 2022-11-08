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
"""
Functions to align waveforms
"""

# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
from __future__ import (absolute_import, print_function)

import sys
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize_scalar

from pycbc.filter import (make_frequency_series, match, matched_filter_core,
                          overlap_cplx)
from pycbc.types import TimeSeries
from pycbc.waveform import (amplitude_from_polarizations,
                            phase_from_polarizations)
from pycbc.pnutils import *
from glue.ligolw import ligolw, lsctables

from gwnr.utils import (find_nearest, trim_leading_zeros, trim_trailing_zeros)


class ContentHandler(ligolw.LIGOLWContentHandler):
    pass


lsctables.use_in(ContentHandler)


def shift_waveform_phase_time(hp,
                              hc,
                              t_shift,
                              ph_shift,
                              shift_epochs_only=True,
                              trim_leading=False,
                              trim_trailing=True,
                              verbose=False):
    """
    Input:  hp, hc, where h = hp(t) + i hx(t) = Amp(t) * exp(-i * Phi(t))
    Output: hp, hc, where h = Amp(t - t_c) * exp( -i * [Phi(t - t_c) + phi_c] )
    """
    hpnew = TimeSeries(hp,
                       epoch=hp._epoch,
                       delta_t=hp.delta_t,
                       dtype=hp.dtype,
                       copy=True)
    hcnew = TimeSeries(hc,
                       epoch=hc._epoch,
                       delta_t=hc.delta_t,
                       dtype=hc.dtype,
                       copy=True)
    # First apply phase shift
    if ph_shift != 0.:
        amplitude = amplitude_from_polarizations(hpnew, hcnew)
        phase = phase_from_polarizations(hpnew,
                                         hcnew,
                                         remove_start_phase=False)
        if verbose:
            print(("shifting by %f radians" % ph_shift))
        phase = phase + ph_shift
        hpnew = TimeSeries(amplitude * np.cos(phase),
                           epoch=hpnew._epoch,
                           delta_t=hpnew.delta_t,
                           dtype=hpnew.dtype)
        hcnew = TimeSeries(amplitude * np.sin(phase),
                           epoch=hcnew._epoch,
                           delta_t=hcnew.delta_t,
                           dtype=hcnew.dtype)  # Now apply time shift
    # Only positive time shifts can be applied by rolling forward data
    # if negative time shifts are asked for, we can only shift epochs
    if t_shift != 0:
        if shift_epochs_only or t_shift <= 0:
            hpnew._epoch += t_shift
            hcnew._epoch += t_shift
        else:
            time_vals = hpnew.get_sample_times()
            hpnewI = InterpolatedUnivariateSpline(time_vals, hpnew.data)
            hcnewI = InterpolatedUnivariateSpline(time_vals, hcnew.data)

            shifted_time_vals = np.arange(
                time_vals[0],  # start of new timeseries
                time_vals[-1] + t_shift + 0 * hpnew.delta_t,  # end of it
                hpnew.delta_t)
            mask_times_to_reevaluate = [
                (shifted_time_vals - t_shift >= time_vals[0]) &
                (shifted_time_vals - t_shift <= time_vals[-1])
            ]
            hp_shifted = np.zeros(len(shifted_time_vals))
            hp_shifted[mask_times_to_reevaluate] = hpnewI(
                shifted_time_vals[mask_times_to_reevaluate] - t_shift)
            hpnew = TimeSeries(hp_shifted,
                               epoch=hpnew._epoch + t_shift,
                               delta_t=hpnew.delta_t)

            hc_shifted = np.zeros(len(shifted_time_vals))
            hc_shifted[mask_times_to_reevaluate] = hcnewI(
                shifted_time_vals[mask_times_to_reevaluate] - t_shift)
            hcnew = TimeSeries(hc_shifted,
                               epoch=hcnew._epoch + t_shift,
                               delta_t=hcnew.delta_t)

    if trim_trailing:
        hpnew = trim_trailing_zeros(hpnew)
        hcnew = trim_trailing_zeros(hcnew)
    if trim_leading:
        hpnew = trim_leading_zeros(hpnew)
        hcnew = trim_leading_zeros(hcnew)
    return hpnew, hcnew


def shift_waveform_phase(hp,
                         hc,
                         ph_shift,
                         trim_leading=False,
                         trim_trailing=True,
                         verbose=False):
    """
    Input:  hp, hc, where h = hp(t) + i hx(t) = Amp(t) * exp(-i * Phi(t))
    Output: hp, hc, where h = Amp(t) * exp( -i * [Phi(t) + phi_c] )
    """
    return shift_waveform_phase_time(hp,
                                     hc,
                                     0,
                                     ph_shift,
                                     trim_leading=trim_leading,
                                     trim_trailing=trim_trailing,
                                     verbose=verbose)


def shift_waveform_time(hp,
                        hc,
                        t_shift,
                        shift_epochs_only=True,
                        trim_leading=False,
                        trim_trailing=True,
                        verbose=False):
    """
    Input:  hp, hc, where h = hp(t) + i hx(t) = Amp(t) * exp(-i * Phi(t))
    Output: hp, hc, where h = Amp(t - t_c) * exp( -i * [Phi(t - t_c)] )
    """
    return shift_waveform_phase_time(hp,
                                     hc,
                                     t_shift,
                                     shift_epochs_only=shift_epochs_only,
                                     trim_leading=trim_leading,
                                     trim_trailing=trim_trailing,
                                     verbose=verbose)


def align_waveforms_amplitude_peak(hplus1,
                                   hcross1,
                                   hplus2,
                                   hcross2,
                                   shift_epochs_only=True,
                                   trim_leading=False,
                                   trim_trailing=True,
                                   verbose=False):
    """
    Align the two waveforms, shifting only one of the two.
        - AT the Amplitude PEAK
    """
    _dt = 1.0
    if type(hplus1) == TimeSeries:
        _dt = hplus1.delta_t
    elif type(hplus2) == TimeSeries:
        _dt = hplus2.delta_t
    if type(hcross1) != TimeSeries:
        _dt = hcross1.delta_t
    if type(hcross2) != TimeSeries:
        _dt = hcross2.delta_t

    hp1 = TimeSeries(hplus1, delta_t=_dt, copy=True)
    hc1 = TimeSeries(hcross1, delta_t=_dt, copy=True)
    hp2 = TimeSeries(hplus2, delta_t=_dt, copy=True)
    hc2 = TimeSeries(hcross2, delta_t=_dt, copy=True)

    # Get amplitude peak for 1st set of polarizations
    amp1 = amplitude_from_polarizations(hp1, hc1)
    amp1I = InterpolatedUnivariateSpline(amp1.sample_times, -1 * amp1.data)
    x0 = np.float64(
        amp1.sample_times[np.where(amp1.data == max(amp1.data))[0][0]])
    tmp = minimize_scalar(amp1I,
                          x0,
                          method='bounded',
                          bounds=(x0 - 10 * amp1.delta_t,
                                  x0 + 10 * amp1.delta_t))
    h1_max_amp_time = tmp['x']
    h1_max_amp = -1 * tmp['fun']

    # Get amplitude peak for 1st set of polarizations
    amp2 = amplitude_from_polarizations(hp2, hc2)
    amp2I = InterpolatedUnivariateSpline(amp2.sample_times, -1 * amp2.data)
    x0 = np.float64(
        amp2.sample_times[(np.where(amp2.data == max(amp2.data))[0][0])])
    tmp = minimize_scalar(amp2I,
                          x0,
                          method='bounded',
                          bounds=(x0 - 10 * amp2.delta_t,
                                  x0 + 10 * amp2.delta_t))
    h2_max_amp_time = tmp['x']
    h2_max_amp = -1 * tmp['fun']
    if verbose:
        print(("h1 max time = %f, epoch = %f" %
               (h1_max_amp_time, float(hp1._epoch))))
        print(("h2 max time = %f, epoch = %f" %
               (h2_max_amp_time, float(hp2._epoch))))

    # Amplitude location from the start
    t1 = h1_max_amp_time
    t2 = h2_max_amp_time
    t_shift = t1 - t2

    if verbose:
        print(("time shift = %f to be added to waveform 2" % t_shift))

    # Find phase shift
    phs1 = phase_from_polarizations(hp1, hc1, remove_start_phase=False)
    phs2 = phase_from_polarizations(hp2, hc2, remove_start_phase=False)
    phs1I = InterpolatedUnivariateSpline(phs1.sample_times, phs1.data)
    phs2I = InterpolatedUnivariateSpline(phs2.sample_times, phs2.data)

    ph1 = phs1I(h1_max_amp_time)
    ph2 = phs2I(h2_max_amp_time)
    ph_shift = np.float64(ph1 - ph2)

    if verbose:
        print(("phase1 at peak idx = %d, = %f" %
               (int(np.round(t1 / hp1.delta_t)), ph1)))
        print(("phase2 at peak idx = %d, = %f" %
               (int(np.round(t2 / hp2.delta_t)), ph2)))
        print(("phase shift = %f, time shift = %f" % (ph_shift, t_shift)))
    #
    # Shift whichever needs to be shifted to future time.
    # Shifting back in time is tricky.
    if shift_epochs_only:
        hp2, hc2 = shift_waveform_phase_time(hp2,
                                             hc2,
                                             t_shift,
                                             ph_shift,
                                             shift_epochs_only=True,
                                             verbose=verbose)
        # Finally, shift everything's peak to t=0
        hp1._epoch -= h1_max_amp_time
        hc1._epoch -= h1_max_amp_time
        hp2._epoch -= h1_max_amp_time
        hc2._epoch -= h1_max_amp_time
        # Trim leading zeros. If time shifts are actual shifts of data along the
        # array, the leading zeros have meaning and cannot be trimmed.
        if trim_leading and shift_epochs_only:
            hp1 = trim_leading_zeros(hp1)
            hc1 = trim_leading_zeros(hc1)
            hp2 = trim_leading_zeros(hp2)
            hc2 = trim_leading_zeros(hc2)
    else:
        t_shift += (amp2.sample_times[0] - amp1.sample_times[0])
        if verbose:
            print("phase shift is actually = {}, time shift = {}".format(
                ph_shift, t_shift))

        if t_shift >= 0:
            hp2, hc2 = shift_waveform_phase_time(hp2,
                                                 hc2,
                                                 t_shift,
                                                 ph_shift,
                                                 shift_epochs_only=False,
                                                 verbose=verbose)
            hp1._epoch -= h1_max_amp_time
            hc1._epoch -= h1_max_amp_time
            hp2._epoch = hp1._epoch
            hc2._epoch = hc1._epoch
        else:
            hp1, hc1 = shift_waveform_phase_time(hp1,
                                                 hc1,
                                                 -1 * t_shift,
                                                 ph_shift,
                                                 shift_epochs_only=False,
                                                 verbose=verbose)
            hp2._epoch -= h2_max_amp_time
            hc2._epoch -= h2_max_amp_time
            hp1._epoch = hp2._epoch
            hc1._epoch = hc2._epoch

    # Trim any trailing zeros
    if trim_trailing:
        hp1 = trim_trailing_zeros(hp1)
        hc1 = trim_trailing_zeros(hc1)
        hp2 = trim_trailing_zeros(hp2)
        hc2 = trim_trailing_zeros(hc2)

    return hp1, hc1, hp2, hc2


def align_waveforms_at_frequency(hplus1,
                                 hcross1,
                                 hplus2,
                                 hcross2,
                                 falign,
                                 trim_leading=False,
                                 trim_trailing=True,
                                 verbose=False):
    #
    # Find amplitude peaks
    #
    hp1 = TimeSeries(hplus1)
    hc1 = TimeSeries(hcross1)
    hp2 = TimeSeries(hplus2)
    hc2 = TimeSeries(hcross2)
    #
    # Get time at flign for wave 1
    #
    freq1 = frequency_from_polarizations(hp1, hc1)
    obj_func = np.abs(np.abs(freq1.data) - falign)
    f1I = InterpolatedUnivariateSpline(freq1.sample_times, obj_func)
    id_start = np.where(obj_func == np.min(obj_func))[0][0]
    for idx in range(id_start, len(freq1)):
        if freq1[idx] > 2 * falign and freq1[idx + 1] > 2 * falign:
            break
    tmp = minimize_scalar(f1I,
                          freq1.sample_times[id_start],
                          method='bounded',
                          bounds=(freq1.sample_times[id_start],
                                  freq1.sample_times[idx]))
    f1_align_time = tmp['x']
    #
    # Get time at flign for wave 2
    #
    freq2 = frequency_from_polarizations(hp2, hc2)
    obj_func = np.abs(np.abs(freq2.data) - falign)
    f2I = InterpolatedUnivariateSpline(freq2.sample_times, obj_func)
    id_start = np.where(obj_func == np.min(obj_func))[0][0]
    for idx in range(id_start, len(freq2)):
        if freq2[idx] > 2 * falign and freq2[idx + 1] > 2 * falign:
            break
    tmp = minimize_scalar(f2I,
                          freq2.sample_times[id_start],
                          method='bounded',
                          bounds=(freq2.sample_times[id_start],
                                  freq2.sample_times[idx]))
    f2_align_time = tmp['x']
    #
    t1 = f1_align_time
    t2 = f2_align_time
    t_shift = t1 - t2
    if verbose:
        print(("time shift = %f to be added to waveform 2" % t_shift))
    #
    # Find phase shift at the time
    #
    phs1 = phase_from_polarizations(hp1, hc1, remove_start_phase=False)
    phs2 = phase_from_polarizations(hp2, hc2, remove_start_phase=False)
    phs1I = InterpolatedUnivariateSpline(phs1.sample_times, phs1.data)
    phs2I = InterpolatedUnivariateSpline(phs2.sample_times, phs2.data)

    ph1 = phs1I(f1_align_time)
    ph2 = phs2I(f2_align_time)
    ph_shift = (ph1 - ph2) * 1
    if verbose:
        print(("time @ f1 = %f : %f" % (falign, f1_align_time)))
        print(("time @ f2 = %f : %f" % (falign, f2_align_time)))
        print(("ph1 = %f, ph2 = %f" % (ph1, ph2)))
        print(("phase shift = %f, time shift = %f" % (ph_shift, t_shift)))
        print(("type of ph_shift, t_shift: ", type(ph_shift), type(t_shift)))
    #
    # Shift whichever needs to be shifted to future time.
    # Shifting back in time is tricky.
    hp2, hc2 = shift_waveform_phase_time(hp2,
                                         hc2,
                                         t_shift,
                                         ph_shift,
                                         verbose=verbose)
    #
    if trim_trailing:
        hp1 = trim_trailing_zeros(hp1)
        hc1 = trim_trailing_zeros(hc1)
        hp2 = trim_trailing_zeros(hp2)
        hc2 = trim_trailing_zeros(hc2)
    if trim_leading:
        hp1 = trim_leading_zeros(hp1)
        hc1 = trim_leading_zeros(hc1)
        hp2 = trim_leading_zeros(hp2)
        hc2 = trim_leading_zeros(hc2)
    #
    return hp1, hc1, hp2, hc2


def align_waveforms_optimally(hplus1,
                              hcross1,
                              hplus2,
                              hcross2,
                              psd='aLIGOZeroDetHighPower',
                              low_frequency_cutoff=None,
                              high_frequency_cutoff=None,
                              tsign=1,
                              phsign=-1,
                              verify=True,
                              phase_tolerance=1e-3,
                              overlap_tolerance=1e-3,
                              trim_leading=False,
                              trim_trailing=False,
                              verbose=False):
    """
    Align waveforms such that their inner product (noise weighted) is optimal
    without requiring any phase or time shift.

    The appropriate time and phase shifts are determined iteratively and applied
    to the second set of (hplus, hcross) vectors.
    """
    #############################################################################
    # First copy over data into local memory, ensure lengths of time and
    # frequency domain vectors are consistent, and compute the maximized overlap
    #
    # 1) Cast into time-series
    h_plus1 = TimeSeries(hplus1,
                         epoch=hplus1._epoch,
                         delta_t=hplus1.delta_t,
                         dtype=hplus1.dtype,
                         copy=True)
    h_cross1 = TimeSeries(hcross1,
                          epoch=hplus1._epoch,
                          delta_t=hplus1.delta_t,
                          dtype=hplus1.dtype,
                          copy=True)
    h_plus2 = TimeSeries(hplus2,
                         epoch=hplus2._epoch,
                         delta_t=hplus2.delta_t,
                         dtype=hplus2.dtype,
                         copy=True)
    h_cross2 = TimeSeries(hcross2,
                          epoch=hplus2._epoch,
                          delta_t=hplus2.delta_t,
                          dtype=hplus2.dtype,
                          copy=True)
    #
    # 2) Ensure both input hplus vectors are equal in length
    if len(hplus2) > len(hplus1):
        h_plus1.append_zeros(len(hplus2) - len(hplus1))
        h_cross1.append_zeros(len(hplus2) - len(hplus1))
    elif len(hplus2) < len(hplus1):
        h_plus2.append_zeros(len(hplus1) - len(hplus2))
        h_cross2.append_zeros(len(hplus1) - len(hplus2))
    #
    # 3) Set the upper frequency cutoff to Nyquist if not set by User
    if high_frequency_cutoff == None:
        high_frequency_cutoff = 1. / h_plus1.delta_t / 2.
    #
    # 4) Compute LIGO noise psd
    if psd == None:
        raise IOError("Need compatible psd [or name] as input!")
    elif type(psd) == str:
        htilde = make_frequency_series(h_plus1)
        psd_name = psd
        psd = from_string(psd_name, len(htilde), htilde.delta_f,
                          low_frequency_cutoff)
    ##
    # 5) Calculate Overlap (maximized) before alignment
    m = match(h_plus1,
              h_plus2,
              psd=psd,
              low_frequency_cutoff=low_frequency_cutoff,
              high_frequency_cutoff=high_frequency_cutoff)
    optimal_overlap = m[0]  # FIXME
    if verbose:
        print(("Overlap BEFORE ALIGNMENT:",
               overlap_cplx(h_plus1,
                            h_plus2,
                            psd=psd,
                            low_frequency_cutoff=low_frequency_cutoff,
                            high_frequency_cutoff=high_frequency_cutoff,
                            normalized=True)))
        print(("Match BEFORE ALIGNMENT:", m))
    #############################################################################
    # Iterate to obtain the correct phase and time shifts, using which we
    # align the two waveforms such that their unmaximized and maximized overlaps
    # agree.

    #
    # 1) Initialize phase/time offset counters
    t_shift_counter = 0
    ph_shift_counter = 0
    #
    # 2) Initialize initial garbage values to enter the while loop
    idx = 0
    ph_shift = t_shift = 1e9
    olap = 0 + 0j
    #
    # 3) Iteration begins
    # >>>>>>
    while np.abs(ph_shift) > phase_tolerance or \
            np.abs(t_shift) > h_plus1.delta_t or \
            np.abs(np.abs(olap.real) - optimal_overlap) > overlap_tolerance:
        if idx == 0:
            hp2, hc2 = h_plus2, h_cross2
        #
        # 1) Determine the phase and time shifts for optimal match
        #    by comparing hplus1/hcross1 with hp2/hc2 which is phase/time shifted
        #    in previous iteration
        snr, corr, snr_norm = matched_filter_core(h_plus1, hp2, psd,
                                                  low_frequency_cutoff,
                                                  high_frequency_cutoff, None)
        max_snr, max_id = snr.abs_max_loc()

        if max_id != 0:
            t_shift = snr.delta_t * (len(snr) - max_id)
        else:
            t_shift = snr.delta_t * max_id

        ph_shift = np.angle(snr[max_id])

        #
        # 2) Add them to running time/phase offset counter
        t_shift_counter += t_shift
        ph_shift_counter += ph_shift
        #
        if verbose:
            print((" >> Iteration %d\n" % (idx + 1)))
            print(("max_id = %d, id_shift = %d" %
                   (max_id, int(t_shift / snr.delta_t))))
            print(("t_shift = %f,\n ph_shift = %f" % (t_shift, ph_shift)))
        #
        ####
        # 3) Shift the second hp/hc pair (ORIGINAL) by cumulative phase/time offset
        hp2, hc2 = shift_waveform_phase_time(h_plus2,
                                             h_cross2,
                                             tsign * t_shift_counter,
                                             phsign * ph_shift_counter,
                                             verbose=verbose)
        #
        ###
        # 4) As time shifting can change array lengths, equalize again, compute psd
        ##
        if len(h_plus1) > len(hp2):
            hp2.append_zeros(len(h_plus1) - len(hp2))
            htilde = make_frequency_series(h_plus1)
            psd = from_string(psd_name, len(htilde), htilde.delta_f,
                              low_frequency_cutoff)
        elif len(h_plus1) < len(hp2):
            h_plus1.append_zeros(len(hp2) - len(h_plus1))
            htilde = make_frequency_series(h_plus1)
            psd = from_string(psd_name, len(htilde), htilde.delta_f,
                              low_frequency_cutoff)
        #
        # 5) Compute UNMAXIMIZED overlap.
        olap = overlap_cplx(h_plus1,
                            hp2,
                            psd=psd,
                            low_frequency_cutoff=low_frequency_cutoff,
                            high_frequency_cutoff=high_frequency_cutoff,
                            normalized=True)
        if verbose:
            print(("Overlap AFTER ALIGNMENT = ", olap))
            print(("Optimal Overlap = ", optimal_overlap))
        #
        idx += 1
        if verbose:
            print("\n")
    # >>>>>>
    # 3) Iteration ended.

    #############################################################################
    # Verify the alignment
    ###
    if verify:
        #
        print("Verifying time alignment...")
        #
        # 1) Determine the phase and time shifts for optimal match
        snr, corr, snr_norm = matched_filter_core(h_plus1, hp2, psd,
                                                  low_frequency_cutoff,
                                                  high_frequency_cutoff, None)
        max_snr, max_id = snr.abs_max_loc()
        if verbose:
            print(
                ("Post-Alignment Index of MAX SNR (should be 0 or 1 or %d): %d"
                 % (len(snr) - 1, max_id)))
            print(("Length of whole SNR time-series: ", len(snr)))
        #
        # 2) Test if current time shift is within tolerance
        if max_id != 0 and max_id != 1 and \
                max_id != (len(snr)-1) and max_id != (len(snr)-2):
            raise RuntimeError("Warning: ALIGNMENT NOT CORRECT (see above)")
        else:
            print("Alignment in time correct..")
        #
        # 3) Test if current phase shift is within tolerance
        print("Verifying phase alignment...")
        ph_shift = np.angle(snr[max_id])
        if np.abs(ph_shift) > phase_tolerance:
            if verbose:
                print(("dphi, dphi+pi, dphi-pi: ", ph_shift, ph_shift + np.pi,
                       ph_shift - np.pi))
                print(
                    ("dphi/pi, dphi*pi: ", ph_shift / np.pi, ph_shift * np.pi))
            raise RuntimeError(
                "Warning: Phasing alignment possibly incorrect.")
        else:
            if verbose:
                print(("Post-Alignmend Phase shift (should be < %.2e): %.2e" %
                       (phase_tolerance, np.abs(ph_shift))))
            print(("Alignment in phasing correct.. (within tol %.2e)" %
                   phase_tolerance))
        #

    #############################################################################
    # TRIM the output arrays and return
    if trim_trailing:
        hp2 = trim_trailing_zeros(hp2)
        hc2 = trim_trailing_zeros(hc2)
    if trim_leading:
        hp2 = trim_leading_zeros(hp2)
        hc2 = trim_leading_zeros(hc2)
    #
    return hplus1, hcross1, hp2, hc2


def align_curves(x1,
                 y1,
                 x2,
                 y2,
                 delta_x=None,
                 x_low_lim=None,
                 x_high_lim=None,
                 offset_low_lim=None,
                 offset_high_lim=None,
                 num_retries=10,
                 eps_solution=1e-6,
                 verbose=False,
                 debug=False):
    """
    This function maximizes the alignment between two 1D functions, varying the
    x coordinate alone. It minimizes:

    f(x_offset) := \int_{x_low_lim}^{x_high_lim} |y2(x + x_offset) - y1(x)| dx

    over x_offset.

    Notes:
    ------

    1) [x_low_lim, x_high_lim] are with respect to the (x1, y1) pair.
       The other pair (x1, y2) is the one effectively shifted.

    2) Not specifying [x_low_lim, x_high_lim] is equivalent to integrating
       the mean-square difference over the complete (x2) vector.
    """
    if delta_x is None: delta_x = x1[1] - x1[0]
    if x_low_lim is None: x_low_lim = np.min(x1)
    if x_high_lim is None: x_high_lim = np.max(x1)

    def multiple_of_dx(x_):
        return int(np.round(x_ / delta_x)) * delta_x

    def objective_function_alignment(x, *args):
        objective_function_alignment.counter += 1

        # Get offset for this iteration
        x_offset = multiple_of_dx(x)

        # Get original x1 and x2 values, as well as splines
        # for y1(x2) and y2(x2)
        x1, y1, x2_, y2, low_lim, high_lim = args
        x2 = x2_ + x_offset

        s = 0
        for idx1, x1val in enumerate(x1):
            if x1val < low_lim or x1val > high_lim:
                continue
            idx2, x2val = find_nearest(x2, x1val)
            if np.abs(x2val - x1val) > delta_x:
                if objective_function_alignment.counter % 100 == 0 and verbose:
                    print(np.abs(x2val - x1val))
            s += (y2[idx2] - y1[idx1])**2

        if objective_function_alignment.counter % 100 == 0:
            print("Objective function for offset = %.3f is %.6f" % (x, s))

        return s

    objective_function_alignment.counter = 0

    opt_args = (x1, y1, x2, y2, x_low_lim, x_high_lim)

    if debug:
        print("Testing objective function")
        print("Offset 0: ", objective_function_alignment(0, *opt_args))
        print("Offset 550: ", objective_function_alignment(-0.5, *opt_args))

    # SET THE RANGE OF OFFSETS TO BE PROBED
    xd1, xd2 = x1[-1] - x2[0], x1[0] - x2[-1]

    if offset_low_lim is None: x_min = np.min([xd1, xd2])
    else: x_min = offset_low_lim

    if offset_high_lim is None: x_max = np.max([xd1, xd2])
    else: x_max = offset_high_lim

    if verbose:
        print("Searching for optimal offset in range:", x_min, " to ", x_max)

    # Minimize the objective function
    for idx in range(num_retries):
        if verbose:
            print("\nTry %d to compute alignment" % idx, file=sys.stdout)
            sys.stdout.flush()
        retval = minimize_scalar(objective_function_alignment,
                                 args=opt_args,
                                 bounds=(x_min, x_max))
        if objective_function_alignment(retval.x, *opt_args) < eps_solution:
            if verbose:
                print("optimization took {} objective func evals".format(
                    objective_function_alignment.counter))
                sys.stdout.flush()
            return [multiple_of_dx(retval.x), retval]

    raise RuntimeError("""Cannot solve the problem without either
         a) INcreasing delta_x, or
         b) interpolation.
        Vectors are not sampled finely enough.""")


######################################################################
#
#       DEPRECATED FUNCTIONS
#


def align_waveforms_suboptimally(hplus1,
                                 hcross1,
                                 hplus2,
                                 hcross2,
                                 psd='aLIGOZeroDetHighPower',
                                 low_frequency_cutoff=None,
                                 high_frequency_cutoff=None,
                                 tsign=1,
                                 phsign=1,
                                 verify=True,
                                 trim_leading=False,
                                 trim_trailing=False,
                                 verbose=False):
    # Cast into time-series
    h_plus1 = TimeSeries(hplus1,
                         epoch=hplus1._epoch,
                         delta_t=hplus1.delta_t,
                         dtype=hplus1.dtype)
    h_cross1 = TimeSeries(hcross1,
                          epoch=hplus1._epoch,
                          delta_t=hplus1.delta_t,
                          dtype=hplus1.dtype)
    h_plus2 = TimeSeries(hplus2,
                         epoch=hplus2._epoch,
                         delta_t=hplus2.delta_t,
                         dtype=hplus2.dtype)
    h_cross2 = TimeSeries(hcross2,
                          epoch=hplus2._epoch,
                          delta_t=hplus2.delta_t,
                          dtype=hplus2.dtype)
    #
    # Ensure both input hplus vectors are equal in length
    if len(hplus2) > len(hplus1):
        h_plus1.append_zeros(len(hplus2) - len(hplus1))
        h_cross1.append_zeros(len(hplus2) - len(hplus1))
    elif len(hplus2) < len(hplus1):
        h_plus2.append_zeros(len(hplus1) - len(hplus2))
        h_cross2.append_zeros(len(hplus1) - len(hplus2))
    #
    htilde = make_frequency_series(h_plus1)
    stilde = make_frequency_series(h_plus2)
    #
    if high_frequency_cutoff == None:
        high_frequency_cutoff = 1. / h_plus1.delta_t / 2.
    #
    if psd == None:
        raise IOError("Need compatible psd [or name] as input!")
    elif type(psd) == str:
        psd_name = psd
        psd = from_string(psd_name, len(htilde), htilde.delta_f,
                          low_frequency_cutoff)
    #
    # Determine the phase and time shifts for optimal match
    snr, corr, snr_norm = matched_filter_core(
        htilde,
        stilde,
        # h_plus1, h_plus2,
        psd,
        low_frequency_cutoff,
        high_frequency_cutoff,
        None)
    max_snr, max_id = snr.abs_max_loc()

    if max_id != 0:
        t_shift = snr.delta_t * (len(snr) - max_id)
    else:
        t_shift = snr.delta_t * max_id

    ph_shift = np.angle(snr[max_id]) - 0.24850315030 - 0.0465881735639
    #
    if verbose:
        print(("max_id = %d, id_shift = %d" %
               (max_id, int(t_shift / snr.delta_t))))
        print(("t_shift = %f,\n ph_shift = %f" % (t_shift, ph_shift)))
    #
    # print(OVERLAPS
    if verbose:
        print(("Overlap BEFORE ALIGNMENT:",
               overlap_cplx(h_plus1,
                            h_plus2,
                            psd=psd,
                            low_frequency_cutoff=low_frequency_cutoff,
                            high_frequency_cutoff=high_frequency_cutoff,
                            normalized=True)))
        print(("Match BEFORE ALIGNMENT:",
               match(h_plus1,
                     h_plus2,
                     psd=psd,
                     low_frequency_cutoff=low_frequency_cutoff,
                     high_frequency_cutoff=high_frequency_cutoff)))

    # Shift whichever needs to be shifted to future time.
    # Shifting back in time is tricky.
    if t_shift >= 0:
        hp2, hc2 = shift_waveform_phase_time(h_plus2,
                                             h_cross2,
                                             tsign * t_shift,
                                             phsign * ph_shift,
                                             verbose=verbose)
    else:
        hp2, hc2 = shift_waveform_phase_time(h_plus2,
                                             h_cross2,
                                             tsign * t_shift,
                                             phsign * ph_shift,
                                             verbose=verbose)
    #
    # Ensure both input hplus vectors are equal in length
    if len(h_plus1) > len(hp2):
        hp2.append_zeros(len(h_plus1) - len(hp2))
    elif len(h_plus1) < len(hp2):
        h_plus1.append_zeros(len(hp2) - len(h_plus1))

    if verbose:
        htilde = make_frequency_series(h_plus1)
        psd = from_string(psd_name, len(htilde), htilde.delta_f,
                          low_frequency_cutoff)
        print(("Overlap AFTER ALIGNMENT:",
               overlap_cplx(h_plus1,
                            hp2,
                            psd=psd,
                            low_frequency_cutoff=low_frequency_cutoff,
                            high_frequency_cutoff=high_frequency_cutoff,
                            normalized=True)))
        print(("Match AFTER ALIGNMENT:",
               match(h_plus1,
                     hp2,
                     psd=psd,
                     low_frequency_cutoff=low_frequency_cutoff,
                     high_frequency_cutoff=high_frequency_cutoff)))
    if verify:
        #
        print("Verifying time alignment...")
        # Determine the phase and time shifts for optimal match
        snr, corr, snr_norm = matched_filter_core(  # htilde, stilde,
            h_plus1, hp2, psd, low_frequency_cutoff, high_frequency_cutoff,
            None)
        max_snr, max_id = snr.abs_max_loc()
        print(("Post-Alignment Index of MAX SNR (should be 0 or 1 or %d): %d" %
               (len(snr) - 1, max_id)))
        print(("Length of whole SNR time-series: ", len(snr)))
        if max_id != 0 and max_id != 1 and max_id != (
                len(snr) - 1) and max_id != (len(snr) - 2):
            # raise RuntimeError( "Warning: ALIGNMENT NOT CORRECT (see above)" )
            print("Warning: ALIGNMENT NOT CORRECT (see above)")
        else:
            print("Alignment in time correct..")
        #
        print("Verifying phase alignment...")
        ph_shift = np.angle(snr[max_id])
        if ph_shift != 0:
            print("Warning: Phasing alignment possibly incorrect.")
            print(("dphi, dphi+pi, dphi-pi: ", ph_shift, ph_shift + np.pi,
                   ph_shift - np.pi))
            print(("dphi/pi, dphi*pi: ", ph_shift / np.pi, ph_shift * np.pi))
        #

    #
    if trim_trailing:
        hp1 = trim_trailing_zeros(hp1)
        hc1 = trim_trailing_zeros(hc1)
        hp2 = trim_trailing_zeros(hp2)
        hc2 = trim_trailing_zeros(hc2)
    if trim_leading:
        hp1 = trim_leading_zeros(hp1)
        hc1 = trim_leading_zeros(hc1)
        hp2 = trim_leading_zeros(hp2)
        hc2 = trim_leading_zeros(hc2)
    #
    return hplus1, hcross1, hp2, hc2
