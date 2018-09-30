#!/usr/bin/env python
# Copyright (C) 2014 Prayush Kumar
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

from matplotlib import use
use('Agg')
import sys, os
import matplotlib as plt # FIXME

import h5py
import glob
import commands as cmd
import time
from numpy import *
import numpy as np
import scipy as sp
import math

from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
from scipy.optimize import minimize_scalar

from pycbc.pnutils import nearest_larger_binary_number

######################################################################
__author__   = "Prayush Kumar <prayush@astro.cornell.edu>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])
verbose      = True


######################################################################
######################################################################
#
#     Waveform Smoothing
#
######################################################################
######################################################################

#############################
def smooth(x,window_len=11,window='flat'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

#############################
def moving_window_average(x, i, window_len=10):
    """
    Return the average of x[i - window_len/2 : i + window_len/2]
    """
    imin = np.maximum(0, i - window_len/2)
    imax = np.minimum(len(x), i + window_len/2)
    return np.mean( x[imin:imax] )



#############################
def windowing_tanh(waveform_array, bin_to_center_window, sharpness):
  #{{{
  waveform_array = asarray(waveform_array)
  length_of_waveform = size(waveform_array)
  x = arange(length_of_waveform)
  window_function = (tanh(sharpness * (x - bin_to_center_window)) + 1.)/2.
  temp = window_function*waveform_array
  return temp
  #}}}

######################################################################
######################################################################
#
#     Functions to align waveforms
#
######################################################################
######################################################################

#############################
def get_time_at_frequency_from_polarizations(hp, hc, fvalue):
  fr = frequency_from_polarizations(hp, hc)
  obj_func = np.abs(np.abs(fr) - fvalue)
  id_start = np.where(obj_func  ==  np.min(obj_func))[0][0]
  for idx in range(id_start, len(fr)):
    if fr[idx] > 2*fvalue and fr[idx+1] > 2*fvalue: break
  #
  frI = InterpolatedUnivariateSpline( fr.sample_times, obj_func )
  tmp = minimize_scalar(frI, fr.sample_times[id_start],
                      method='bounded',
                      bounds=(fr.sample_times[id_start], fr.sample_times[idx]))
  return tmp['x']

#############################
def get_time_at_frequency(fr, fvalue):
    return get_time_at_y(fr, fvalue)

#############################
def get_time_at_y(fr, fvalue):
    """
Finds the closest match to `fvalue` in a TimeSeries.
Input a TimeSeries with epoch set correctly.
    """
    ## Define time interval to be searched
    idx_first = int(len(fr) * 0.2) # 20% margin for junk - TOO MUCH?
    idx_end = np.where(np.abs(fr.sample_times.data) == np.abs(fr.sample_times.data).min())[0][0] # Assume a properly aligned TimeSeries
    ## Starting guess
    obj_func = np.abs(np.abs(fr) - fvalue)[idx_first:idx_end]
    id_start = np.where(obj_func  ==  np.min(obj_func))[0][int(np.ceil(len(np.where(obj_func  ==  np.min(obj_func))[0])/2))]
    ## Interpolate and find
    frI = InterpolatedUnivariateSpline( fr.sample_times[idx_first:idx_end], obj_func )
    tmp = minimize_scalar(frI, fr.sample_times[id_start],
                      method='bounded',
                      bounds=(fr.sample_times[idx_first], fr.sample_times[idx_end]))
    ## Return time value
    return tmp['x']

#############################
def shift_waveform_phase_time(hp, hc, t_shift, ph_shift,
                        trim_leading=False, trim_trailing=True,
                        verbose=False):
  """
  Input:  hp, hc, where h = hp(t) + i hx(t) = Amp(t) * exp(-i * Phi(t))
  Output: hp, hc, where h = Amp(t - t_c) * exp( -i * [Phi(t - t_c) + phi_c] )
  """
  hpnew = TimeSeries(hp, epoch=hp._epoch, delta_t=hp.delta_t, dtype=hp.dtype,
                      copy=True)
  hcnew = TimeSeries(hc, epoch=hc._epoch, delta_t=hc.delta_t, dtype=hc.dtype,
                      copy=True)
  # First apply phase shift
  if ph_shift != 0.:
    amplitude = amplitude_from_polarizations(hpnew, hcnew)
    phase = phase_from_polarizations(hpnew, hcnew)
    if verbose: print "shifting by %f radians" % ph_shift
    phase = phase + ph_shift
    hpnew = TimeSeries(amplitude * np.cos(phase + np.pi), \
                  epoch=hpnew._epoch, delta_t=hpnew.delta_t, dtype=hpnew.dtype)
    hcnew = TimeSeries(amplitude * np.sin(phase + np.pi), \
                  epoch=hcnew._epoch, delta_t=hcnew.delta_t, dtype=hcnew.dtype)
  # Now apply time shift
  if t_shift != 0:
    id_shift = int(np.round(np.abs(t_shift) / hpnew.delta_t))
    if verbose: print "shifting by %d (%f)" % (id_shift, t_shift)
    if t_shift > 0:
      hpnew.append_zeros(id_shift)
      hcnew.append_zeros(id_shift)
    else:
      hpnew.prepend_zeros(id_shift)
      hcnew.prepend_zeros(id_shift)
    hpnew.roll(id_shift * np.sign(t_shift))
    hcnew.roll(id_shift * np.sign(t_shift))
  if trim_trailing:
    hpnew = trim_trailing_zeros(hpnew)
    hcnew = trim_trailing_zeros(hcnew)
  if trim_leading:
    hpnew = trim_leading_zeros(hpnew)
    hcnew = trim_leading_zeros(hcnew)
  # RETURN
  return hpnew, hcnew


#############################
def align_waveforms_amplitude_peak(hplus1, hcross1, hplus2, hcross2,
                        trim_leading=False, trim_trailing=True,
                        verbose=False):
  """
  Align the two waveforms, shifting only one of the two.
      - AT the Amplitude PEAK
  """
  hp1, hc1 = TimeSeries(hplus1), TimeSeries(hcross1)
  hp2, hc2 = TimeSeries(hplus2), TimeSeries(hcross2)
  amp1 = amplitude_from_polarizations(hp1, hc1)
  amp2 = amplitude_from_polarizations(hp2, hc2)
  # Get amplitude peaks
  amp1I = InterpolatedUnivariateSpline(amp1.sample_times, -1 * amp1.data)
  x0   = np.float64(np.where(amp1.data==max(amp1.data))[0][0]*amp1.delta_t +\
                                amp1._epoch)
  tmp = minimize_scalar( amp1I, x0, method='bounded',\
                          bounds=(x0-10*amp1.delta_t, x0+10*amp1.delta_t) )
  h1_max_amp_time = tmp['x']
  h1_max_amp = -1 * tmp['fun']
  amp2I = InterpolatedUnivariateSpline(amp2.sample_times, -1 * amp2.data)
  x0   = np.float64(np.where(amp2.data==max(amp2.data))[0][0]*amp2.delta_t +\
                                amp2._epoch)
  tmp = minimize_scalar( amp2I, x0, method='bounded',\
                          bounds=(x0-10*amp2.delta_t, x0+10*amp2.delta_t) )
  h2_max_amp_time = tmp['x']
  h2_max_amp = -1 * tmp['fun']
  if verbose:
    print "h1 max time = %f, epoch = %f" % (h1_max_amp_time, float(hp1._epoch))
    print "h2 max time = %f, epoch = %f" % (h2_max_amp_time, float(hp2._epoch))

  # Amplitude location from the start
  t1 = h1_max_amp_time
  t2 = h2_max_amp_time
  t_shift = t1 - t2
  if verbose: print "time shift = %f to be added to waveform 2" % t_shift
  #
  # Find phase shift
  #
  phs1 = phase_from_polarizations(hp1, hc1)
  phs2 = phase_from_polarizations(hp2, hc2)
  phs1I = InterpolatedUnivariateSpline(phs1.sample_times, phs1.data)
  phs2I = InterpolatedUnivariateSpline(phs2.sample_times, phs2.data)

  ph1 = phs1I(h1_max_amp_time)
  ph2 = phs2I(h2_max_amp_time)
  ph_shift = np.float64(ph1 - ph2)

  if verbose:
    print "phase1 at peak idx = %d, = %f" % (int(np.round(t1 / hp1.delta_t)), ph1)
    print "phase2 at peak idx = %d, = %f" % (int(np.round(t2 / hp2.delta_t)), ph2)
    print "phase shift = %f, time shift = %f" % (ph_shift, t_shift)
  #
  # Shift whichever needs to be shifted to future time.
  # Shifting back in time is tricky.
  if t_shift >= 0:
    hp2, hc2 = shift_waveform_phase_time(hp2, hc2, t_shift, ph_shift, verbose=verbose )
  else:
    hp2, hc2 = shift_waveform_phase_time(hp2, hc2, t_shift, ph_shift, verbose=verbose )
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


#############################
def align_waveforms_at_frequency(hplus1, hcross1, hplus2, hcross2, falign,
      trim_leading=False, trim_trailing=True, verbose=False):
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
  obj_func = np.abs(np.abs(freq1.data)-falign)
  f1I = InterpolatedUnivariateSpline(freq1.sample_times, obj_func)
  id_start = np.where(obj_func  ==  np.min(obj_func))[0][0]
  for idx in range(id_start, len(freq1)):
    if freq1[idx] > 2*falign and freq1[idx+1] > 2*falign:
      break
  tmp = minimize_scalar(f1I, freq1.sample_times[id_start],
                method='bounded',
                bounds=(freq1.sample_times[id_start], freq1.sample_times[idx]))
  f1_align_time = tmp['x']
  #
  # Get time at flign for wave 2
  #
  freq2 = frequency_from_polarizations(hp2, hc2)
  obj_func = np.abs(np.abs(freq2.data)-falign)
  f2I = InterpolatedUnivariateSpline(freq2.sample_times, obj_func)
  id_start = np.where(obj_func  ==  np.min(obj_func))[0][0]
  for idx in range(id_start, len(freq2)):
    if freq2[idx] > 2*falign and freq2[idx+1] > 2*falign:
      break
  tmp = minimize_scalar(f2I, freq2.sample_times[id_start],
                method='bounded',
                bounds=(freq2.sample_times[id_start], freq2.sample_times[idx]))
  f2_align_time = tmp['x']
  #
  t1 = f1_align_time
  t2 = f2_align_time
  t_shift = t1 - t2
  if verbose: print "time shift = %f to be added to waveform 2" % t_shift
  #
  # Find phase shift at the time
  #
  phs1 = phase_from_polarizations(hp1, hc1)
  phs2 = phase_from_polarizations(hp2, hc2)
  phs1I = InterpolatedUnivariateSpline(phs1.sample_times, phs1.data)
  phs2I = InterpolatedUnivariateSpline(phs2.sample_times, phs2.data)

  ph1 = phs1I(f1_align_time)
  ph2 = phs2I(f2_align_time)
  ph_shift = (ph1 - ph2) * 1
  if verbose:
    print "time @ f1 = %f : %f" % (falign, f1_align_time)
    print "time @ f2 = %f : %f" % (falign, f2_align_time)
    print "ph1 = %f, ph2 = %f" % (ph1, ph2)
    print "phase shift = %f, time shift = %f" % (ph_shift, t_shift)
    print "type of ph_shift, t_shift: ", type(ph_shift), type(t_shift)
  #
  # Shift whichever needs to be shifted to future time.
  # Shifting back in time is tricky.
  hp2, hc2 = shift_waveform_phase_time(hp2, hc2, t_shift, ph_shift, verbose=verbose )
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


#############################
def align_waveforms_optimally(hplus1, hcross1, hplus2, hcross2,
                    psd='aLIGOZeroDetHighPower',\
                    low_frequency_cutoff=None,
                    high_frequency_cutoff=None,\
                    tsign=1, phsign=-1,
                    verify=True, phase_tolerance=1e-3, overlap_tolerance=1e-3,
                    trim_leading=False, trim_trailing=False,
                    verbose=False):
  """
  Align waveforms such that their inner product (noise weighted) is optimal
  without requiring any phase or time shift.

  The appropriate time and phase shifts are determined iteratively and applied
  to the second set of (hplus, hcross) vectors.
  """
  #############################################################################
  ## First copy over data into local memory, ensure lengths of time and
  ## frequency domain vectors are consistent, and compute the maximized overlap
  #
  # 1) Cast into time-series
  h_plus1  = TimeSeries(hplus1, epoch=hplus1._epoch, delta_t=hplus1.delta_t,\
                                  dtype=hplus1.dtype, copy=True)
  h_cross1 = TimeSeries(hcross1,epoch=hplus1._epoch, delta_t=hplus1.delta_t,\
                                  dtype=hplus1.dtype, copy=True)
  h_plus2  = TimeSeries(hplus2, epoch=hplus2._epoch, delta_t=hplus2.delta_t,\
                                  dtype=hplus2.dtype, copy=True)
  h_cross2 = TimeSeries(hcross2,epoch=hplus2._epoch, delta_t=hplus2.delta_t,\
                                  dtype=hplus2.dtype, copy=True)
  #
  # 2) Ensure both input hplus vectors are equal in length
  if len(hplus2) > len(hplus1):
    h_plus1.append_zeros( len(hplus2)-len(hplus1) )
    h_cross1.append_zeros( len(hplus2)-len(hplus1) )
  elif len(hplus2) < len(hplus1):
    h_plus2.append_zeros( len(hplus1)-len(hplus2) )
    h_cross2.append_zeros( len(hplus1)-len(hplus2) )
  #
  # 3) Set the upper frequency cutoff to Nyquist if not set by User
  if high_frequency_cutoff == None:
    high_frequency_cutoff = 1./h_plus1.delta_t / 2.
  #
  # 4) Compute LIGO noise psd
  if psd == None:
    raise IOError("Need compatible psd [or name] as input!")
  elif type(psd)==str:
    htilde = make_frequency_series(h_plus1)
    psd_name = psd
    psd = from_string(psd_name, len(htilde), htilde.delta_f, low_frequency_cutoff)
  ##
  # 5) Calculate Overlap (maximized) before alignment
  m = match(h_plus1, h_plus2, psd=psd,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff)
  optimal_overlap = m[0] # FIXME
  if verbose:
    print "Overlap BEFORE ALIGNMENT:", \
        overlap_cplx(h_plus1, h_plus2, psd=psd,
                    low_frequency_cutoff=low_frequency_cutoff,
                    high_frequency_cutoff=high_frequency_cutoff,
                    normalized=True)
    print "Match BEFORE ALIGNMENT:", m
  #############################################################################
  ## Iterate to obtain the correct phase and time shifts, using which we
  ## align the two waveforms such that their unmaximized and maximized overlaps
  ## agree.

  #
  # 1) Initialize phase/time offset counters
  t_shift_counter = 0
  ph_shift_counter= 0
  #
  # 2) Initialize initial garbage values to enter the while loop
  idx = 0
  ph_shift = t_shift = 1e9
  olap = 0 + 0j
  #
  # 3) Iteration begins
  ## >>>>>>
  while np.abs(ph_shift) > phase_tolerance or \
        np.abs(t_shift) > h_plus1.delta_t or \
        np.abs(np.abs(olap.real) - optimal_overlap) > overlap_tolerance:
    if idx == 0: hp2, hc2 = h_plus2, h_cross2
    #
    # 1) Determine the phase and time shifts for optimal match
    #    by comparing hplus1/hcross1 with hp2/hc2 which is phase/time shifted
    #    in previous iteration
    snr, corr, snr_norm = \
        matched_filter_core( h_plus1, hp2,
                        psd, low_frequency_cutoff, high_frequency_cutoff, None)
    max_snr, max_id = snr.abs_max_loc()

    if max_id != 0: t_shift  = snr.delta_t * ( len(snr) - max_id )
    else: t_shift = snr.delta_t * max_id

    ph_shift = np.angle(snr[max_id])

    #
    # 2) Add them to running time/phase offset counter
    t_shift_counter += t_shift
    ph_shift_counter+= ph_shift
    #
    if verbose:
      print " >> Iteration %d\n" % (idx+1)
      print "max_id = %d, id_shift = %d" % (max_id, int(t_shift / snr.delta_t))
      print "t_shift = %f,\n ph_shift = %f" % (t_shift, ph_shift)
    #
    ####
    ## 3) Shift the second hp/hc pair (ORIGINAL) by cumulative phase/time offset
    hp2, hc2 = shift_waveform_phase_time(h_plus2, h_cross2,
                                        tsign * t_shift_counter,
                                        phsign * ph_shift_counter,
                                        verbose=verbose)
    #
    ###
    ## 4) As time shifting can change array lengths, equalize again, compute psd
    ##
    if len(h_plus1) > len(hp2):
      hp2.append_zeros( len(h_plus1)-len(hp2) )
      htilde = make_frequency_series(h_plus1)
      psd = from_string(psd_name, len(htilde), htilde.delta_f, low_frequency_cutoff)
    elif len(h_plus1) < len(hp2):
      h_plus1.append_zeros( len(hp2)-len(h_plus1) )
      htilde = make_frequency_series(h_plus1)
      psd = from_string(psd_name, len(htilde), htilde.delta_f, low_frequency_cutoff)
    #
    # 5) Compute UNMAXIMIZED overlap.
    olap = overlap_cplx(h_plus1, hp2, psd=psd,
                      low_frequency_cutoff=low_frequency_cutoff,
                      high_frequency_cutoff=high_frequency_cutoff,
                      normalized=True)
    if verbose:
      print "Overlap AFTER ALIGNMENT = ", olap
      print "Optimal Overlap = ", optimal_overlap
    #
    idx += 1
    if verbose: print "\n"
  ## >>>>>>
  # 3) Iteration ended.

  #############################################################################
  ## Verify the alignment
  ###
  if verify:
    #
    print "Verifying time alignment..."
    #
    # 1) Determine the phase and time shifts for optimal match
    snr, corr, snr_norm = \
      matched_filter_core( h_plus1, hp2,
                        psd, low_frequency_cutoff, high_frequency_cutoff, None)
    max_snr, max_id = snr.abs_max_loc()
    if verbose:
      print "Post-Alignment Index of MAX SNR (should be 0 or 1 or %d): %d" %\
                (len(snr)-1, max_id)
      print "Length of whole SNR time-series: ", len(snr)
    #
    # 2) Test if current time shift is within tolerance
    if max_id !=0 and max_id !=1 and \
        max_id != (len(snr)-1) and max_id != (len(snr)-2):
      raise RuntimeError( "Warning: ALIGNMENT NOT CORRECT (see above)" )
    else:
      print "Alignment in time correct.."
    #
    # 3) Test if current phase shift is within tolerance
    print "Verifying phase alignment..."
    ph_shift = np.angle(snr[max_id])
    if np.abs(ph_shift) > phase_tolerance:
      if verbose:
        print "dphi, dphi+pi, dphi-pi: ", ph_shift, ph_shift + np.pi, ph_shift - np.pi
        print "dphi/pi, dphi*pi: ", ph_shift / np.pi, ph_shift * np.pi
      raise RuntimeError( "Warning: Phasing alignment possibly incorrect." )
    else:
      if verbose:
        print "Post-Alignmend Phase shift (should be < %.2e): %.2e" %\
         (phase_tolerance, np.abs(ph_shift))
      print "Alignment in phasing correct.. (within tol %.2e)" % phase_tolerance
    #

  #############################################################################
  ## TRIM the output arrays and return
  if trim_trailing:
    hp2 = trim_trailing_zeros(hp2)
    hc2 = trim_trailing_zeros(hc2)
  if trim_leading:
    hp2 = trim_leading_zeros(hp2)
    hc2 = trim_leading_zeros(hc2)
  #
  return hplus1, hcross1, hp2, hc2






######################################################################
######################################################################
#
#       DEPRECATED FUNCTIONS
#
######################################################################
######################################################################

#############################
def align_waveforms_suboptimally(hplus1, hcross1, hplus2, hcross2,
                    psd='aLIGOZeroDetHighPower',\
                    low_frequency_cutoff=None,
                    high_frequency_cutoff=None,\
                    tsign=1, phsign=1,
                    verify=True,
                    trim_leading=False, trim_trailing=False,
                    verbose=False):
  # Cast into time-series
  h_plus1  = TimeSeries(hplus1, epoch=hplus1._epoch, delta_t=hplus1.delta_t,\
                                  dtype=hplus1.dtype)
  h_cross1 = TimeSeries(hcross1,epoch=hplus1._epoch, delta_t=hplus1.delta_t,\
                                  dtype=hplus1.dtype)
  h_plus2  = TimeSeries(hplus2, epoch=hplus2._epoch, delta_t=hplus2.delta_t,\
                                  dtype=hplus2.dtype)
  h_cross2 = TimeSeries(hcross2,epoch=hplus2._epoch, delta_t=hplus2.delta_t,\
                                  dtype=hplus2.dtype)
  #
  # Ensure both input hplus vectors are equal in length
  if len(hplus2) > len(hplus1):
    h_plus1.append_zeros( len(hplus2)-len(hplus1) )
    h_cross1.append_zeros( len(hplus2)-len(hplus1) )
  elif len(hplus2) < len(hplus1):
    h_plus2.append_zeros( len(hplus1)-len(hplus2) )
    h_cross2.append_zeros( len(hplus1)-len(hplus2) )
  #
  htilde = make_frequency_series(h_plus1)
  stilde = make_frequency_series(h_plus2)
  #
  if high_frequency_cutoff == None:
    high_frequency_cutoff = 1./h_plus1.delta_t / 2.
  #
  if psd == None:
    raise IOError("Need compatible psd [or name] as input!")
  elif type(psd)==str:
    psd_name = psd
    psd = from_string(psd_name, len(htilde), htilde.delta_f, low_frequency_cutoff)
  #
  # Determine the phase and time shifts for optimal match
  snr, corr, snr_norm = \
      matched_filter_core( htilde, stilde,
                        #h_plus1, h_plus2,
                        psd, low_frequency_cutoff, high_frequency_cutoff, None)
  max_snr, max_id = snr.abs_max_loc()

  if max_id != 0: t_shift  = snr.delta_t * ( len(snr) - max_id )
  else: t_shift = snr.delta_t * max_id

  ph_shift = np.angle(snr[max_id]) -0.24850315030-0.0465881735639
  #
  if verbose:
    print "max_id = %d, id_shift = %d" % (max_id, int(t_shift / snr.delta_t))
    print "t_shift = %f,\n ph_shift = %f" % (t_shift, ph_shift)
  #
  # print OVERLAPS
  if verbose:
    print "Overlap BEFORE ALIGNMENT:", \
        overlap_cplx(h_plus1, h_plus2, psd=psd,
                    low_frequency_cutoff=low_frequency_cutoff,
                    high_frequency_cutoff=high_frequency_cutoff,
                    normalized=True)
    print "Match BEFORE ALIGNMENT:", \
        match(h_plus1, h_plus2, psd=psd,
                    low_frequency_cutoff=low_frequency_cutoff,
                    high_frequency_cutoff=high_frequency_cutoff)

  # Shift whichever needs to be shifted to future time.
  # Shifting back in time is tricky.
  if t_shift >= 0:
    hp2, hc2 = shift_waveform_phase_time(h_plus2, h_cross2,
                                        tsign * t_shift,
                                        phsign * ph_shift,
                                        verbose=verbose)
  else:
    hp2, hc2 = shift_waveform_phase_time(h_plus2, h_cross2,
                                        tsign * t_shift,
                                        phsign * ph_shift,
                                        verbose=verbose)
  #
  # Ensure both input hplus vectors are equal in length
  if len(h_plus1) > len(hp2):
    hp2.append_zeros( len(h_plus1)-len(hp2) )
  elif len(h_plus1) < len(hp2): h_plus1.append_zeros( len(hp2)-len(h_plus1) )

  if verbose:
    htilde = make_frequency_series(h_plus1)
    psd = from_string(psd_name, len(htilde), htilde.delta_f, low_frequency_cutoff)
    print "Overlap AFTER ALIGNMENT:", \
        overlap_cplx(h_plus1, hp2, psd=psd,
                    low_frequency_cutoff=low_frequency_cutoff,
                    high_frequency_cutoff=high_frequency_cutoff,
                    normalized=True)
    print "Match AFTER ALIGNMENT:", \
        match(h_plus1, hp2, psd=psd,
                    low_frequency_cutoff=low_frequency_cutoff,
                    high_frequency_cutoff=high_frequency_cutoff)
  if verify:
    #
    print "Verifying time alignment..."
    # Determine the phase and time shifts for optimal match
    snr, corr, snr_norm = \
      matched_filter_core( #htilde, stilde,
                        h_plus1, hp2,
                        psd, low_frequency_cutoff, high_frequency_cutoff, None)
    max_snr, max_id = snr.abs_max_loc()
    print "Post-Alignment Index of MAX SNR (should be 0 or 1 or %d): %d" %\
                (len(snr)-1, max_id)
    print "Length of whole SNR time-series: ", len(snr)
    if max_id !=0 and max_id !=1 and max_id != (len(snr)-1) and max_id != (len(snr)-2):
      #raise RuntimeError( "Warning: ALIGNMENT NOT CORRECT (see above)" )
      print "Warning: ALIGNMENT NOT CORRECT (see above)"
    else:
      print "Alignment in time correct.."
    #
    print "Verifying phase alignment..."
    ph_shift = np.angle(snr[max_id])
    if ph_shift != 0:
      print "Warning: Phasing alignment possibly incorrect."
      print "dphi, dphi+pi, dphi-pi: ", ph_shift, ph_shift + np.pi, ph_shift - np.pi
      print "dphi/pi, dphi*pi: ", ph_shift / np.pi, ph_shift * np.pi
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
