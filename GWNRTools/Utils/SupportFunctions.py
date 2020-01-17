#!/usr/bin/env python
# Copyright (C) 2014 Prayush Kumar
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
import sys, os
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

from pycbc.types import TimeSeries, FrequencySeries, complex_same_precision_as
from pycbc.pnutils import nearest_larger_binary_number
######################################################################
__author__   = "Prayush Kumar <prayush@astro.cornell.edu>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])
verbose      = True

######################################################################
######################################################################
#
#     NR UTILITIES
#
######################################################################
######################################################################

#############################
def extrapolated_outdir_from_cce_outdir( outdir ):
  #
  # Accept SKS_d19.8-q1-sA_0_0_-0.8_sB_0_0_-0.8
  # Return BBH_SKS_d19.8_q1_sA_0_0_-0.800_sB_0_0_-0.800
  #
  #{{{
  outdir = outdir.strip('/').split('/')[-1]
  try: idtype, dq, s1x, s1y, s1z, _, s2x, s2y, s2z = outdir.split('_')
  except ValueError:
    if outdir[0] == 'd':
      outdir = 'CF_' + outdir
      idtype, dq, s1x, s1y, s1z, _, s2x, s2y, s2z = outdir.split('_')
    else: raise ValueError('Cannot translate dir name to extrapolated dir')
  if idtype == 'CF': idtype += 'MS'
  d, q, _ = dq.split('-')
  print q
  if '.' in q: q = 'q%.2f' % np.float64(q[1:])
  if np.float64( d[1:] ) == np.round(np.float64( d[1:] )):
    d = 'd' + str(int(np.float64(d[1:]) ))
  print s1z, s2z
  if np.float(s1z) == 0.: s1z = '0'
  else: s1z = '%.3f' % np.float128(s1z)
  if np.float(s2z) == 0.: s2z = '0'
  else: s2z = '%.3f' % np.float128(s2z)
  retdir = 'BBH_%s_%s_%s_sA_%s_%s_%s_sB_%s_%s_%s' % (idtype, d, q, \
              s1x, s1y, s1z, s2x, s2y, s2z)
  return retdir
  #}}}

#############################
def initial_frequency_from_metadata( id_string, lev=None, xml_table=None ):
  #{{{
  if xml_table==None: raise IOError("Please provide the catalog table")
  if lev==None: raise IOError("What Lev is the waveform..?")
  for line in xml_table:
    if id_string in line.waveform and lev in line.waveform:
      return line.f_lower
  raise IOError("Waveform not found in the catalog..! Lev missing?")
  #}}}


######################################################################
######################################################################
#
#    GENERAL Utilities
#
######################################################################
######################################################################

#############################
def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return idx, a.flat[idx]


#############################
def approx_equal(A, B, eps=1.e-4):
    d = np.abs(A-B)/(np.abs(A)+np.abs(B))
    if d < eps: return True
    else: return False

#############################
def update_progress(progress):
    print '\r\r[{0}] {1:.2%}'.format('#'*(int(progress*100)/2)+' '*(50-int(progress*100)/2),
            progress),
    if progress == 100:
        print "Done"
    sys.stdout.flush()

#############################
def nextpow2(n): return 2**int( ceil( log2( n ) ) )

#############################
def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

#############################
# Use Green's theorem to compute the area
# enclosed by a given contour.
def area_inside_contour(vs):
    x = vs[:,0]
    y = vs[:,1]
    a = 0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
    return np.abs(a)

#############################
def zero_pad_beginning( h, steps=1 ):
    h.data = np.roll( h.data, steps )
    return h

#############################
def getSec(s):
    l = s.split(':')
    return int(l[0]) * 3600 + int(l[1]) * 60 + int(l[2])

def get_time(date, time):
  date_contrib = int(date.split('/')[-1]) * 24 * 60 * 60
  time_contrib = getsec(time)
  return date_contrib + time_contrib

#############################
def trim_trailing_zeros(hp):
  for i in np.arange( len(hp)-1, 0, -1):
    if hp[i]!=0: break
  return hp[:i+1]

#############################
def trim_leading_zeros(hp):
  for i in np.arange( len(hp) ):
    if hp[i]!=0: break
  return hp[i:]

#############################
def join_list_of_strings(lt):
    out_string = ""
    for s in lt: out_string = out_string + "    " + s
    return out_string

######################################################################
######################################################################
#
#    Functions related to LAL and PyCBC datatypes
#
######################################################################
######################################################################

#############################
def convert_TimeSeries_to_lalREAL8TimeSeries( h, name=None ):
    tmp = lal.CreateREAL8Sequence( len(h) )
    tmp.data = np.array(h.data)
    hnew = lal.REAL8TimeSeries()
    hnew.data = tmp
    hnew.deltaT = h.delta_t
    hnew.epoch = h._epoch
    if name is not None: hnew.name = name
    return hnew

#############################
def convert_lalREAL8TimeSeries_to_TimeSeries( h ):
    return TimeSeries(h.data.data, delta_t=h.deltaT, \
                      copy=True, epoch=h.epoch, dtype=h.data.data.dtype)

#############################
def extend_waveform_TimeSeries(wav, filter_N):
  #{{{
  if len(wav) != filter_N:
    try:
      _wav = TimeSeries(np.zeros(filter_N), delta_t=wav.delta_t,
                        dtype=real_same_precision_as(wav), epoch=wav._epoch)
    except MemoryError as merr:
      print "Do you really want to allocate %d doubles?" % filter_N
      raise MemoryError(merr)
    _wav[:len(wav)] = wav
  else: _wav = wav
  return _wav
  #}}}

#############################
def extend_waveform_FrequencySeries(wav, filter_n, force_fit=False):
  #{{{
  if len(wav) < filter_n:
    _wav = FrequencySeries(np.zeros(filter_n), delta_f=wav.delta_f,
                        dtype=complex_same_precision_as(wav), epoch=wav._epoch)
    _wav[:len(wav)] = wav
  elif len(wav) == filter_n: _wav = wav
  elif force_fit:
    print "WARNING: Ignoring high-frequency content above %.3f Hz" % (wav.delta_f * filter_n)
    _wav = FrequencySeries(np.zeros(filter_n), delta_f=wav.delta_f,
                        dtype=complex_same_precision_as(wav), epoch=wav._epoch)
    _wav[:len(_wav)] = wav[:len(_wav)]
  else:
    raise IOError("Passed FrequencySeries has length %d, cannot shrink to %d" % (len(wav), filter_n))
  return _wav
  #}}}

def convert_numpy_to_pycbc_type(arr, out_type, sample_rate=None, time_length=None):
    """
Convert numpy.array to pycbc.types, TimeSeries and FrequencySeries.
Output array is extended to length consistent with time_length passed

ALL ARGUMENTS ARE NECESSARY.
    """
    delta_t = 1./sample_rate
    delta_f = 1./time_length
    N = sample_rate * time_length
    n = N/2 + 1
    if out_type == TimeSeries:
        out_arr = TimeSeries(arr, delta_t=delta_t)
        out_arr = extend_waveform_TimeSeries(out_arr, N)
    elif out_type == FrequencySeries:
        out_arr = FrequencySeries(arr, delta_f=delta_f)
        out_arr = extend_waveform_FrequencySeries(out_arr, n)
    return out_arr


def make_padded_frequency_series(vec, filter_N=None, delta_f=None):
    """
Convert vec (TimeSeries or FrequencySeries) to a FrequencySeries. For
a TimeSeries input, first it is resized to filter_N and  is  If
filter_N and/or delta_f are given, the output will take those values. If
    not told otherwise the code will attempt to pad a timeseries first such that
    the waveform will not wraparound. However, if delta_f is specified to be
    shorter than the waveform length then wraparound *will* be allowed.
    """
    #{{{
    if filter_N is None: filter_N = nearest_larger_binary_number(len(vec))
    filter_n = filter_N / 2 + 1

    if isinstance(vec, FrequencySeries):
        vectilde = FrequencySeries(zeros(filter_n), delta_f=vec.get_delta_f(),
            dtype=complex_same_precision_as(vec))
        cplen = min(len(vec), len(vectilde))
        vectilde[:cplen] = vec[:cplen]
        if delta_f is not None:
            delta_f_ratio = max(1, int(ceil(vectilde.get_delta_f() / delta_f)))
            vectilde = vectilde[:len(vectilde):delta_f_ratio]
    
    elif isinstance(vec, TimeSeries):
        delta_f_from_filter_N = 1. / filter_N / vec.get_delta_t()
        vec.resize(filter_N)
        v_tilde = vec.to_frequencyseries()
        vectilde = FrequencySeries(v_tilde[:], delta_f = delta_f_from_filter_N,
                    dtype = complex_same_precision_as(vec), copy = True)
    
    return vectilde
    #}}}


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
#      OVERLAP CALCULATIONS
#
######################################################################
######################################################################


#############################
def get_uniform_mass_range( m_lower, m_upper, m_sep ):
  #{{{
  mlist = [m_lower]
  for m in np.arange( np.ceil(m_lower), np.floor(m_upper), m_sep ):
    mlist.append( m )
  mlist.append( m_upper )
  return np.array( mlist )
  #}}}


#############################
def blend(hin, mm, sample, time, t_opt, WinID=-1):
    # Only dealing with real part, don't do hc calculations
    # t_opt is length-5 array describing multiples of mm
    # Returns length-5 array of TimeSeries (1 per blending)
    #{{{
    hp0, hc0 = hin.rescale_to_totalmass( mm )
    hp0._epoch = hc0._epoch = 0
    amp = TimeSeries(np.sqrt(hp0**2 + hc0**2), copy=True, delta_t=hp0.delta_t)
    max_a, max_a_index = amp.abs_max_loc()
    print "\n\n In blend:\nTotal Mass = %f, len(hp0,hc0) = %d, %d = %f s" %\
          (mm, len(hp0), len(hc0), hp0.sample_times[-1]-hp0.sample_times[0])
    print "Waveform max = %e, located at %d" % (max_a, max_a_index)
    #amp_after_peak = amp
    #amp_after_peak[:max_a_index] = 0
    mtsun = lal.MTSUN_SI
    amp_after_peak = amp[max_a_index:]
    iA, vA = min(enumerate(amp_after_peak),key=lambda x:abs(x[1]-0.01*max_a))
    iA += max_a_index
    #iA, vA = min(enumerate(amp_after_peak),key=lambda x:abs(x[1]-0.01*max_a))
    iB, vB = min(enumerate(amp_after_peak),key=lambda x:abs(x[1]-0.1*max_a))
    iB += max_a_index
    if iA <= max_a_index:
      print >>sys.stdout,"iA = %d, iB = %d, vA = %e, vB = %e" % (iA,iB,vA,vB)
      sys.stdout.flush()
      raise RuntimeError("Couldnt find amplitude threshold time iA")
      # do something
      #fout = open('hpdump.dat','w+')
      #for i in range( len(amp) ):
      #  if i > max_a_index and amp[i] == 0: break
      #  fout.write('%e\t%e\n' % (amp.sample_times[i],amp[i]))
      #fout.close()
      # Find the point the hard way
      target_amp = max_a * 0.01
      tmp_data = amp.data
      for idx in range( max_a_index, len(amp) ):
        if tmp_data[idx] < target_amp: break
      iA = idx
      print "Newfound iA = %d" % iA
      # Yet another way
      amp_after_peak = amp[max_a_index:]
      iA, vA = min(enumerate(amp_after_peak),key=lambda x:abs(x[1]-0.01*max_a))
      iA += max_a_index
      print "Newfound iA another way = %d" % iA
      raise RuntimeError("Had to find amplitude threshold the hard way")
    if iB <= max_a_index:
      raise RuntimeError("Couldnt find amplitude threshold time iB")
      # this doesn't happen yet
      pass
    print "NEW: iA = %d, iB = %d, vA = %e, vB = %e" % (iA, iB, vA, vB)
    t = [ [ t_opt[0]*mm,500*mm,hp0.sample_times.data[iA]/mtsun,hp0.sample_times.data[iA]/mtsun+t_opt[3]*mm], # Prayush's E
          [ t_opt[0]*mm,t_opt[1]*mm,hp0.sample_times.data[iA]/mtsun,hp0.sample_times.data[iA]/mtsun+t_opt[3]*mm ],
          [ t_opt[0]*mm,t_opt[1]*mm,hp0.sample_times.data[iB]/mtsun,hp0.sample_times.data[iB]/mtsun+t_opt[4]*mm ],
          [ t_opt[0]*mm,t_opt[2]*mm,hp0.sample_times.data[iA]/mtsun,hp0.sample_times.data[iA]/mtsun+t_opt[3]*mm ],
          [ t_opt[0]*mm,t_opt[2]*mm,hp0.sample_times.data[iB]/mtsun,hp0.sample_times.data[iB]/mtsun+t_opt[4]*mm ] ]
    hphc = []
    hphc.append(hp0)
    for i in range(len(t)):
      if (WinID >= 0 and WinID < len(t)) and i != WinID: continue
      print "Testing window with t = ", t[i]
      hphc.append(hin.blending_function(hp0=hp0,t=t[i],sample_rate=sample,time_length=time))
    print "No of blending windows being tested = %d" % (len(hphc)-1)
    return hphc
    #}}}

#############################
def blendTimeSeries(hp0, hc0, mm, sample, time, t_opt):
    """
    [DEPRECATED - use "blend"]
    Only dealing with real part, don't do hc calculations
    t_opt is length-5 array describing multiples of mm
    Returns length-5 array of TimeSeries (1 per blending)
    """
    #{{{
    from UseNRinDA import nr_waveform
    nrtool = nr_waveform()
    amp = TimeSeries(np.sqrt(hp0**2 + hc0**2), copy=True, delta_t=hp0.delta_t)
    max_a, max_a_index = amp.abs_max_loc()
    print "Waveform max = %e, located at %d" % (max_a, max_a_index)
    amp_after_peak = amp
    amp_after_peak[:max_a_index] = 0
    mtsun = lal.MTSUN_SI
    iA, vA = min(enumerate(amp_after_peak),key=lambda x:abs(x[1]-0.01*max_a))
    iB, vB = min(enumerate(amp_after_peak),key=lambda x:abs(x[1]-0.1*max_a))
    print iA, iB
    t = [ [ t_opt[0]*mm,500*mm,hp0.sample_times.data[iA]/mtsun,hp0.sample_times.data[iA]/mtsun+t_opt[3]*mm], # Prayush's E
          [ t_opt[0]*mm,t_opt[1]*mm,hp0.sample_times.data[iA]/mtsun,hp0.sample_times.data[iA]/mtsun+t_opt[3]*mm ],
          [ t_opt[0]*mm,t_opt[1]*mm,hp0.sample_times.data[iB]/mtsun,hp0.sample_times.data[iB]/mtsun+t_opt[4]*mm ],
          [ t_opt[0]*mm,t_opt[2]*mm,hp0.sample_times.data[iA]/mtsun,hp0.sample_times.data[iA]/mtsun+t_opt[3]*mm ],
          [ t_opt[0]*mm,t_opt[2]*mm,hp0.sample_times.data[iB]/mtsun,hp0.sample_times.data[iB]/mtsun+t_opt[4]*mm ] ]
    hphc = []
    #hphc.append(hp0)
    for i in range(len(t)):
      print t[i]
      hphc.append(nrtool.blending_function_Tukey(hp0=hp0,t=t[i],\
                          sample_rate=sample,time_length=time))
    print "No of blending windows being tested = %d" % len(hphc)
    return hphc
    #}}}

#############################
def planck_window( N=None, eps=None, one_sided=True, winstart=0 ):
  #{{{
  if N is None or eps is None:
    raise IOError("Please provide the window length and smoothness")
  #
  N = N - winstart
  win = ones(N)
  N1 = int(eps * (N - 1.)) + 1
  den_t1_Zp = 1. + 2. * win / (N - 1.)
  Zp = 2. * eps * (1./den_t1_Zp + 1./(den_t1_Zp - 2.*eps))
  win[0:N1] = array(1. / (exp(Zp) + 1.))[0:N1]
  ##
  if one_sided is not True:
    N2 = (1. - eps) * (N - 1.) + 1
    den_t1_Zm = 1. - 2. * win / (N - 1.)
    Zm = 2. * eps * (1./den_t1_Zm + 1./(den_t1_Zm - 2.*eps))
    win[N2:] = array(1. / (exp(Zm) + 1.))[N2:]
  ##
  win = append( ones(winstart), win )
  return win
  #}}}

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
