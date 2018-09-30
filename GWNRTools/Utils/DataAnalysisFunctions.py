#!/usr/bin/env python
# Copyright (C) 2014 Prayush Kumar, Heather Fong
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
