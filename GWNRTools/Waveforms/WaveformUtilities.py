#!/usr/bin/env python

########################################
## IMPORTS
########################################
import time
import sys, os

import glob, commands as cmd
from math import pow
import re
import h5py
import numpy as np
from numpy import loadtxt,complex64,float32, any, isnan, isinf

from pyswarm import pso
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize, minimize_scalar
from scipy.interpolate import interp1d

os.environ['LD_LIBRARY_PATH'] =\
    '/home/prayush/research/Eccentric_IMRGPR/Code/MergerRingdownModel/C_implementation/bin/'

from optparse import OptionParser

from glue.ligolw import utils as ligolw_utils
from glue.ligolw import table, lsctables, ligolw

from pycbc.pnutils import *
from pycbc.detector import overhead_antenna_pattern as generate_fplus_fcross
from pycbc.waveform import get_td_waveform, get_fd_waveform, td_approximants, fd_approximants
from pycbc.waveform import amplitude_from_polarizations, phase_from_polarizations
from pycbc import DYN_RANGE_FAC
from pycbc.types import FrequencySeries, TimeSeries, zeros, real_same_precision_as, complex_same_precision_as
from pycbc.filter import match, overlap, sigma, overlap_cplx, make_frequency_series
from pycbc.scheme import CPUScheme, CUDAScheme
from pycbc.fft import fft
import pycbc.psd

class ContentHandler(ligolw.LIGOLWContentHandler):
    pass
lsctables.use_in(ContentHandler)

_starting_time = time.time()
########################################
## Other FUNCTIONS
########################################
def get_detector_response(ra, dec, psi, detector_tag, gmst=0):
    detMap =  { 'H1': lal.LALDetectorIndexLHODIFF, \
                'H2': lal.LALDetectorIndexLHODIFF, \
                'L1': lal.LALDetectorIndexLLODIFF, \
                'G1': lal.LALDetectorIndexGEO600DIFF, \
                'V1': lal.LALDetectorIndexVIRGODIFF, \
                'T1': lal.LALDetectorIndexTAMA300DIFF, \
                'AL1': lal.LALDetectorIndexLLODIFF, \
                'AH1': lal.LALDetectorIndexLHODIFF, \
                'AV1': lal.LALDetectorIndexVIRGODIFF
              }
    detector=detMap[detector_tag]
    # get detector
    detval = lal.CachedDetectors[detector]
    # get its response Tensor
    response = detval.response
    # get plus and cross polarization response
    return lal.ComputeDetAMResponse(response, ra, dec, psi, gmst)


def generate_detector_strain(template_params, h_plus, h_cross):
    #{{{
    latitude = 0
    longitude = 0
    polarization = 0

    if hasattr(template_params, 'latitude'):
        latitude = template_params.latitude
    else:
        latitude = template_params['latitude']
    if hasattr(template_params, 'longitude'):
        longitude = template_params.longitude
    else:
        longitude = template_params['longitude']
    if hasattr(template_params, 'polarization'):
        polarization = template_params.polarization
    else:
        polarization = template_params['polarization']

    f_plus, f_cross = generate_fplus_fcross(longitude, latitude, polarization)

    return h_plus * f_plus + h_cross * f_cross
    #}}}


def get_ncycles_to_merger(hp, hc):
    if type(hp) == FrequencySeries: return -1
    a = amplitude_from_polarizations(hp, hc)
    p = phase_from_polarizations(hp, hc)
    idx = a.abs_max_loc()[-1]
    ncyc = np.abs(p[idx] - p[0]) / np.pi / 2.0
    return ncyc

########################################
## USER-FACING FUNCTIONS
########################################
def align_curves(x1, y1, x2, y2,
                 delta_x = 1.0,
                 x_low_lim=None,
                 x_high_lim=None,
                 offset_low_lim=None,
                 offset_high_lim=None,
                 num_retries=1,
                 verbose=False,
                 debug=False):
    """
This function maximizes the alignment between two 1D functions, varying the
x coordinate alone.

[Goal]
To minimize:
    f(x_offset) := \int_{x_low_lim}^{x_high_lim} |y2(x + x_offset) - y1(x)| dx
over x_offset.

[Notes
1) [x_low_lim, x_high_lim] are with respect to the (x1, y1) pair.
   The other pair (x1, y2) is the one effectively shifted.
2) Not specifying [x_low_lim, x_high_lim] is equivalent to integrating
   the mean-square difference over the complete (x2) vector.
    """
    ### {{{
    ### 4) DEFINE AN OBJECTIVE FUNCTION FOR PSO TO MINIMIZE
    def objective_function_alignment(x, *args):
        objective_function_alignment.counter += 1
        # Get offset for this iteration
        x_offset = x
        # Get original x1 and x2 values, as well as splines for y1(x2) and y2(x2)
        x1, y1, x2tmp, y2, x_low_lim, x_high_lim = args
        x2                = x2tmp + x_offset
        s                 = 0
        for idx1, x1val in enumerate(x1):
            if x1val < x_low_lim or x1val > x_high_lim: continue
            idx2, x2val = find_nearest(x2, x1val)
            if np.abs(x2val - x1val) > delta_x:
                if objective_function_alignment.counter % 100 == 0 and verbose:
                    print np.abs(x2val - x1val)
                #raise RuntimeError("""Cannot solve the problem without either a) INcreasing delta_x, or b) interpolation. Vectors are not sampled finely enough.""")
            s += (y2[idx2] - y1[idx1]) ** 2
        #s = s ** 0.5
        if objective_function_alignment.counter % 100 == 0:
            print "Objective function for offset = %.3f is %.6f" % (x, s)
        return s
    objective_function_alignment.counter = 0
    ###
    if x_low_lim is None: x_low_lim = np.min(x1)
    if x_high_lim is None: x_high_lim = np.max(x1)
    opt_args = (x1, y1, x2, y2, x_low_lim, x_high_lim)

    if debug:
        print "Testing objective function"
        print "Offset 0: ", objective_function_alignment(0, *opt_args)
        print "Offset 550: ", objective_function_alignment(-550, *opt_args)

    ## NOW SET THE RANGE OF OFFSETS TO BE PROBED
    xd1, xd2 = x1[-1] - x2[0], x1[0] - x2[-1]
    if offset_low_lim is None:
        x_min = np.min([xd1, xd2])
    else: x_min = offset_low_lim
    if offset_high_lim is None:
        x_max = np.max([xd1, xd2])
    else: x_max = offset_high_lim
    #
    if verbose:
        print "Searching for optimal offset in range:", x_min, " to ", x_max
    ##
    for idx in range(num_retries):
        if verbose:
            print >>sys.stdout, "\nTry %d to compute alignment" % idx
            sys.stdout.flush()
        retval = minimize_scalar(objective_function_alignment,
                                 args=opt_args,
                                 bounds=(x_min, x_max))
    ##
    if verbose:
        print >>sys.stdout,\
            "optimization took %d objective func evals" % objective_function_alignment.counter
        sys.stdout.flush()
    ### RETURN OPTIMIZED PARAMETERS
    return [retval.x, retval]
    ###}}}


def get_waveform(approximant, phase_order, amplitude_order, spin_order, template_params, start_frequency, sample_rate, length, datafile=None, verbose=False):
    #{{{
    print "IN hERE"
    delta_t  = 1./sample_rate
    delta_f  = 1./length
    filter_N = int(length)
    filter_n = filter_N / 2 + 1
    if approximant in fd_approximants() and 'Eccentric' not in approximant:
        print "NORMAL FD WAVEFORM for", approximant
        delta_f = sample_rate / length
        hplus, hcross = get_fd_waveform(template_params, approximant=approximant, spin_order=spin_order,
                               phase_order=phase_order, delta_f=delta_f,
                               f_lower=start_frequency, amplitude_order=amplitude_order)
    elif approximant in td_approximants() and 'Eccentric' not in approximant:
        print "NORMAL TD WAVEFORM for", approximant
        hplus,hcross = get_td_waveform(template_params, approximant=approximant, spin_order=spin_order,
                                   phase_order=phase_order, delta_t=1.0 / sample_rate,
                                   f_lower=start_frequency, amplitude_order=amplitude_order)
    elif 'EccentricIMR' in approximant:
        #{{{
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
          ecc   = getattr(template_params, 'alpha1')
          if 'E0' in approximant: ecc = 0
          anom  = getattr(template_params, 'alpha2')
          inc   = getattr(template_params, 'inclination')
          rtrans= getattr(template_params, 'alpha')
          beta  = 0
        except:
          raise RuntimeError(\
                "template_params does not have alpha{,1,2} or inclination")
        tol   = 1.e-16
        fmin  = start_frequency
        sample_rate = sample_rate
        #
        print >>sys.stdout, " Using phase order: %d" % phase_order
        sys.stdout.flush()
        hplus, hcross = Ecc.generate_eccentric_waveform(mass1, mass2,\
                            ecc, anom, inc, beta,\
                            tol,\
                            r_transition=rtrans,\
                            phase_order=phase_order,\
                            fmin=fmin,\
                            sample_rate=sample_rate,\
                            inspiral_only=False)
        #}}}
    elif 'EccentricInspiral' in approximant:
        #{{{
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
          ecc   = getattr(template_params, 'alpha1')
          if 'E0' in approximant: ecc = 0
          anom  = getattr(template_params, 'alpha2')
          inc   = getattr(template_params, 'inclination')
          beta  = getattr(template_params, 'alpha')
        except:
          raise RuntimeError(\
                "template_params does not have alpha{,1,2} or inclination")
        tol   = 1.e-16
        fmin  = start_frequency
        sample_rate = sample_rate
        #
        hplus, hcross = Ecc.generate_eccentric_waveform(mass1, mass2,\
                            ecc, anom, inc, beta,\
                            tol,\
                            phase_order=phase_order,\
                            fmin=fmin,\
                            sample_rate=sample_rate,\
                            inspiral_only=True)
        #}}}
    elif 'EccentricFD' in approximant:
        #{{{
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
          ecc   = getattr(template_params, 'alpha1')
          if 'E0' in approximant: ecc = 0
          anom  = getattr(template_params, 'alpha2')
          inc   = getattr(template_params, 'inclination')
        except:
          raise RuntimeError(\
                "template_params does not have alpha{1,2} or inclination")
        eccPar = ls.SimInspiralCreateTestGRParam("inclination_azimuth", inc)
        ls.SimInspiralAddTestGRParam(eccPar, "e_min", ecc)
        fmin = start_frequency
        fmax = sample_rate / 2
        #
        thp, thc = ls.SimInspiralChooseFDWaveform(0, delta_f,\
                        mass1*lal.MSUN_SI, mass2*lal.MSUN_SI,\
                        0,0,0,0,0,0,\
                        fmin, fmax, 0, 1.e6 * lal.PC_SI,\
                        inc, 0, 0, None, eccPar, -1, 7, ls.EccentricFD)
        hplus = FrequencySeries(thp.data.data[:], delta_f=thp.deltaF, epoch=thp.epoch)
        hcross= FrequencySeries(thc.data.data[:], delta_f=thc.deltaF, epoch=thc.epoch)
        #}}}
    elif 'FromDataFile' in approximant:
        #{{{
        # Legacy support
        if not os.path.exists(datafile): raise IOError("File %s not found!" % datafile)
        if verbose:
          print "Reading from data file %s" % datafile

        # Figure out waveform parameters from filename
        #q_value, M_value, w_value, _, _ = EA.get_q_m_e_pn_o_from_filename(datafile)
        q_value, M_value, w_value = EA.get_q_m_e_from_filename(datafile)

        # Read data, down-sample (assume data file is more finely sampled than
        # needed, i.e. interpolation is NOT supported, nor will be)
        data = np.loadtxt(datafile)
        dt = data[1,0] - data[0,0]
        delta_t = 1./sample_rate
        downsample_ratio = delta_t / dt
        if not approx_equal(downsample_ratio, np.int(downsample_ratio)):
          raise RuntimeError("Cannot handling resampling at a fractional factor = %e" % downsample_ratio)
        elif verbose:
          print "Downsampling by a factor of %d" % int(downsample_ratio)
        h_real = TimeSeries(data[::int(downsample_ratio),1]/DYN_RANGE_FAC, delta_t=delta_t)
        h_imag = TimeSeries(data[::int(downsample_ratio),2]/DYN_RANGE_FAC, delta_t=delta_t)

        if verbose:
          print "max, min,len of h_real = ", max(h_real.data), min(h_real.data), len(h_real.data)

        # Compute Strain
        tmplt_pars = template_params
        wav = generate_detector_strain(tmplt_pars, h_real, h_imag)
        wav = extend_waveform_TimeSeries(wav, filter_N)

        # Return TimeSeries with (m1, m2, w_value)
        m1, m2 = mtotal_eta_to_mass1_mass2(M_value, q_value / (1. + q_value)**2)
        htilde = make_frequency_series( wav )
        htilde = extend_waveform_FrequencySeries(htilde, filter_n)

        if verbose:
          print "ISNAN(htilde from file) = ", np.any(np.isnan(htilde.data))
        return htilde, [m1, m2, w_value, dt]
        #}}}
    else: raise IOError(".. APPROXIMANT %s not found.." % approximant)
    ##
    hvec = hplus
    htilde = make_frequency_series( hvec )
    htilde = extend_waveform_FrequencySeries(htilde, filter_n)
    #
    print "type of hplus, hcross = ", type(hplus.data), type(hcross.data)
    if any(isnan(hplus.data)) or any(isnan(hcross.data)):
      print "..### %s hplus or hcross have NANS!!" % approximant
    #
    if any(isinf(hplus.data)) or any(isinf(hcross.data)):
      print "..### %s hplus or hcross have INFS!!" % approximant
    #
    if any(isnan(htilde.data)):
      print "..### %s Fourier transform htilde has NANS!!" % approximant
    #
    if any(isinf(htilde.data)):
      print "..### %s Fourier transform htilde has INFS!!" % approximant
    #
    return htilde
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

######################################################################
######################################################################
#
#      UNUSED FUNCTIONS
#
######################################################################
######################################################################

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

