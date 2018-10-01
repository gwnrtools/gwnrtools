#!/bin/env python
# Copyright (C) 2014 Prayush Kumar, Heather Fong
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

try: from scipy.interpolate import interp1d
except ImportError: print "Warning: Dont uniformly sample output"

try:
  from DiscreteFunction import *
  from WaveFunction import *
  from DataAnalysis import *
except:
  print "Warning: Could not import Psi4->h integration modules"
  pass

try:
  import lal
  from glue.ligolw import utils as ligolw_utils
  from glue.ligolw import ligolw, table, lsctables
  @lsctables.use_in
  class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
    pass
  from pycbc.waveform import *
  from pycbc.psd import from_txt
  from pycbc.filter import match
  from pycbc.types import *
  from pycbc.detector import overhead_antenna_pattern as generate_fplus_fcross
except: 
  print "Warning: Could not import LAL/PyCBC modules"
  pass

#########################################################################
__author__ = "Prayush Kumar <prkumar@cita.utoronto.ca>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])

#########################################################################
#########################################################################

def update_progress(progress):
    print '\r\r[{0}] {1:.2%}'.format('#'*(int(progress*100)/2)+' '*(50-int(progress*100)/2), 
            progress),
    if progress == 100:
        print "Done"
    sys.stdout.flush()

## Remove the need for these functions ########################################
    
def generate_detector_strain(template_params, h_plus, h_cross):
    latitude = 0 
    longitude = 0 
    polarization = 0 

    if hasattr(template_params, 'latitude'):
        latitude = template_params.latitude
    if hasattr(template_params, 'longitude'):
        longitude = template_params.longitude
    if hasattr(template_params, 'polarization'):
        polarization = template_params.polarization

    f_plus, f_cross = generate_fplus_fcross(longitude, latitude, polarization)

    return h_plus * f_plus + h_cross * f_cross


def nextpow2(n): return 2**int( ceil( log2( n ) ) )

def getSec(s):
    l = s.split(':')
    return int(l[0]) * 3600 + int(l[1]) * 60 + int(l[2])

def get_uniform_mass_range( m_lower, m_upper, m_sep ):
  #{{{
  mlist = [m_lower]
  for m in np.arange( np.ceil(m_lower), np.floor(m_upper), m_sep ):
    mlist.append( m )
  mlist.append( m_upper )
  return np.array( mlist )
  #}}}

def extend_waveform_TimeSeries(wav, filter_N):
  #{{{
  if len(wav) != filter_N:
    _wav = TimeSeries(np.zeros(filter_N), delta_t=wav.delta_t,
                        dtype=real_same_precision_as(wav), epoch=wav._epoch)
    _wav[:len(wav)] = wav
  else: _wav = wav
  return _wav
  #}}}

def extend_waveform_FrequencySeries(wav, filter_n):
  #{{{
  if len(wav) != filter_n:
    _wav = FrequencySeries(np.zeros(filter_n), delta_f=wav.delta_f,
                        dtype=complex_same_precision_as(wav), epoch=wav._epoch)
    _wav[:len(wav)] = wav
  else: _wav = wav
  return _wav
  #}}}

def overlaps_vs_totalmass( wav1, wav2, psd=None, mf_lower=-1., \
                        m_lower=-1., m_upper=100., m_delta=5. ):
  # Need two wobjects of nr_waveform class. 
  # Waveforms are rescaled to different total masses and their overlaps computed
  # Returns an array of total masses and overlaps
  #{{{
  #print min(wav1.rawhp), max(wav1.rawhp), max(wav2.rawhp), min(wav2.rawhp)
  if psd is None: raise IOError("Provide the PSD please!")
  if mf_lower < 0:
    print "Initial orbital frequencies will be deduced after blending"
  #
  t2_opt = [1000,2000]
  t_option = [100,t2_opt[0],t2_opt[1],50,100]
  f_lower = 15. - 0.5
  # Calculate lowest total mass, from a) mf_lower, b) m_lower, c) calculate
  if mf_lower > 0: m_lower = mf_lower / f_lower / lal.MTSUN_SI
  elif m_lower <= 0:
    rescaled_mass, orbit_freq1 = wav1.get_orbital_frequency(t=max(t2_opt))
    rescaled_mass, orbit_freq2 = wav2.get_orbital_frequency(t=max(t2_opt))
    m_lower = max(orbit_freq1,orbit_freq2) * rescaled_mass / f_lower 
  print orbit_freq1, orbit_freq2, "lowest total Mass = %f" % m_lower
  #
  overlaps = []
  mass_range = get_uniform_mass_range( m_lower, m_upper, m_delta )
  for mtot in mass_range:
    #wav1.rescale_to_totalmass( mtot )
    #wav2.rescale_to_totalmass( mtot )
    wav_blended1 = blend(wav1,mtot,wav1.sample_rate,wav1.time_length,t_option) # blending
    wav_blended2 = blend(wav2,mtot,wav1.sample_rate,wav1.time_length,t_option) # blending
    if len(wav_blended1) != len(wav_blended2): 
      raise RuntimeError("blending function return different sets of waveforms!!")
    tmp_overlaps = [mtot]
    for ii in range( len(wav_blended1) ):
      hp1, hp2 = wav_blended1[ii], wav_blended2[ii]
      olap = overlap_between_waveforms(hp1, hp2,psd=psd)
      tmp_overlaps.append( olap )
      print "--In OvsM: window %d, overlap = %f" % (ii,olap)
    overlaps.append( tmp_overlaps )
  return overlaps
  #}}}

def overlap_between_waveforms( wav1, wav2, psd=None, f_lower=15. ):
  # Return overlap between two TimeSEries with psd needed as a FrequencySeries
  #{{{
  try:
    if psd == None: psd = self.psd
  except: raise IOError("Please compute and store PSD")
  #
  len1, len2, lenp = len(wav1), len(wav2), len(psd)
  if len1 != len2: raise IOError("Length of waveforms not equal: %d,%d"%(len1,len2))
  if wav1.delta_t != wav2.delta_t: raise IOError("Mismatched wave sample rate")
  if len1 != 2*lenp-2: raise IOError("PSD length inconsistent with waveforms")
  #
  return match(wav1, wav2, psd=psd, low_frequency_cutoff=f_lower)[0]
  #}}}

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

def blendTimeSeries(hp0, hc0, mm, sample, time, t_opt):
    # Only dealing with real part, don't do hc calculations
    # t_opt is length-5 array describing multiples of mm
    # Returns length-5 array of TimeSeries (1 per blending)
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

def initial_frequency_from_metadata( id_string, lev=None, xml_table=None ):
  #{{{
  if xml_table==None: raise IOError("Please provide the catalog table")
  if lev==None: raise IOError("What Lev is the waveform..?")
  for line in xml_table:
    if id_string in line.waveform and lev in line.waveform:
      return line.f_lower
  raise IOError("Waveform not found in the catalog..! Lev missing?")
  #}}}


def calculate_mismatch_between_levs_hdf5(self, \
              wavefilename='rhOverM_CcePITT_Asymptotic_GeometricUnits.h5', \
              outdir='matches', outputfile='OverlapsLevs.h5', catalogfile=None,\
              m_upper=100., m_delta=5.):
    #{{{
    cmd.getoutput('mkdir -p %s/%s' % (self.outdir,outdir))
    fout = h5py.File(self.outdir+'/'+outdir + '/' + outputfile, "a")
    #
    # Get the waveforms for different levs
    self.read_waveforms_from_hdf5_files(wavefilename=wavefilename)
    # Get PSD
    sample_rate, time_length = self.sample_rate, self.time_length
    N = sample_rate * time_length
    self.psd = self.get_psd()
    #
    ccefiles = self.wavefiles[self.levs[0]].keys()
    #ccefiles = list(np.sort( self.ccefiles[self.levs[0]] )[num_runs:])
    # Obtain the waveform files for given CceR, at Lev3,4,5
    # In pairs, compare Lev3,4,5
    self.levs.sort()
    for ccef in ccefiles:
      # choose a pair of levs
      for i1 in range(len(self.levs)):
        ld1 = self.levs[i1]
        for i2 in range(i1, len(self.levs)): # Include self overlaps
          ld2 = self.levs[i2]
          if ccef not in self.hwaveforms[ld1].keys() or \
                ccef not in self.hwaveforms[ld2].keys():
            print ccef, " waveforms not found in both %s and %s" % (ld1, ld2)
            continue
          # Create a group in output file for this ccefile
          if ccef not in fout.keys(): fout.create_group(ccef)
          # Compute matches
          if self.verbose:
            print >>sys.stderr, "\n\nOverlaps for %s Between %s and %s" % (ccef, ld1, ld2)
          overlaps = overlaps_vs_totalmass(self.hwaveforms[ld1][ccef],\
                      self.hwaveforms[ld2][ccef], psd=self.psd, \
                      m_upper=m_upper, m_delta=m_delta)
          # Add matches and masses as a dataset to the group
          dsetname = ld1 + '_' + ld2 + '.dat'
          fout[ccef].create_dataset(dsetname, data=overlaps)   
    #
    fout.flush()
    fout.close()
    return
    #}}}
  #
