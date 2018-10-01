#!/bin/env python
# Copyright @ 2015, Prayush Kumar

import os, sys, commands as cmd
import matplotlib as mp
mp.use('Agg')
mp.rc('text', usetex=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams.update({'text.usetex':True})

#import ccerun
import numpy as np
import h5py
import glob

#########################################################
#### Miscellaneous Functions
#########################################################

def add_strings(strlist):
  out = ''
  for s in strlist: out = out + s
  return out

def string_from_dir( dirname ):
  if '.dir' in dirname: return dirname.strip('.dir')
  else: return dirname

def dir_from_string( strname ):
  return strname + '.dir'

def string_from_dset( dsetname ):
  dtmp = dsetname.strip('.dat').split('_')
  #print dtmp
  # Get index of first dir
  for idx in range( len(dtmp) ):
    if '.dir' in dtmp[idx]: break
  d1, d2 = add_strings( dtmp[:idx+1] ), add_strings( dtmp[idx+1:] )
  d1, d2 = string_from_dir(d1), string_from_dir(d2)
  return d1 + '_' + d2

def dset_from_string( dsetname ):
  dtmp = dsetname.strip('.dat').split('_')
  #print dtmp
  # Get index of first dir
  for idx in range( len(dtmp) ):
    if '.dir' in dtmp[idx]: break
  d1, d2 = add_strings( dtmp[:idx+1] ), add_strings( dtmp[idx+1:] )
  d1, d2 = string_from_dir(d1), string_from_dir(d2)
  return d1 + '_' + d2

def spins_to_PNeffective_spin(m1, m2, chi1, chi2):
  chieff = (113.*m1*m1*chi1 + 113.*m2*m2*chi2 + 75.*m1*m2*(chi1 + chi2)) /\
          (113. * (m1 + m2)**2)
  return chieff

def spins_to_2PNeffective_spin(m1, m2, chi1, chi2):
  q1, q2 = 1, 1
  num = (1. + 80.*q1)* m1**2 *chi1**2 + (1. + 80.*q2)* m2**2 *chi2**2 \
        + 158. * m1*m2*chi1*chi2
  den = 16. * (m1 + m2)**2
  return num / den

def spins_to_massweighted_spin(m1, m2, chi1, chi2):
  chiwt = m1 * chi1 + m2 * chi2
  return chiwt / (m1+m2)

def spins_to_damoureffective_spin(m1, m2, chi1, chi2):
  chiwt = 4 * m1**2 * chi1 + 4 * m2**2 * chi2 + 3.*m1*m2*(chi1+chi2)
  return chiwt / 4. / (m1+m2)**2

def eta_to_q(eta):
  a = c = 1.
  b = 2. - 1./np.array(eta)
  D = (b**2 - 4.*a*c)**0.5
  #print type(a), type(b), type(c), type(D)
  roots = [(-b + D)/(2.*a), (-b - D)/(2.*a)]
  return roots[0]


#########################################################
#### Overlap storage classes
#########################################################

class overlaps_vs_totalmass():
  #{{{
  def __init__(self, dataset=None, verbose=True):
    #{{{
    if dataset is None: raise IOError("Need a dataset to initialize")
    self.M, self.O = dataset[:,0], dataset[:,1:]
    self.nWindows = len(self.O[0,:])
    #}}}
  #
  def X(self): return self.M
  #
  def Y(self, idx):
    #{{{
    if idx >= self.nWindows: raise IOError("This column does not exist in data")
    return self.O[:,idx]
    #}}}
  def get_overlap_mass_taper(self, mass, taperid):
    #{{{
    X = self.M
    Y = self.O
    if taperid < 0 or taperid >= self.nWindows:
      raise IOError("only have %d(%d) taperwins" % (self.nWindows,taperid))
    idx, = np.where( X==mass )
    if len(idx)==1: return Y[idx,taperid]
    for idx, mm in enumerate(X):
      if (mm-mass) < 1.e-12:
        return Y[idx,taperid]
    raise IOError("This mass value not found")
    #}}}
  #}}}
  
class Overlaps():
  """ Hide the complexity of HDF5 file structure here.
      Provide an interable, corresponding to subdirs in HDF5 file,
      Provide overlaps, labels corresponding to each subdir
  """
  #{{{
  def __init__(self, filename=None, outdir=None, verbose=True, debug=False):
    self.verbose = verbose
    self.debug = debug
    self.filename = filename
    self.outdir = outdir
    self.fullfilename = self.outdir + '/' + self.filename
    self.data = {}
    self.filedirs = {}
    self.filekeys = {}
    self.read_data()
    self.keys = self.iterables()
  #
  def string_from_dir(self, dirname ):
    if '.dir' in dirname: return dirname.strip('.dir')
    else: return dirname
  #
  def dir_from_string(self, strname ):
    return strname + '.dir'
  #
  def string_from_dset(self, dsetname ):
    dtmp = dsetname.strip('.dat').split('_')
    if self.debug: print dtmp
    # Get index of first dir
    for idx in range( len(dtmp) ):
      if '.dir' in dtmp[idx]: break
    if idx==len(dtmp)-1 and idx==1: idx = 0
    d1, d2 = add_strings( dtmp[:idx+1] ), add_strings( dtmp[idx+1:] )
    d1, d2 = self.string_from_dir(d1), self.string_from_dir(d2)
    d1, d2 = add_strings( dtmp[:idx+1] ), add_strings( dtmp[idx+1:] )
    d1, d2 = self.string_from_dir(d1), self.string_from_dir(d2)
    print "d1, d2 = ", d1, d2
    return d1 + '_' + d2
  #
  def dset_from_string(self, dsetname ):
    dtmp = dsetname.strip('.dat').split('_')
    if self.debug: print dtmp
    # Get index of first dir
    for idx in range( len(dtmp) ):
      if '.dir' in dtmp[idx]: break
    d1, d2 = add_strings( dtmp[:idx+1] ), add_strings( dtmp[idx+1:] )
    d1, d2 = string_from_dir(d1), string_from_dir(d2)
    return d1 + '_' + d2
  #
  def read_data( self ):
    self.fin = h5py.File(self.fullfilename, 'r')
    for k in self.fin.keys():
      self.filedirs[k] = []
      self.read_dir(l1dir=k, openfile=False)
    self.fin.close()
  #
  def read_dir( self, l1dir=None, openfile=True ):
    #{{{
    if openfile: self.fin = h5py.File(self.fullfilename, 'r')
    if l1dir not in self.filedirs: self.filedirs[l1dir] = []
    keys = self.fin[l1dir].keys()
    for k in keys:
      self.filedirs[l1dir].append( k )
      self.read_dataset( l1dir=l1dir, dset=k, openfile=False )
    return
    #}}}
  #
  def read_dataset( self, l1dir=None, dset=None, openfile=False ):
    #{{{
    if l1dir is None: raise IOError("No dir name given for reading dset")
    if dset is None: raise IOError("No dset name given for reading dset")
    if openfile: self.fin = h5py.File(self.fullfilename, 'r')
    itr = self.get_iterable( l1dir=l1dir, dsetname=dset )
    if itr not in self.filekeys.keys(): self.filekeys[itr] = [l1dir,dset]
    self.data[itr] = overlaps_vs_totalmass(dataset=self.fin[l1dir][dset].value)
    self.keys = self.data.keys()
    return
    #}}}
  #
  def get_dirdsetname_from_iterable(self, itr=None):
    #{{{
    if itr not in self.filekeys.keys():
      raise RuntimeError("Iter %s not defined" % itr)
    return self.filekeys[itr]
    #}}}
  #
  def get_iterable(self, l1dir=None, dsetname=None):
    #{{{
    if l1dir==None: raise IOError("No dir name given for reading dset")
    if dsetname==None: raise IOError("No dir name given for reading dset")
    try:
      itr = self.string_from_dir(l1dir) + '/' + self.string_from_dset(dsetname)
    except: raise("Problem with get_iterable for %s, %s" % (l1dir, dsetname))
    self.filekeys[itr] = [l1dir, dsetname]
    return itr
    #}}}
  #
  def iterables(self): return self.data.keys()
  #
  # This is the main function to return the inner dataset
  def overlaps_vs_totalmass(self, itr=None):
    #{{{
    if itr not in self.keys: raise IOError("Invalid iterable %s" % itr)
    return self.data[itr]
    #}}}
  #
  def overlaps(self, itr):
    #{{{
    return self.overlaps_vs_totalmass(itr=itr).O
    #}}}
  #
  def mismatches(self, itr):
    #{{{
    return 1. - self.overlaps_vs_totalmass(itr=itr).O
    #}}}
  #}}}

#### Classes to manipulate overlap data for one simulation
class errors_in_sim():
  """ Abstract the details of different error sources for each sim here
      Provide an iterable for each error source, as well as the dataset
    # LIST all the sources of mismatches HERE
    # 0. Method of waveform extraction -- CCE (1) or Extrapolation (2)
    # 1a. Extraction Radii
    # 1b. Levs
    # 1c. Tapering window
    # 2a. Extrapolation Order
    # 2b. Levs
    # 2c. Tapering window  """
  #{{{
  def __init__(self, simdir=None, matchdirs=['matches'], verbose=True,\
                      debug=False):
    #{{{
    self.verbose = verbose
    self.debug = debug
    self.simdir = simdir
    self.matchdirs = matchdirs
    if len( self.matchdirs ):
      for d in self.matchdirs:
        if not os.path.exists( self.simdir + '/' + d ):
          raise IOError("Match directories do not exist for %s" % self.simdir)
    for d in self.matchdirs:
      self.read_all_overlaps(matchdir=d)
    #}}}
  #
  def read_all_overlaps(self, matchdir=None):
    #{{{
    try: 
      if matchdir==None: matchdir = self.matchdirs[0]
    except IndexError: return
    pwd = cmd.getoutput('pwd')
    os.chdir(self.simdir)
    # Read in the matches
    self.read_ccer_overlaps(matchdir=matchdir)
    self.read_ccelev_overlaps(matchdir=matchdir)
    self.read_cceextrap_overlaps(matchdir=matchdir)
    if matchdir not in self.matchdirs: self.matchdirs.extend( matchdir )
    #
    os.chdir(pwd)
    return
    #}}}
  #
  def read_ccer_overlaps(self, matchdir=None,\
                          matchfile='OverlapsExtractionRadii.h5'):
    #{{{
    if self.verbose:
      print "Reading CCER overlaps frm %s/%s" % (matchdir,matchfile)
    self.ccer_overlaps = Overlaps(\
                  filename=matchfile,\
                  outdir=self.simdir+'/'+matchdir,\
                  verbose=self.verbose, debug=self.debug)
    self.ccer_dirnames = self.ccer_overlaps.filedirs.keys()
    self.ccer_dsetnames= {}
    for d in self.ccer_dirnames:
      self.ccer_dsetnames[d] = self.ccer_overlaps.filedirs[d]
    self.ccelevs =\
      [self.ccer_overlaps.string_from_dir(d) for d in self.ccer_dirnames]
    return
    #}}}
  #
  def read_ccelev_overlaps(self, matchdir=None,\
                          matchfile='OverlapsLevs.h5'):
    #{{{
    if self.verbose:
      print "Reading CCELev overlaps frm %s/%s" % (matchdir,matchfile)
    self.ccelev_overlaps = Overlaps(\
                  filename=matchfile,\
                  outdir=self.simdir+'/'+matchdir,\
                  verbose=self.verbose, debug=self.debug)
    self.ccelev_dirnames = self.ccelev_overlaps.filedirs.keys()
    self.ccelev_dsetnames= {}
    for d in self.ccelev_dirnames:
      self.ccelev_dsetnames[d] = self.ccelev_overlaps.filedirs[d]
    self.cceradii =\
      [self.ccer_overlaps.string_from_dir(d) for d in self.ccelev_dirnames]
    return
    #}}}
  #
  def read_cceextrap_overlaps(self, matchdir=None,\
                          matchfile='OverlapsExtrapolated.h5'):
    #{{{
    if self.verbose:
      print "Reading CCEExrapolation overlaps frm %s/%s" % (matchdir,matchfile)
    self.cceextrap_overlaps = Overlaps(\
                  filename=matchfile,\
                  outdir=self.simdir+'/'+matchdir,\
                  verbose=self.verbose, debug=self.debug)
    #
    self.cceextrap_dirnames = self.cceextrap_overlaps.filedirs.keys()
    self.cceextrap_dsetnames= {}
    for d in self.cceextrap_dirnames:
      self.cceextrap_dsetnames[d] = self.cceextrap_overlaps.filedirs[d]
    #
    if self.debug: print self.cceextrap_dsetnames
    self.extraporders = []
    for dsetname in self.cceextrap_dsetnames[self.cceextrap_dirnames[0]]:
      tmpsplit = dsetname.split('_')
      #if self.debug == True: print "tmpsplit = ", tmpsplit
      for idx in range(len(tmpsplit)):
        if 'Extrapolated' in tmpsplit[idx]:
          eo = tmpsplit[idx+1].split('.')[0]
          break
        elif 'Outer' in tmpsplit[idx]:
          eo = tmpsplit[idx].split('.')[0]
          break
      if eo not in self.extraporders: self.extraporders.append( eo )
      if self.debug == True: print "idx, tmpsplit = ", idx, tmpsplit[idx:]
      if self.debug: print "extraporders = ", self.extraporders
    return
    #}}}
  #
  def ccer(self, key=None, noduplicate=True, onlyduplicate=False):
    #{{{
    if noduplicate and onlyduplicate:
      noduplicate=False
      if self.verbose: print "Only returning duplicate comparisons"
    #tmp = [[itr, self.ccer_overlaps.overlaps_vs_totalmass(itr=itr)]\
    #            for itr in self.ccer_overlaps.keys]
    tmp = {}
    for itr in self.ccer_overlaps.keys:
      # Filter by all keys in the list key
      if key is not None:
        if type(key)==str and key not in itr: continue
        if type(key)==list:
          contflag = False
          for kk in key:
            if kk not in itr: contflag = True
          if contflag: continue
      #
      a, b = itr.split('/')[-1].split('_')
      if noduplicate and a==b: continue
      if onlyduplicate and a!=b: continue
      tmp[itr] = self.ccer_overlaps.overlaps_vs_totalmass(itr=itr)
    return tmp
    #}}}
  #
  def ccelev(self, key=None, noduplicate=True, onlyduplicate=False):
    #{{{
    if noduplicate and onlyduplicate:
      noduplicate=False
      if self.verbose: print "Only returning duplicate comparisons"
    #tmp = [[itr, self.ccelev_overlaps.overlaps_vs_totalmass(itr=itr)]\
    #            for itr in self.ccelev_overlaps.keys]
    tmp = {}
    for itr in self.ccelev_overlaps.keys:
      # Filter by all keys in the list key
      if key is not None:
        if type(key)==str and key not in itr: continue
        if type(key)==list:
          contflag = False
          for kk in key:
            if kk not in itr: contflag = True
          if contflag: continue
      #
      a, b = itr.split('/')[-1].split('_')
      if noduplicate and a==b: continue
      if onlyduplicate and a!=b: continue
      tmp[itr] = self.ccelev_overlaps.overlaps_vs_totalmass(itr=itr)
    return tmp
    #}}}
  #
  def cceextrapolated(self, key=None, noduplicate=True, onlyduplicate=False):
    #{{{
    if noduplicate and onlyduplicate:
      noduplicate=False
      if self.verbose: print "Only returning duplicate comparisons"
    #tmp = [[itr, self.cceextrap_overlaps.overlaps_vs_totalmass(itr=itr)]\
    #            for itr in self.cceextrap_overlaps.keys]
    tmp = {}
    for itr in self.cceextrap_overlaps.keys:
      # Filter by all keys in the list key
      if key is not None:
        if type(key)==str and key not in itr: continue
        if type(key)==list:
          contflag = False
          for kk in key:
            if kk not in itr: contflag = True
          if contflag: continue
      #
      a, b = itr.split('/')[-1].split('_')
      if noduplicate and a==b: continue
      if onlyduplicate and a!=b: continue
      tmp[itr] = self.cceextrap_overlaps.overlaps_vs_totalmass(itr=itr)    
    return tmp
    #}}}
  #
  def get_max_cce_mismatch(self, spl_lev='Lev5', spl_extrap=['N2','N3','N4'],\
                          spl_ccer=None):
    """ The purpose here is to estimate the error in the highest Lev waveforms
        As that is Lev5 for all simulations, its in that waveform that we need 
        to estimate the errors in. Instead of adding up (in some way) errors 
        from different sources, we approximate by taking the maximum of all 
        errors. This function will return the MAX error as a function of total 
        mass.
        
        List of mismatches MAXed over:
        1. For r in CCERs, Lev4 vs 5
        2. For Lev = Lev5 (?), CceR1 vs R2
        3. For lev = Lev5, CCER = ROUTER vs N2
        4. For lev = Lev5, CCER = ROUTER vs N3
        5. For lev = Lev5, CCER = ROUTER vs N4
        Further ones, to be added:
        6. Tapered vs Untapered SEOBNRV2.
    """
    #{{{
    if spl_ccer is None: 
      spl_ccer= 'CceR%04d' % max([int(x[-4:]) for x in self.cceradii])
    #
    const_combinations = [\
          [self.cceradii[0], spl_lev, 'Lev%d' % (int(spl_lev[-1])-1)]\
          ,[self.cceradii[1], spl_lev, 'Lev%d' % (int(spl_lev[-1])-1)]\
          ,[self.cceradii[0], self.cceradii[1], spl_lev]\
          ,[spl_ccer, spl_lev, 'N2']\
          ,[spl_ccer, spl_lev, 'N3']\
          ,[spl_ccer, spl_lev, 'N4']\
          ]
    const_funcs = [\
          self.ccelev\
          ,self.ccelev\
          ,self.ccer\
          ,self.cceextrapolated\
          ,self.cceextrapolated\
          ,self.cceextrapolated\
          ]
    if spl_lev != 'Lev5':
      const_combinations = [\
          [self.cceradii[0], spl_lev, 'Lev%d' % (int(spl_lev[-1])+1)]\
          ,[self.cceradii[1], spl_lev, 'Lev%d' % (int(spl_lev[-1])+1)]\
          ] + const_combinations
      const_funcs = [\
          self.ccelev\
          ,self.ccelev\
          ] + const_funcs
    #
    overlaps_all = []
    for idx, key_combo in enumerate(const_combinations):
      overlaps = const_funcs[idx](key=key_combo, noduplicate=True)
      if len(overlaps.values()) > 1:
        raise RuntimeError("keys %s gave %d resuls" %\
                          (key_combo, len(overlaps.keys())))
      overlaps_all.append( overlaps.values()[0] ) # proceed if only 1 dataset
    #
    # Now to find the MIN overlaps from all data sets
    # Find the lowest total masses
    max_min_mtotal = -1e8
    for olap in overlaps_all: 
      if max_min_mtotal < olap.M[0]:
        max_min_mtotal = max(max_min_mtotal, olap.M[0])
        min_masses = olap.X()
    num_taper_windows = olap.nWindows
    max_overlaps = np.ones(len(min_masses), 1+num_taper_windows) * -1.
    for taperid in range(num_taper_windows):
      for mid, mass in enumerate(min_masses):
        olaps = [obj.get_overlap_mass_taper(mass, taperid)\
                          for obj in overlaps_all]
        max_overlaps[mid, 1+taperid] = max(olaps)
        max_overlaps[mid, 0] = mass        
    # 
    return overlaps_vs_totalmass(dataset=max_overlaps)
    #}}}
  #}}}

#########################################################
#### Fitting-Factor storage classes
#########################################################

class effectualness_vs_totalmass():
  # For ALL simulations
  #{{{
  def __init__(self, outdir='.', verbose=True):
    #{{{
    self.verbose = verbose
    self.outdir = outdir
    self.data = {}
    #}}}
  #
  def read_data_from_combined_file(self, simtags=None, filename=None):
    """ Reads in the effectualness and parameter bias information from 
      a single file containing this information. Keep the data structure
      as before when reading from different files
    """
    #{{{
    if self.verbose: print >>sys.stderr, "Reading ", filename
    f = h5py.File(os.path.join(self.outdir, filename), 'r')
    sims = f.keys()
    if simtags != None and len(simtags) > 0:
      newsims = []
      for s in sims:
        for st in simtags:
          if st in s:
            newsims.append(s)
            break
      sims = newsims		
    if len(sims) == 0:
      print "File %s does not contain data for " % filename, simtags
      f.close()
      return
    for sim in sims:
      sim = str(sim)
      if sim not in self.data.keys(): self.data[sim] = {}
      simdata = f[sim]
      approxes = simdata.keys()
      for approx in approxes:
        approx = str(approx)
        if approx not in self.data[sim].keys(): self.data[sim][approx] = {}
        data = simdata[approx].value
        for mtot, nr_q, nr_s1, nr_s2, ff, mc_diff, et_diff, s1_diff, s2_diff in data:
          if mtot not in self.data[sim][approx].keys():
            # convert parameter bias data parameter values
            nr_et = nr_q / (1. + nr_q)**2
            nr_mc = mtot * nr_et**0.6
            sig_mc = nr_mc - nr_mc * mc_diff
            sig_et = nr_et - nr_et * et_diff
            sig_s1 = nr_s1 - s1_diff
            sig_s2 = nr_s2 - s2_diff
            # out_row = np.array([mtot, sig_et, sig_s1, sig_s2,\
            #                tmp_mc, tmp_et, tmp_s1, tmp_s2, olap])
            out_row = np.array([mtot, nr_et, nr_s1, nr_s2,\
							sig_mc, sig_et, sig_s1, sig_s2, ff])
            self.data[sim][approx][mtot] = np.array([out_row])
    return
    #}}}
  #
  def read_data_from_file(self, simtags=None, filename=None, keepalldata=False):
    """ Reads in data for ALL simulations and ALL approxes in a single output
        file. Stores them so they can be fetched from a simulation's nametag.
        If simtags are given (as a list), then only those simulations' data is 
        read, and nothing else is.
    """
    #{{{
    if self.verbose: print >>sys.stderr, "Reading ", filename
    num_of_aux_cols, num_of_data_cols = 10, 9
    # NR signal parameters and clumns
    # mtotal num_of_aux_cols, eta 1, spin1z 2, spin2z 3
    # template parameters and columns
    # mchirp 5, eta 6, spin1z 7, spin2z 8
    f = h5py.File(self.outdir+'/'+filename,'r')
    sims = f.keys()
    if simtags != None and len(simtags) > 0:
      newsims = []
      for s in sims:
        for st in simtags:
          if st in s:
            newsims.append( s )
            break
      sims = newsims
    if len(sims)==0: 
      #raise IOError("File %s is empty!" % filename)
      print "File %s does not contain data for " % filename, simtags
      f.close()
      return
    for sim in sims:
      sim = str(sim)
      if sim not in self.data.keys(): self.data[sim] = {}
      simdata = f[sim]
      approxes = simdata.keys()
      for approx in approxes:
        approx = str(approx)
        if approx not in self.data[sim].keys(): self.data[sim][approx] = {}
        data = simdata[approx].value
        rowidx = 0
        for row in data:
          #print "\n\n RowID = ", rowidx
          rowidx +=1
          for midx in np.arange(num_of_aux_cols,len(row),2):
            mtot = np.round(row[midx] * 100.) / 100.
            olap = row[midx + 1]
            if mtot < 0 or olap < 0: continue
            tmp_mc, tmp_et, tmp_s1, tmp_s2 = row[5:9]
            sig_et, sig_s1, sig_s2 = row[1:4]
            out_row = np.array([mtot, sig_et, sig_s1, sig_s2,\
                            tmp_mc, tmp_et, tmp_s1, tmp_s2, olap])
            #
            if mtot not in self.data[sim][approx].keys():
              self.data[sim][approx][mtot] = out_row
            else:
              try:
                nr,nc = np.shape(self.data[sim][approx][mtot])
                self.data[sim][approx][mtot] = \
                  np.append( self.data[sim][approx][mtot], [out_row], axis=0 )
              except:
                self.data[sim][approx][mtot] = \
                  np.append( [self.data[sim][approx][mtot]], [out_row], axis=0 )
        # Keep only the point with the maximum overlap
        if not keepalldata:
          for mtot in self.data[sim][approx].keys():
            tmp_data = self.data[sim][approx][mtot]
            max_idx = np.where( tmp_data[:,-1] == max(tmp_data[:,-1]) )[0][0]
            self.data[sim][approx][mtot] = np.array( [tmp_data[max_idx,:]] )
      #
    f.close()
    return
    #}}}
  #
  def effectualness_vs_totalmass(self, inkey=None, approx=None):
    #{{{
    for kk in self.data.keys():
      if inkey in kk: break
    for app in self.data[kk].keys():
      if approx in app: break
    masses = np.array( self.data[inkey][app].keys() )
    masses.sort()
    ff = np.array([max(self.data[inkey][app][mm][:,-1]) for mm in masses])
    return masses, ff
    #}}}
  #
  def best_match_parameters(self, inkey=None, approx=None):
    #{{{
    for kk in self.data.keys():
      if inkey in kk: break
    for app in self.data[kk].keys():
      if approx in app: break
    masses = np.array( self.data[inkey][app].keys() )
    masses.sort()
    ff = np.array([max(self.data[inkey][app][mm][:,-1]) for mm in masses])
    sig_mc = np.array(\
          [self.data[inkey][app][mm][np.where(self.data[inkey][app][mm][:,-1]==\
            max(self.data[inkey][app][mm][:,-1]))[0][0],-5] for mm in masses])
    sig_et = np.array(\
          [self.data[inkey][app][mm][np.where(self.data[inkey][app][mm][:,-1]==\
            max(self.data[inkey][app][mm][:,-1]))[0][0],-4] for mm in masses])
    sig_s1 = np.array(\
          [self.data[inkey][app][mm][np.where(self.data[inkey][app][mm][:,-1]==\
            max(self.data[inkey][app][mm][:,-1]))[0][0],-3] for mm in masses])
    sig_s2 = np.array(\
          [self.data[inkey][app][mm][np.where(self.data[inkey][app][mm][:,-1]==\
            max(self.data[inkey][app][mm][:,-1]))[0][0],-2] for mm in masses])
    # NR parameters are fixed for a simulation, so they dont need to be 
    # accumulated
    nr_et = self.data[inkey][app][mm][0,1]
    nr_q = (1. + (1. - 4.*nr_et)**0.5 - 2.*nr_et)/(2.*nr_et)
    nr_q = np.ones(len(ff)) * nr_q # Its constant
    nr_et= np.ones(len(ff)) * nr_et
    nr_mc = masses * nr_et**0.6
    nr_s1 = np.ones(len(ff)) * self.data[inkey][app][mm][0,2] # Its constant
    nr_s2 = np.ones(len(ff)) * self.data[inkey][app][mm][0,3] # Its constant
    #
    return ff, nr_mc, nr_et, nr_s1, nr_s2, sig_mc, sig_et, sig_s1, sig_s2
    #}}}
  #
  def parameterbiases_vs_parameters(self, inkey=None, approx=None,\
      chieff=False, total_mass=False):
    """ Computes the relative/absolute differences between maximum-overlap
    parameters and those of the injections themselves. Default behavior is to 
    return biases in 
    - chirp mass, 
    - eta, 
    - spin1, 
    - spin2, but
    the following input flags alter this : 
    - total_mass = true ==> instead of chirp mass, total mass diffs're returned
    """
    #{{{
    if chieff: from pycbc import pnutils
    else: print >>sys.stdout, "Not using effective spin"
    for kk in self.data.keys():
      if inkey in kk: break
    for app in self.data[kk].keys():
      if approx in app: break
    masses = np.array( self.data[inkey][app].keys() )
    masses.sort()
    ff = np.array([max(self.data[inkey][app][mm][:,-1]) for mm in masses])
    sig_mc = np.array(\
          [self.data[inkey][app][mm][np.where(self.data[inkey][app][mm][:,-1]==\
            max(self.data[inkey][app][mm][:,-1]))[0][0],-5] for mm in masses])
    sig_et = np.array(\
          [self.data[inkey][app][mm][np.where(self.data[inkey][app][mm][:,-1]==\
            max(self.data[inkey][app][mm][:,-1]))[0][0],-4] for mm in masses])
    if chieff:
      sig_m1, sig_m2 = pnutils.mchirp_eta_to_mass1_mass2(sig_mc, sig_et)
    if total_mass: sig_mt = sig_mc * sig_et**-0.6
    sig_s1 = np.array(\
          [self.data[inkey][app][mm][np.where(self.data[inkey][app][mm][:,-1]==\
            max(self.data[inkey][app][mm][:,-1]))[0][0],-3] for mm in masses])
    sig_s2 = np.array(\
          [self.data[inkey][app][mm][np.where(self.data[inkey][app][mm][:,-1]==\
            max(self.data[inkey][app][mm][:,-1]))[0][0],-2] for mm in masses])
    # NR parameters are fixed for a simulation, so they dont need to be 
    # accumulated
    nr_et = self.data[inkey][app][mm][0,1]
    nr_q = (1. + (1. - 4.*nr_et)**0.5 - 2.*nr_et)/(2.*nr_et)
    nr_q = np.ones(len(ff)) * nr_q # Its constant
    nr_et= np.ones(len(ff)) * nr_et
    nr_mc = masses * nr_et**0.6
    if chieff: nr_m1, nr_m2 = pnutils.mchirp_eta_to_mass1_mass2(nr_mc, nr_et)
    nr_s1 = np.ones(len(ff)) * self.data[inkey][app][mm][0,2] # Its constant
    nr_s2 = np.ones(len(ff)) * self.data[inkey][app][mm][0,3] # Its constant
    #
    if chieff:
      # Compute PN effective spins
      #nr_seff = spins_to_PNeffective_spin(nr_m1, nr_m2, nr_s1, nr_s2)
      #sig_seff= spins_to_PNeffective_spin(sig_m1, sig_m2, sig_s1, sig_s2)
      nr_seff = spins_to_massweighted_spin(nr_m1, nr_m2, nr_s1, nr_s2)
      sig_seff= spins_to_massweighted_spin(sig_m1, sig_m2, sig_s1, sig_s2)
    #mc_diff = (nr_mc-sig_mc)/nr_mc
    #eta_diff = (nr_et-sig_et)/nr_et
    #
    # NB: here 'mc_diff' really can contain either mchirp or mtotal differences,
    # please do not pay attention to the nomenclature 'mc'
    if total_mass: mc_diff = (sig_mt - masses)/masses
    else: mc_diff = (sig_mc - nr_mc)/nr_mc
    eta_diff = (sig_et-nr_et)/nr_et
    # Handle spins specially for q = 1
    if 'q1' not in inkey:
      if chieff: s1_diff = sig_seff - nr_seff
      else: s1_diff = sig_s1 - nr_s1 #nr_s1-sig_s1
      s2_diff = sig_s2 - nr_s2 #nr_s2-sig_s2
    else:
      s1_diff, s2_diff = np.zeros(len(nr_s1)), np.zeros(len(nr_s2))
      s11d, s12d = sig_s1-nr_s1, sig_s2-nr_s1 #nr_s1 - sig_s1, nr_s1 - sig_s2
      s21d, s22d = sig_s1-nr_s2, sig_s2-nr_s2 #nr_s2 - sig_s1, nr_s2 - sig_s2
      s1122rms = (s11d**2 + s22d**2)**0.5
      s1221rms = (s12d**2 + s21d**2)**0.5
      mask = s1122rms < s1221rms
      s1_diff[mask] = s11d[mask]
      s2_diff[mask] = s22d[mask]
      mask = s1122rms >=s1221rms
      s1_diff[mask] = s12d[mask]
      s2_diff[mask] = s21d[mask]
      if chieff: s1_diff = sig_seff - nr_seff
    return masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff
    #}}}
  #
  def effectualness_vs_parameters(self, inkey=None, approx=None):
    #{{{
    for kk in self.data.keys():
      if inkey in kk: break
    for app in self.data[kk].keys():
      if approx in app: break
    masses = np.array( self.data[inkey][app].keys() )
    masses.sort()
    ff = np.array([max(self.data[inkey][app][mm][:,-1]) for mm in masses])
    nr_et = self.data[inkey][app][mm][0,1]
    nr_q = (1. + (1. - 4.*nr_et)**0.5 - 2.*nr_et)/(2.*nr_et)
    nr_q = np.ones(len(ff)) * nr_q # Its constant
    nr_s1 = np.ones(len(ff)) * self.data[inkey][app][mm][0,2] # Its constant
    nr_s2 = np.ones(len(ff)) * self.data[inkey][app][mm][0,3] # Its constant
    return masses, nr_q, nr_s1, nr_s2, ff
    #}}}
  #}}}

#########################################################
#########################################################
####### PLOTTING CLASSES
#########################################################
#########################################################

#########################################################
####### EFFECTUALNESS
#########################################################
  
class plot_effectualness_vs_totalmass():
  #{{{
  def __init__(self, outdir='.', infiles = ['matches/match1.h5'],\
                plotdir='plots', verbose=True):
    self.verbose = verbose
    self.data = None
    self.outdir = outdir
    self.infiles = infiles
    self.plotdir = outdir + '/' + plotdir
    self.ApproxList = ['SEOBNRv1.dat', 'SEOBNRv2.dat',\
                        'IMRPhenomC.dat', 'IMRPhenomD.dat',\
                        'SpinTaylorT4.dat', 'TaylorF2.dat']
    self.lines   = ["-x","-o","-.","-^","-v","--"]
    self.markers = ["o","x","s","^","v","*",'.','<']
    self.colors  = ["blue","red","green","magenta","cyan","gold","black",\
                    "darkorange"]
  #
  def read_data_from_combined_file(self, simtags=[], \
	filename="EffectualnessParameterBiases_AllSims.h5"):
	#{{{
	if not os.path.exists(filename) or os.path.getsize(filename) == 0:
		raise IOError("%s not found" % filename)
	pwd = cmd.getoutput('pwd')
	os.chdir(self.outdir)
	self.data = effectualness_vs_totalmass(outdir=self.outdir)
	if self.verbose: print >>sys.stderr, "reading from >> ", filename
	self.data.read_data_from_combined_file(simtags=simtags, filename=filename)
	os.chdir(pwd)
	return
	#}}}
  #
  def read_data_from_all_files(self, tags=['./matches/*.h5'], simtags=[]):
    #{{{
    pwd = cmd.getoutput('pwd')
    os.chdir(self.outdir)
    self.data = effectualness_vs_totalmass(outdir=self.outdir)
    if len( tags ) > 0:
      infiles = []
      for tag in tags: infiles = infiles + glob.glob(tag)
    elif len(self.infiles) > 0: infiles = self.infiles
    else: raise IOError("Please specify which files to read, as a list OR tag")
    #print "USING ONLY 10 DATA FILES -- FIXME!"
    #infiles = infiles[:10] # FIXME
    if self.verbose: print >>sys.stderr, "reading from >> ", infiles
    for f in infiles: self.data.read_data_from_file(filename=f,simtags=simtags)
    os.chdir(pwd)
    return
    #}}}
  #
  def plot_effectualness_vs_totalmass(self, inkey=None,\
                          logy=True, figtype='pdf'):
    #{{{
    try: import matplotlib.pyplot as plt
    except: return
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    for sim in all_sims:
      plt.figure(int(1e7 * np.random.random()))
      for idx, app in enumerate(self.ApproxList):
        mm, ff = self.data.effectualness_vs_totalmass(inkey=sim, approx=app)
        #print "Masses = ", mm
        #print "FF = ", ff
        if not logy:
          plt.plot(mm, ff, label=app, \
                  linestyle=self.lines[-1],\
                  lw=3,\
                  marker=self.markers[idx],\
                  markersize=3,\
                  color=self.colors[idx])
        else:
          plt.semilogy(mm, 1.-ff, label=app, \
                  linestyle=self.lines[-1],\
                  lw=3,\
                  marker=self.markers[idx],\
                  markersize=3,\
                  color=self.colors[idx])
        plt.hold(True)
      plt.ylim(1.e-4,1)
      plt.legend(loc='best')
      plt.grid()
      plt.xlabel('Total Mass')# ($M_\odot$)')
      plt.ylabel('Effectualness')
      plt.title(sim.replace('_','-'))
      plt.savefig(self.plotdir+'/FF_%s.%s' % (sim[:-4],figtype))
    return
    #}}}    
  def plot_effectualness_totalmass_vs_parameters(self, inkey=None,\
                          logy=True, figtype='pdf'):
    #{{{
    try: import matplotlib.pyplot as plt
    except: return
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                        np.array([]), np.array([]), np.array([]), np.array([])
    for idx, app in enumerate(self.ApproxList):
      plt.figure(int(1e7 * np.random.random()))
      for sim in all_sims:
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_ff = np.append( nr_ff, ff )
      #
      # Mass - spin1
      if self.verbose: print >>sys.stderr, "Making M-S1 plot for ", app
      make_scatter_plot(nr_masses, nr_spin1z, nr_ff, \
          xlabel='Total Mass', ylabel='Spin of Bigger BH',\
          zlabel='Fitting Factor', title=app[:-4], \
          savefig=self.plotdir+'/FF_TotalMass_Spin1z_%s.%s'%(app[:-4],figtype))
      # Mass - spin2
      if self.verbose: print >>sys.stderr, "Making M-S2 plot for ", app
      make_scatter_plot(nr_masses, nr_spin2z, nr_ff, \
          xlabel='Total Mass', ylabel='Spin of Smaller BH',\
          zlabel='Fitting Factor', title=app[:-4], \
          savefig=self.plotdir+'/FF_TotalMass_Spin2z_%s.%s'%(app[:-4],figtype))
      # MassRatio - spin1
      if self.verbose: print >>sys.stderr, "Making Q-S1 plot for ", app
      make_scatter_plot(nr_massratios, nr_spin1z, nr_ff, \
          xlabel='Mass Ratio', ylabel='Spin of Bigger BH',\
          zlabel='Fitting Factor', title=app[:-4], \
          savefig=self.plotdir+'/FF_MassRatio_Spin1z_%s.%s'%(app[:-4],figtype))
      # MassRatio - spin2
      if self.verbose: print >>sys.stderr, "Making Q-S2 plot for ", app
      make_scatter_plot(nr_massratios, nr_spin2z, nr_ff, \
          xlabel='Mass Ratio', ylabel='Spin of Smaller BH',\
          zlabel='Fitting Factor', title=app[:-4], \
          savefig=self.plotdir+'/FF_MassRatio_Spin2z_%s.%s'%(app[:-4],figtype))
    return
    #}}}    
  def plot_mchirperror_vs_totalmass_parameters(self, inkey=None,\
                          logy=True, figtype='pdf'):
    #{{{
    try: import matplotlib.pyplot as plt
    except: return
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff,\
    mchirp_diff, eta_diff, spin1z_diff, spin2z_diff =\
     np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), \
     np.array([]), np.array([]), np.array([]), np.array([])
    for idx, app in enumerate(self.ApproxList):
      plt.figure(int(1e7 * np.random.random()))
      for sim in all_sims:
        masses, nr_q, nr_s1, nr_s2, mc_d, et_d, s1_d, s2_d, ff = \
                self.data.parameterbiases_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_ff = np.append( nr_ff, ff )
        mchirp_diff = np.append(mchirp_diff, mc_d)
        eta_diff = np.append(eta_diff, et_d)
        spin1z_diff = np.append(spin1z_diff, s1_d)
        spin2z_diff = np.append(spin2z_diff, s2_d)
      #
      # Mass - spin1
      if self.verbose: print >>sys.stderr, "Making M-S1 plot for ", app
      make_scatter_plot(nr_masses, nr_spin1z, mchirp_diff, \
          xlabel='Total Mass', ylabel='Spin of Bigger BH',\
          zlabel='Chirp Mass Fractional Bias', title=app[:-4], logz=False, \
          savefig=self.plotdir+\
                '/ChirpMassBias_TotalMass_Spin1z_%s.%s'%(app[:-4],figtype))
      # Mass - spin2
      if self.verbose: print >>sys.stderr, "Making M-S2 plot for ", app
      make_scatter_plot(nr_masses, nr_spin2z, mchirp_diff, \
          xlabel='Total Mass', ylabel='Spin of Smaller BH',\
          zlabel='Chirp Mass Fractional Bias', title=app[:-4], logz=False, \
          savefig=self.plotdir+\
              '/ChirpMassBias_TotalMass_Spin2z_%s.%s'%(app[:-4],figtype))
      # MassRatio - spin1
      if self.verbose: print >>sys.stderr, "Making Q-S1 plot for ", app
      make_scatter_plot(nr_massratios, nr_spin1z, mchirp_diff, \
          xlabel='Mass Ratio', ylabel='Spin of Bigger BH',\
          zlabel='Chirp Mass Fractional Bias', title=app[:-4], logz=False, \
          savefig=self.plotdir+\
              '/ChirpMassBias_MassRatio_Spin1z_%s.%s'%(app[:-4],figtype))
      # MassRatio - spin2
      if self.verbose: print >>sys.stderr, "Making Q-S2 plot for ", app
      make_scatter_plot(nr_massratios, nr_spin2z, mchirp_diff, \
          xlabel='Mass Ratio', ylabel='Spin of Smaller BH',\
          zlabel='Chirp Mass Fractional Bias', title=app[:-4], logz=False, \
          savefig=self.plotdir+\
              '/ChirpMassBias_MassRatio_Spin2z_%s.%s'%(app[:-4],figtype))
    return
    #}}}    
  def plot_etaerror_vs_totalmass_parameters(self, inkey=None,\
                          logy=True, figtype='pdf'):
    #{{{
    try: import matplotlib.pyplot as plt
    except: return
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff,\
    mchirp_diff, eta_diff, spin1z_diff, spin2z_diff =\
     np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), \
     np.array([]), np.array([]), np.array([]), np.array([])
    for idx, app in enumerate(self.ApproxList):
      plt.figure(int(1e7 * np.random.random()))
      for sim in all_sims:
        masses, nr_q, nr_s1, nr_s2, mc_d, et_d, s1_d, s2_d, ff = \
                self.data.parameterbiases_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_ff = np.append( nr_ff, ff )
        mchirp_diff = np.append(mchirp_diff, mc_d)
        eta_diff = np.append(eta_diff, et_d)
        spin1z_diff = np.append(spin1z_diff, s1_d)
        spin2z_diff = np.append(spin2z_diff, s2_d)
      #
      # Mass - spin1
      if self.verbose: print >>sys.stderr, "Making M-S1 plot for ", app
      make_scatter_plot(nr_masses, nr_spin1z, eta_diff, \
          xlabel='Total Mass', ylabel='Spin of Bigger BH',\
          zlabel='Eta Fractional Bias', title=app[:-4], logz=False, \
          savefig=self.plotdir+\
                '/EtaBias_TotalMass_Spin1z_%s.%s'%(app[:-4],figtype))
      # Mass - spin2
      if self.verbose: print >>sys.stderr, "Making M-S2 plot for ", app
      make_scatter_plot(nr_masses, nr_spin2z, eta_diff, \
          xlabel='Total Mass', ylabel='Spin of Smaller BH',\
          zlabel='Eta Fractional Bias', title=app[:-4], logz=False, \
          savefig=self.plotdir+\
              '/EtaBias_TotalMass_Spin2z_%s.%s'%(app[:-4],figtype))
      # MassRatio - spin1
      if self.verbose: print >>sys.stderr, "Making Q-S1 plot for ", app
      make_scatter_plot(nr_massratios, nr_spin1z, eta_diff, \
          xlabel='Mass Ratio', ylabel='Spin of Bigger BH',\
          zlabel='Eta Fractional Bias', title=app[:-4], logz=False, \
          savefig=self.plotdir+\
              '/EtaBias_MassRatio_Spin1z_%s.%s'%(app[:-4],figtype))
      # MassRatio - spin2
      if self.verbose: print >>sys.stderr, "Making Q-S2 plot for ", app
      make_scatter_plot(nr_massratios, nr_spin2z, eta_diff, \
          xlabel='Mass Ratio', ylabel='Spin of Smaller BH',\
          zlabel='Eta Fractional Bias', title=app[:-4], logz=False, \
          savefig=self.plotdir+\
              '/EtaBias_MassRatio_Spin2z_%s.%s'%(app[:-4],figtype))
    return
    #}}}    
  def plot_spin1error_vs_totalmass_parameters(self, inkey=None,\
                          logy=True, figtype='pdf'):
    #{{{
    try: import matplotlib.pyplot as plt
    except: return
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff,\
    mchirp_diff, eta_diff, spin1z_diff, spin2z_diff =\
     np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), \
     np.array([]), np.array([]), np.array([]), np.array([])
    for idx, app in enumerate(self.ApproxList):
      plt.figure(int(1e7 * np.random.random()))
      for sim in all_sims:
        masses, nr_q, nr_s1, nr_s2, mc_d, et_d, s1_d, s2_d, ff = \
                self.data.parameterbiases_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_ff = np.append( nr_ff, ff )
        mchirp_diff = np.append(mchirp_diff, mc_d)
        eta_diff = np.append(eta_diff, et_d)
        spin1z_diff = np.append(spin1z_diff, s1_d)
        spin2z_diff = np.append(spin2z_diff, s2_d)
      #
      # Mass - spin1
      if self.verbose: print >>sys.stderr, "Making M-S1 plot for ", app
      make_scatter_plot(nr_masses, nr_spin1z, spin1z_diff, \
          xlabel='Total Mass', ylabel='Spin of Bigger BH',\
          zlabel='Spin1 Fractional Bias', title=app[:-4], logz=False, \
          savefig=self.plotdir+\
                '/Spin1Bias_TotalMass_Spin1z_%s.%s'%(app[:-4],figtype))
      # Mass - spin2
      if self.verbose: print >>sys.stderr, "Making M-S2 plot for ", app
      make_scatter_plot(nr_masses, nr_spin2z, spin1z_diff, \
          xlabel='Total Mass', ylabel='Spin of Smaller BH',\
          zlabel='Spin1 Fractional Bias', title=app[:-4], logz=False, \
          savefig=self.plotdir+\
              '/Spin1Bias_TotalMass_Spin2z_%s.%s'%(app[:-4],figtype))
      # MassRatio - spin1
      if self.verbose: print >>sys.stderr, "Making Q-S1 plot for ", app
      make_scatter_plot(nr_massratios, nr_spin1z, spin1z_diff, \
          xlabel='Mass Ratio', ylabel='Spin of Bigger BH',\
          zlabel='Spin1 Fractional Bias', title=app[:-4], logz=False, \
          savefig=self.plotdir+\
              '/Spin1Bias_MassRatio_Spin1z_%s.%s'%(app[:-4],figtype))
      # MassRatio - spin2
      if self.verbose: print >>sys.stderr, "Making Q-S2 plot for ", app
      make_scatter_plot(nr_massratios, nr_spin2z, spin1z_diff, \
          xlabel='Mass Ratio', ylabel='Spin of Smaller BH',\
          zlabel='Spin1 Fractional Bias', title=app[:-4], logz=False, \
          savefig=self.plotdir+\
              '/Spin1Bias_MassRatio_Spin2z_%s.%s'%(app[:-4],figtype))
    return
    #}}}    
  def plot_spin2error_vs_totalmass_parameters(self, inkey=None,\
                          logy=True, figtype='pdf'):
    #{{{
    try: import matplotlib.pyplot as plt
    except: return
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff,\
    mchirp_diff, eta_diff, spin1z_diff, spin2z_diff =\
     np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), \
     np.array([]), np.array([]), np.array([]), np.array([])
    for idx, app in enumerate(self.ApproxList):
      plt.figure(int(1e7 * np.random.random()))
      for sim in all_sims:
        masses, nr_q, nr_s1, nr_s2, mc_d, et_d, s1_d, s2_d, ff = \
                self.data.parameterbiases_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_ff = np.append( nr_ff, ff )
        mchirp_diff = np.append(mchirp_diff, mc_d)
        eta_diff = np.append(eta_diff, et_d)
        spin1z_diff = np.append(spin1z_diff, s1_d)
        spin2z_diff = np.append(spin2z_diff, s2_d)
      #
      # Mass - spin1
      if self.verbose: print >>sys.stderr, "Making M-S1 plot for ", app
      make_scatter_plot(nr_masses, nr_spin1z, spin2z_diff, \
          xlabel='Total Mass', ylabel='Spin of Bigger BH',\
          zlabel='Spin2 Fractional Bias', title=app[:-4], logz=False, \
          savefig=self.plotdir+\
                '/Spin2Bias_TotalMass_Spin1z_%s.%s'%(app[:-4],figtype))
      # Mass - spin2
      if self.verbose: print >>sys.stderr, "Making M-S2 plot for ", app
      make_scatter_plot(nr_masses, nr_spin2z, spin2z_diff, \
          xlabel='Total Mass', ylabel='Spin of Smaller BH',\
          zlabel='Spin2 Fractional Bias', title=app[:-4], logz=False, \
          savefig=self.plotdir+\
              '/Spin2Bias_TotalMass_Spin2z_%s.%s'%(app[:-4],figtype))
      # MassRatio - spin1
      if self.verbose: print >>sys.stderr, "Making Q-S1 plot for ", app
      make_scatter_plot(nr_massratios, nr_spin1z, spin2z_diff, \
          xlabel='Mass Ratio', ylabel='Spin of Bigger BH',\
          zlabel='Spin2 Fractional Bias', title=app[:-4], logz=False, \
          savefig=self.plotdir+\
              '/Spin2Bias_MassRatio_Spin1z_%s.%s'%(app[:-4],figtype))
      # MassRatio - spin2
      if self.verbose: print >>sys.stderr, "Making Q-S2 plot for ", app
      make_scatter_plot(nr_massratios, nr_spin2z, spin2z_diff, \
          xlabel='Mass Ratio', ylabel='Spin of Smaller BH',\
          zlabel='Spin2 Fractional Bias', title=app[:-4], logz=False, \
          savefig=self.plotdir+\
              '/Spin2Bias_MassRatio_Spin2z_%s.%s'%(app[:-4],figtype))
    return
    #}}}    
  def plot_effectualness_vs_parameters(self, inkey=None,\
            logy=True, elevation=30, azimuthal=30, alpha=0.8, figtype='pdf'):
    #{{{
    try: import matplotlib.pyplot as plt
    except: return
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                        np.array([]), np.array([]), np.array([]), np.array([])
    for idx, app in enumerate(self.ApproxList):
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_ff = np.append( nr_ff, ff )
      #
      # Mass - spin1
      if self.verbose: print >>sys.stderr, "Making M-S1 plot for ", app
      print "With ", nr_spin1z, nr_spin2z, nr_masses, nr_ff
      make_scatter_plot3D(nr_spin1z, nr_spin2z, nr_masses, nr_ff, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, label=inkey,\
          xlabel='Spin of Bigger BH', ylabel='Spin of Smaller BH',\
          zlabel='Total Mass', clabel='Fitting Factor', title=app[:-4],\
          savefig=self.plotdir+'/FF_TotalMass_Spin1z_Spin2z_%s.%s'\
                                                  %(app[:-4],figtype))
    return
    #}}}    
  def write_effectualness_vs_parameters_mult(self, \
                      outfile='effectualness_parameters.h5'):
    #{{{
    if self.data == None: self.read_data_from_all_files()
    try: fout = h5py.File(outfile, 'w')
    except: raise IOError("Error opening %s for writing" % outfile)
    all_sims = self.data.data.keys()
    for sim in all_sims:
      fout.create_group(sim)
      for idx, app in enumerate(self.ApproxList):
        masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                  self.data.parameterbiases_vs_parameters(inkey=sim, approx=app)
        dout = np.array([[masses[i], nr_q[i], nr_s1[i], nr_s2[i], ff[i], mc_diff[i], eta_diff[i], s1_diff[i], s2_diff[i]] for i in range(len(ff))])
        fout[sim].create_dataset(app, data=dout)
    fout.close()
    return
    #}}}    
  def plot_effectualness_vs_parameters_mult(self, logy=True, elevation=30, \
                      azimuthal=30, alpha=0.8, figtype='pdf'):
    #{{{
    try: import matplotlib.pyplot as plt
    except: return
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
    #                    np.array([]), np.array([]), np.array([]), np.array([])
    for idx, app in enumerate(self.ApproxList):
      # q = 1
      inkey = 'q1'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                          np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_ff = np.append( nr_ff, ff )
      nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      # q = 2
      inkey = 'q2'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                          np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_ff = np.append( nr_ff, ff )
      nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      # q = 3
      inkey = 'q3'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                          np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_ff = np.append( nr_ff, ff )
      nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      #
      if self.verbose: print >>sys.stderr, "Making M-S1 plot for ", app
      print "With ", nr_spin1z, nr_spin2z, nr_masses, nr_ff
      if 'SEOBNRv2' in app:
        bounds = np.log10([0.0001,0.005, 0.01, 0.02, 0.03, 0.05])
      elif 'SEOBNRv1' in app:
        bounds = np.log10([0.0001,0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2])
      elif 'PhenomC' in app:
        bounds = np.log10([0.0001,0.005, 0.01, 0.02, 0.03, 0.05])
      elif 'Taylor' in app:
        bounds = np.log10([0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.])
      else: raise IOError("Approximant %s bounds not known" % app)
      make_scatter_plot3D_mult(\
          nr_spin1zq1, nr_spin2zq1, nr_massesq1, 1 - nr_ffq1, \
          nr_spin1zq2, nr_spin2zq2, nr_massesq2, 1 - nr_ffq2, \
          nr_spin1zq3, nr_spin2zq3, nr_massesq3, 1 - nr_ffq3, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='Total Mass$\,(M_{\odot})$', clabel='$\mathcal{M}$',\
          bounds=bounds,\
          title=app[:-4],\
          savefig=self.plotdir+'/FF_TotalMass_Spin1z_Spin2z_%s_1.%s'\
                                                  %(app[:-4],figtype))
      if 'SEOBNRv2' in app:
        bounds = np.log10([0.0001,0.005, 0.01, 0.05, 0.1])
      elif 'SEOBNRv1' in app:
        bounds = np.log10([0.0001,0.005, 0.01, 0.03, 0.05, 0.1, 0.2])
      elif 'PhenomC' in app:
        bounds = np.log10([0.0001,0.005, 0.01, 0.03, 0.05])
      elif 'Taylor' in app:
        bounds = np.log10([0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.])
      else: raise IOError("Approximant %s bounds not known" % app)
      make_scatter_plot3D_mult(\
          nr_spin1zq1, nr_spin2zq1, nr_massesq1, 1 - nr_ffq1, \
          nr_spin1zq2, nr_spin2zq2, nr_massesq2, 1 - nr_ffq2, \
          nr_spin1zq3, nr_spin2zq3, nr_massesq3, 1 - nr_ffq3, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='Total Mass$\,(M_{\odot})$', clabel='$\mathcal{M}$',\
          bounds=bounds,\
          title=app[:-4],\
          savefig=self.plotdir+'/FF_TotalMass_Spin1z_Spin2z_%s_3.%s'\
                                                  %(app[:-4],figtype))
      if 'SEOBNRv2' in app:
        bounds = np.log10([0.0001,0.005, 0.01, 0.02, 0.03, 0.1])
      elif 'SEOBNRv1' in app:
        bounds = np.log10([0.0001,0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 1.])
      elif 'PhenomC' in app:
        bounds = np.log10([0.0001,0.005, 0.01, 0.02, 0.03, 0.05])
      elif 'Taylor' in app:
        bounds = np.log10([0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.])
      else: raise IOError("Approximant %s bounds not known" % app)
      make_scatter_plot3D_mult(\
          nr_spin1zq1, nr_spin2zq1, nr_massesq1, 1 - nr_ffq1, \
          nr_spin1zq2, nr_spin2zq2, nr_massesq2, 1 - nr_ffq2, \
          nr_spin1zq3, nr_spin2zq3, nr_massesq3, 1 - nr_ffq3, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='Total Mass$\,(M_{\odot})$', clabel='$\mathcal{M}$',\
          bounds=bounds,\
          title=app[:-4],\
          savefig=self.plotdir+'/FF_TotalMass_Spin1z_Spin2z_%s_2.%s'\
                                                  %(app[:-4],figtype))
    return
    #}}}    
  def plot_effectualness_vs_parameters_multrow(self, logy=True, elevation=30, \
            ApproxList=[],\
            onlyimr=True, bounds=None, azimuthal=30, alpha=0.8, figtype='pdf'):
    #{{{
    try: import matplotlib.pyplot as plt
    except: return
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    # which approximants to plot ?
    if type(ApproxList) == list and len(ApproxList) != 0:
      approx_present = True
      for app in ApproxList:
        if app not in self.ApproxList: approx_present = False
      if not approx_present: ApproxList = self.ApproxList
    else: ApproxList = self.ApproxList
    #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
    #                    np.array([]), np.array([]), np.array([]), np.array([])
    Xs, Ys, Zs, Cs = [], [], [], []
    titles = []
    for idx, app in enumerate(ApproxList):
      if onlyimr:
        if 'Taylor' in app: continue
      else:
        if 'Taylor' not in app: continue
      print "\n\n Adding %s for plotting" % app
      # q = 1
      inkey = 'q1'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                          np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_ff = np.append( nr_ff, ff )
      nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      # q = 2
      inkey = 'q2'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                          np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_ff = np.append( nr_ff, ff )
      nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      # q = 3
      inkey = 'q3'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                          np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_ff = np.append( nr_ff, ff )
      nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      #
      Xrow = [nr_spin1zq1, nr_spin1zq2, nr_spin1zq3]
      Yrow = [nr_spin2zq1, nr_spin2zq2, nr_spin2zq3]
      Zrow = [nr_massesq1, nr_massesq2, nr_massesq3]
      Crow = [1. -nr_ffq1, 1. -nr_ffq2, 1. -nr_ffq3]
      #
      Xs.append( Xrow )
      Ys.append( Yrow )
      Zs.append( Zrow )
      Cs.append( Crow )
      #
      titles.append( app[:-4] )
    ###################################
    # Now make the plots
    #print "With ", nr_spin1z, nr_spin2z, nr_masses, nr_ff
    if type(bounds) != np.ndarray and type(bounds) != list:
      if bounds:
        print "SHOULD NOT HAPPEN"
        #bounds = np.array([-0.05,-0.03,-0.01,-0.005,0.005,0.01,0.03,0.05])
    make_scatter_plot3D_multrow( Xs, Ys, Zs, Cs, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='M $(M_\odot)$',\
          #clabel='$\mathcal{M}$',\
          clabel='$1- \mathrm{Fitting~Factor}$',\
          title=titles,\
          logC=False,\
          bounds=bounds,\
          savefig=self.plotdir+'/FF_TotalMass_Spin1z_Spin2z.%s'\
                                                  %(figtype))
    #
    return
    #}}}    
  def plot_effectualness_contours_vs_parameters(self, logy=True, elevation=30, \
            ApproxList=[], selectZ='minC', FForMM='FF', \
            onlyimr=True, bounds=None, colors=[],\
            xlabel='', ylabel='', titles=[],\
            azimuthal=30, alpha=0.8, figtype='pdf'):
    #{{{
    try: import matplotlib.pyplot as plt
    except: return
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    # which approximants to plot ?
    if type(ApproxList) == list and len(ApproxList) != 0:
      approx_present = True
      for app in ApproxList:
        if app not in self.ApproxList: approx_present = False
      if not approx_present: ApproxList = self.ApproxList
    else: ApproxList = self.ApproxList
    #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
    #                    np.array([]), np.array([]), np.array([]), np.array([])
    Xs, Ys, Zs, Cs = [], [], [], []
    title = []
    for idx, app in enumerate(ApproxList):
      if onlyimr:
        if 'Taylor' in app: continue
      else:
        if 'Taylor' not in app: continue
      print "\n\n Adding %s for plotting" % app
      # q = 1
      inkey = 'q1'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                          np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        # Select which Z value to use for plotting
        if selectZ == 'maxC':
          idxmaxC = np.where(ff == np.max(ff))[0][0]
        elif selectZ == 'minC':
          idxmaxC = np.where(ff == np.min(ff))[0][0]
        nr_masses = np.append( nr_masses, masses[idxmaxC] )
        nr_massratios = np.append( nr_massratios, nr_q[idxmaxC] )
        nr_spin1z = np.append( nr_spin1z, nr_s1[idxmaxC] )
        nr_spin2z = np.append( nr_spin2z, nr_s2[idxmaxC] )
        nr_ff = np.append( nr_ff, ff[idxmaxC] )
      nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      # q = 2
      inkey = 'q2'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                          np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        # Select which Z value to use for plotting
        if selectZ == 'maxC':
          idxmaxC = np.where(ff == np.max(ff))[0][0]
        elif selectZ == 'minC':
          idxmaxC = np.where(ff == np.min(ff))[0][0]
        nr_masses = np.append( nr_masses, masses[idxmaxC] )
        nr_massratios = np.append( nr_massratios, nr_q[idxmaxC] )
        nr_spin1z = np.append( nr_spin1z, nr_s1[idxmaxC] )
        nr_spin2z = np.append( nr_spin2z, nr_s2[idxmaxC] )
        nr_ff = np.append( nr_ff, ff[idxmaxC] )
      nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      # q = 3
      inkey = 'q3'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                          np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        # Select which Z value to use for plotting
        if selectZ == 'maxC':
          idxmaxC = np.where(ff == np.max(ff))[0][0]
        elif selectZ == 'minC':
          idxmaxC = np.where(ff == np.min(ff))[0][0]
        nr_masses = np.append( nr_masses, masses[idxmaxC] )
        nr_massratios = np.append( nr_massratios, nr_q[idxmaxC] )
        nr_spin1z = np.append( nr_spin1z, nr_s1[idxmaxC] )
        nr_spin2z = np.append( nr_spin2z, nr_s2[idxmaxC] )
        nr_ff = np.append( nr_ff, ff[idxmaxC] )
      nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      #
      Xrow = [nr_spin1zq1, nr_spin1zq2, nr_spin1zq3]
      Yrow = [nr_spin2zq1, nr_spin2zq2, nr_spin2zq3]
      Zrow = [nr_massesq1, nr_massesq2, nr_massesq3]
      if 'FF' in FForMM: Crow = [nr_ffq1, nr_ffq2, nr_ffq3]
      elif 'MM' in FForMM: Crow = [1. -nr_ffq1, 1. -nr_ffq2, 1. -nr_ffq3]
      #
      Xs.append( Xrow )
      Ys.append( Yrow )
      Zs.append( Zrow )
      Cs.append( Crow )
      #
      title.append( app[:-4] )
    ###################################
    # Now make the plots
    if 'FF' in FForMM: plotprefix = 'FF'
    elif 'MM' in FForMM: plotprefix = 'MM'
    #
    if FForMM == 'FF': clabel = '$\mathrm{Fitting~Factor}$'
    elif FForMM == 'MM': clabel = '$1- \mathrm{Fitting~Factor}$'
    elif FForMM == 'VolFF':
      clabel = 'Detection Fraction'
      nrows, ncols = np.shape(Cs)
      for rowi in range(nrows):
        for coli in range(ncols):
          Cs[rowi][coli] = Cs[rowi][coli]**3. 
    elif FForMM == 'VolMM':
      clabel = 'Detection Loss'
      nrows, ncols = np.shape(Cs)
      for rowi in range(nrows):
        for coli in range(ncols):
          Cs[rowi][coli] = 1. - (1.-Cs[rowi][coli])**3. 
    print "With ", nr_spin1z, nr_spin2z, nr_masses, nr_ff
    if type(bounds) != np.ndarray and type(bounds) != list:
      if bounds:
        print "SHOULD NOT HAPPEN"
        #bounds = np.array([-0.05,-0.03,-0.01,-0.005,0.005,0.01,0.03,0.05])
    #
    if xlabel=='': xlabel='$\chi_1$'
    if ylabel=='': ylabel='$\chi_2$'
    #
    make_contour_plot_multrow( Xs, Ys, Cs, \
          alpha=alpha, logC=logy, \
          xlabel=xlabel, ylabel=ylabel, titles=titles,\
          clabel=clabel, title=title, bounds=bounds, colors=colors,\
          savefig=self.plotdir+'/%s_Spin1z_Spin2z.%s'\
                                                  %(plotprefix,figtype))
    #
    return
    #}}}    
  def plot_parameterbiases_vs_parameters_mult(self, logy=True, elevation=30, \
                      bounds=True, azimuthal=30, alpha=0.8, figtype='pdf'):
    #{{{
    try: import matplotlib.pyplot as plt
    except: return
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
    #                    np.array([]), np.array([]), np.array([]), np.array([])
    for idx, app in enumerate(self.ApproxList):
      # q = 1
      inkey = 'q1'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
      nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                  self.data.parameterbiases_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_mcdiff = np.append( nr_mcdiff, mc_diff )
        nr_etdiff = np.append( nr_etdiff, eta_diff )
        nr_s1diff = np.append( nr_s1diff, s1_diff )
        nr_s2diff = np.append( nr_s2diff, s2_diff )
        nr_ff = np.append( nr_ff, ff )
      nr_mcdiffq1, nr_etdiffq1, nr_s1diffq1, nr_s2diffq1 = nr_mcdiff,\
                    nr_etdiff, nr_s1diff, nr_s2diff
      nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      # q = 2
      inkey = 'q2'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
      nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                  self.data.parameterbiases_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_mcdiff = np.append( nr_mcdiff, mc_diff )
        nr_etdiff = np.append( nr_etdiff, eta_diff )
        nr_s1diff = np.append( nr_s1diff, s1_diff )
        nr_s2diff = np.append( nr_s2diff, s2_diff )
        nr_ff = np.append( nr_ff, ff )
      nr_mcdiffq2, nr_etdiffq2, nr_s1diffq2, nr_s2diffq2 = nr_mcdiff,\
                    nr_etdiff, nr_s1diff, nr_s2diff
      nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      # q = 3
      inkey = 'q3'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
      nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                  self.data.parameterbiases_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_mcdiff = np.append( nr_mcdiff, mc_diff )
        nr_etdiff = np.append( nr_etdiff, eta_diff )
        nr_s1diff = np.append( nr_s1diff, s1_diff )
        nr_s2diff = np.append( nr_s2diff, s2_diff )
        nr_ff = np.append( nr_ff, ff )
      nr_mcdiffq3, nr_etdiffq3, nr_s1diffq3, nr_s2diffq3 = nr_mcdiff,\
                    nr_etdiff, nr_s1diff, nr_s2diff
      nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      #
      # Mass - spin1
      if self.verbose: print >>sys.stderr, "Making M-S1 plot for ", app
      print "With ", nr_spin1z, nr_spin2z, nr_masses, nr_ff
      if type(bounds) != np.ndarray and type(bounds) != list:
          if bounds:
              print "SHOULD NOT HAPPEN"
	      bounds = np.array([-0.05,-0.03,-0.01,-0.005,0.005,0.01,0.03,0.05])
      make_scatter_plot3D_mult(\
          nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_mcdiffq1, \
          nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_mcdiffq2, \
          nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_mcdiffq3, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='Total Mass $(M_\odot)$',\
          clabel='$\Delta\mathcal{M}_c/\mathcal{M}_c$', title=app[:-4],\
          logC=False,\
          bounds=bounds,\
          savefig=self.plotdir+'/ChirpMassError_TotalMass_Spin1z_Spin2z_%s.%s'\
                                                  %(app[:-4],figtype))
      #return
      #
      if type(bounds) != np.ndarray and type(bounds) != list:
	  if bounds:
	      bounds = np.array([-0.5,-0.2,-0.1,-0.05,-0.02,0.02,0.05,0.1,0.2,0.5])
      make_scatter_plot3D_mult(\
          nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_etdiffq1, \
          nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_etdiffq2, \
          nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_etdiffq3, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='Total Mass $(M_\odot)$',\
          clabel='$\Delta\eta/\eta$', title=app[:-4],\
          logC=False,\
          bounds=bounds,\
          savefig=self.plotdir+'/EtaError_TotalMass_Spin1z_Spin2z_%s.%s'\
                                                  %(app[:-4],figtype))
      #
      if type(bounds) != np.ndarray and type(bounds) != list:
	  if bounds:
	      bounds = np.array([-0.5,-0.2,-0.1,-0.05,-0.02,0.02,0.05,0.1,0.2,0.5])
      make_scatter_plot3D_mult(\
          nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_s1diffq1, \
          nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_s1diffq2, \
          nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_s1diffq3, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='Total Mass $(M_\odot)$',\
          clabel='$\Delta\chi_1$', title=app[:-4],\
          logC=False,\
          bounds=bounds,\
          savefig=self.plotdir+'/Chi1Error_TotalMass_Spin1z_Spin2z_%s.%s'\
                                                  %(app[:-4],figtype))
      #
      if type(bounds) != np.ndarray and type(bounds) != list:
	  if bounds:
	      bounds = np.array([-0.5,-0.2,-0.1,-0.05,-0.02,0.02,0.05,0.1,0.2,0.5])
      make_scatter_plot3D_mult(\
          nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_s2diffq1, \
          nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_s2diffq2, \
          nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_s2diffq3, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='Total Mass $(M_\odot)$',\
          clabel='$\Delta\chi_2$', title=app[:-4],\
          logC=False,\
          bounds=bounds,\
          savefig=self.plotdir+'/Chi2Error_TotalMass_Spin1z_Spin2z_%s.%s'\
                                                  %(app[:-4],figtype))
      #
    return
    #}}}
  #
  def plot_parameterbiases_vs_parameters_multrow(self, logy=True, elevation=30,\
            ApproxList=[], onlyimr=True, chieff=False, total_mass=False,\
            colormin=None, colormax=None,\
            bounds=None, azimuthal=30, alpha=0.8, figtype='pdf'):
    #{{{
    try: import matplotlib.pyplot as plt
    except: return
    #
    # which approximants to plot ?
    if type(ApproxList) == list and len(ApproxList) != 0:
      approx_present = True
      for app in ApproxList:
        if app not in self.ApproxList: approx_present = False
      if not approx_present: ApproxList = self.ApproxList
    else: ApproxList = self.ApproxList
    ###############################
    # Read data
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
    #                    np.array([]), np.array([]), np.array([]), np.array([])
    Xs, Ys, Zs, mcCs, etCs, s1Cs, s2Cs = [], [], [], [], [], [], []
    titles = []
    for idx, app in enumerate(ApproxList):
      if onlyimr:
        if 'Taylor' in app: continue
      # q = 1
      inkey = 'q1'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
      nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                  self.data.parameterbiases_vs_parameters(inkey=sim,\
                      chieff=chieff, approx=app, total_mass=total_mass)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_mcdiff = np.append( nr_mcdiff, mc_diff )
        nr_etdiff = np.append( nr_etdiff, eta_diff )
        nr_s1diff = np.append( nr_s1diff, s1_diff )
        nr_s2diff = np.append( nr_s2diff, s2_diff )
        nr_ff = np.append( nr_ff, ff )
      nr_mcdiffq1, nr_etdiffq1, nr_s1diffq1, nr_s2diffq1 = nr_mcdiff,\
                    nr_etdiff, nr_s1diff, nr_s2diff
      nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      # q = 2
      inkey = 'q2'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
      nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                  self.data.parameterbiases_vs_parameters(inkey=sim,\
                      chieff=chieff, approx=app, total_mass=total_mass)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_mcdiff = np.append( nr_mcdiff, mc_diff )
        nr_etdiff = np.append( nr_etdiff, eta_diff )
        nr_s1diff = np.append( nr_s1diff, s1_diff )
        nr_s2diff = np.append( nr_s2diff, s2_diff )
        nr_ff = np.append( nr_ff, ff )
      nr_mcdiffq2, nr_etdiffq2, nr_s1diffq2, nr_s2diffq2 = nr_mcdiff,\
                    nr_etdiff, nr_s1diff, nr_s2diff
      nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      # q = 3
      inkey = 'q3'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
      nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                  self.data.parameterbiases_vs_parameters(inkey=sim,\
                      chieff=chieff, approx=app, total_mass=total_mass)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_mcdiff = np.append( nr_mcdiff, mc_diff )
        nr_etdiff = np.append( nr_etdiff, eta_diff )
        nr_s1diff = np.append( nr_s1diff, s1_diff )
        nr_s2diff = np.append( nr_s2diff, s2_diff )
        nr_ff = np.append( nr_ff, ff )
      nr_mcdiffq3, nr_etdiffq3, nr_s1diffq3, nr_s2diffq3 = nr_mcdiff,\
                    nr_etdiff, nr_s1diff, nr_s2diff
      nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      #
      Xrow = [nr_spin1zq1, nr_spin1zq2, nr_spin1zq3]
      Yrow = [nr_spin2zq1, nr_spin2zq2, nr_spin2zq3]
      Zrow = [nr_massesq1, nr_massesq2, nr_massesq3]
      mcCrow=[nr_mcdiffq1, nr_mcdiffq2, nr_mcdiffq3]
      etCrow=[nr_etdiffq1, nr_etdiffq2, nr_etdiffq3]
      s1Crow=[nr_s1diffq1, nr_s1diffq2, nr_s1diffq3]
      s2Crow=[nr_s2diffq1, nr_s2diffq2, nr_s2diffq3]
      #
      Xs.append( Xrow )
      Ys.append( Yrow )
      Zs.append( Zrow )
      mcCs.append( mcCrow )
      etCs.append( etCrow )
      s1Cs.append( s1Crow )
      s2Cs.append( s2Crow )
      #
      titles.append( app[:-4] )
    ###################################
    # Now make the plots
    #print "With ", nr_spin1z, nr_spin2z, nr_masses, nr_ff
    if type(bounds) != np.ndarray and type(bounds) != list:
      if bounds:
        print "SHOULD NOT HAPPEN"
        bounds = np.array([-0.11, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.11])
    print "bounds before calling plotting function  = ", bounds
    if total_mass:
      make_scatter_plot3D_multrow( Xs, Ys, Zs, mcCs, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='M $(M_\odot)$',\
          clabel='(Recovered $M$ - Injected $M$) / Injected $M$',\
          title=titles,\
          logC=False,\
          bounds=bounds,\
          colormin=colormin, colormax=colormax,\
          savefig=self.plotdir+'/TotalMassError_TotalMass_Spin1z_Spin2z.%s'\
                                                  %(figtype))
    else:
      make_scatter_plot3D_multrow( Xs, Ys, Zs, mcCs, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='M $(M_\odot)$',\
          clabel='(Recovered $\mathcal{M}_c$ - Injected $\mathcal{M}_c$) / Injected $\mathcal{M}_c$',\
          title=titles,\
          logC=False,\
          bounds=bounds,\
          colormin=colormin, colormax=colormax,\
          savefig=self.plotdir+'/ChirpMassError_TotalMass_Spin1z_Spin2z.%s'\
                                                  %(figtype))
    #
    if type(bounds) != np.ndarray and type(bounds) != list:
      if bounds:
        print "SHOULD NOT HAPPEN"
        bounds = np.array([-0.23, -0.15, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10,\
                              0.15, 0.25, 0.33]) 
    make_scatter_plot3D_multrow( Xs, Ys, Zs, etCs, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='M $(M_\odot)$',\
          clabel='(Recovered $\eta$ - Injected $\eta$) / Injected $\eta$',\
          title=titles,\
          logC=False,\
          bounds=bounds,\
          colormin=colormin, colormax=colormax,\
          savefig=self.plotdir+'/EtaError_TotalMass_Spin1z_Spin2z.%s'\
                                                  %(figtype))
    #
    if type(bounds) != np.ndarray and type(bounds) != list:
      if bounds:
        print "SHOULD NOT HAPPEN"
        bounds = np.array([-0.5,-0.2,-0.1,-0.05,-0.02,0.02,0.05,0.1,0.2,0.5])
        if chieff:
          bounds = np.array([-0.15,-0.1,-0.05,-0.02,0.02,0.05,0.1,0.15])
    if chieff:
      #clabel='Recovered $\chi_\mathrm{eff}$ - Injected $\chi_\mathrm{eff}$'
      #clabel='Recovered $\chi_\mathrm{eff2PN}$ - Injected $\chi_\mathrm{eff2PN}$'
      clabel='Recovered $\chi_\mathrm{mw}$ - Injected $\chi_\mathrm{mw}$'
      #figtag='ChiEffPN'
      figtag='ChiMW'
    else:
      clabel='Recovered $\chi_1$ - Injected $\chi_1$'
      figtag='Chi1'
    make_scatter_plot3D_multrow( Xs, Ys, Zs, s1Cs, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='M $(M_\odot)$',\
          clabel=clabel, title=titles,\
          #clabel='$\Delta\chi_\mathrm{eff2PN}$', title=titles,\
          #clabel='$\Delta\chi_\mathrm{eff}$', title=titles,\
          #clabel='$\Delta\chi_\mathrm{mw}$', title=titles,\
          #clabel='$\Delta\chi_\mathrm{effD}$', title=titles,\
          logC=False,\
          bounds=bounds,\
          colormin=colormin, colormax=colormax,\
          savefig=self.plotdir+'/'+figtag+'Error_TotalMass_Spin1z_Spin2z.%s'\
                                                  %(figtype))
    #
    if type(bounds) != np.ndarray and type(bounds) != list:
      if bounds:
        print "SHOULD NOT HAPPEN"
	bounds = np.array([-0.5,-0.2,-0.1,-0.05,-0.02,0.02,0.05,0.1,0.2,0.5])
    make_scatter_plot3D_multrow( Xs, Ys, Zs, s2Cs, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='M $(M_\odot)$',\
          clabel='Recovered $\chi_2$ - Injected $\chi_2$', title=titles,\
          logC=False,\
          bounds=bounds,\
          colormin=colormin, colormax=colormax,\
          savefig=self.plotdir+'/Chi2Error_TotalMass_Spin1z_Spin2z.%s'\
                                                  %(figtype))
    #
    return
    #}}}
  #
  def plot_effectualness_contours_vs_spins(self, inkey=None,\
            logy=True, alpha=0.8, figtype='pdf'):
    #{{{
    try: import matplotlib.pyplot as plt
    except: return
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    #
    for idx, app in enumerate(self.ApproxList):
      # q = 1
      inkey = 'q1'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                          np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_ff = np.append( nr_ff, ff )
      nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      # q = 2
      inkey = 'q2'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                          np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_ff = np.append( nr_ff, ff )
      nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      # q = 3
      inkey = 'q3'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                          np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_ff = np.append( nr_ff, ff )
      nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      #
      # spin1 - spin2
      if self.verbose: print >>sys.stderr, "Making M-S1 plot for ", app
      print "With ", nr_spin1z, nr_spin2z, nr_masses, nr_ff
      make_contourf_mult(nr_spin1zq1, nr_spin2zq1, nr_ffq1, \
          nr_spin1zq2, nr_spin2zq2, nr_ffq2, \
          nr_spin1zq3, nr_spin2zq3, nr_ffq3, \
          alpha=alpha, xlabel='chi1', ylabel='chi2',\
          clabel='Fitting Factor', title=app[:-4],\
          savefig=self.plotdir+'/FF_TotalMass_Spin1z_Spin2z_%s.%s'\
                                                  %(app[:-4],figtype))
    return
    #}}}    
  def plot_effectualness_vs_single_parameter(self, logy=True, \
            ApproxList=[],\
            parameter='chieff', massmid=70., OneMinus=True, ChiNormalized=True,\
            onlyimr=True, bounds=None, azimuthal=30, alpha=0.8, figtype='pdf'):
    #{{{
    try:
      import matplotlib.pyplot as plt
      from pycbc.pnutils import mtotal_eta_to_mass1_mass2
    except:
      print "error import PyCBC modules / matplotlib. Cant make this plot."
      return
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    #
    # which approximants to plot ?
    if type(ApproxList) == list and len(ApproxList) != 0:
      approx_present = True
      for app in ApproxList:
        if app not in self.ApproxList: approx_present = False
      if not approx_present: ApproxList = self.ApproxList
    else: ApproxList = self.ApproxList
    # 
    #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
    #                    np.array([]), np.array([]), np.array([]), np.array([])
    Xs, Ys, Zs, Cs = [], [], [], []
    titles = []
    for idx, app in enumerate(ApproxList):
      if onlyimr:
        if 'Taylor' in app: continue
      else:
        if 'Taylor' not in app: continue
      # q = 1
      inkey = 'q1'
      inq = 1.
      ineta = inq / (1. + inq)**2
      nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {},{},{},{},{}
      nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                        np.array([])
      #
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        idx['high']= np.where( ff == ff.max() )[0][0]
        idx['low'] = np.where( ff == ff.min() )[0][0]
        if massmid > 0: idx['mid'] = np.where( masses == massmid )[0][0]
        else: idx['mid'] = np.where( masses == masses.min() )[0][0]
        for kk in ['low', 'mid', 'high']:
          nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
          nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
          nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
          nr_ff[kk]     = np.append(nr_ff[kk],     ff[idx[kk]])
      nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      nr_mass1q1, nr_mass2q1, nr_chieffq1 = {}, {}, {}
      for kk in ['low', 'mid', 'high']:
        nr_mass1q1[kk], nr_mass2q1[kk] = \
            mtotal_eta_to_mass1_mass2(nr_masses[kk],\
                                      ineta * np.ones(len(nr_masses[kk])))
        nr_chieffq1[kk] = spins_to_PNeffective_spin(\
              nr_mass1q1[kk], nr_mass2q1[kk], nr_spin1zq1[kk], nr_spin2zq1[kk])
        if OneMinus: nr_ffq1[kk] = 1. - nr_ffq1[kk] # Convert FF to 1 - FF
      # q = 2
      inkey = 'q2'
      inq = 2.
      ineta = inq / (1. + inq)**2
      nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {},{},{},{},{}
      nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                        np.array([])
      #
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        idx['high']= np.where( ff == ff.max() )[0][0]
        idx['low'] = np.where( ff == ff.min() )[0][0]
        if massmid > 0: idx['mid'] = np.where( masses == massmid )[0][0]
        else: idx['mid'] = np.where( masses == masses.min() )[0][0]
        for kk in ['low', 'mid', 'high']:
          nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
          nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
          nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
          nr_ff[kk]     = np.append(nr_ff[kk],     ff[idx[kk]])
      nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      nr_mass1q2, nr_mass2q2, nr_chieffq2 = {}, {}, {}
      for kk in ['low', 'mid', 'high']:
        nr_mass1q2[kk], nr_mass2q2[kk] = \
            mtotal_eta_to_mass1_mass2(nr_masses[kk],\
                                      ineta * np.ones(len(nr_masses[kk])))
        nr_chieffq2[kk] = spins_to_PNeffective_spin(\
              nr_mass1q2[kk], nr_mass2q2[kk], nr_spin1zq2[kk], nr_spin2zq2[kk])
        if OneMinus: nr_ffq2[kk] = 1. - nr_ffq2[kk] # Convert FF to 1 - FF
      # q = 3
      inkey = 'q3'
      inq = 3.
      ineta = inq / (1. + inq)**2
      nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {},{},{},{},{}
      nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                        np.array([])
      #
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(inkey=sim, approx=app)
        idx['high']= np.where( ff == ff.max() )[0][0]
        idx['low'] = np.where( ff == ff.min() )[0][0]
        if massmid > 0: idx['mid'] = np.where( masses == massmid )[0][0]
        else: idx['mid'] = np.where( masses == masses.min() )[0][0]
        for kk in ['low', 'mid', 'high']:
          nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
          nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
          nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
          nr_ff[kk]     = np.append(nr_ff[kk],     ff[idx[kk]])
      nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      nr_mass1q3, nr_mass2q3, nr_chieffq3 = {}, {}, {}
      for kk in ['low', 'mid', 'high']:
        nr_mass1q3[kk], nr_mass2q3[kk] = \
            mtotal_eta_to_mass1_mass2(nr_masses[kk],\
                                      ineta * np.ones(len(nr_masses[kk])))
        nr_chieffq3[kk] = spins_to_PNeffective_spin(\
              nr_mass1q3[kk], nr_mass2q3[kk], nr_spin1zq3[kk], nr_spin2zq3[kk])
        if OneMinus: nr_ffq3[kk] = 1. - nr_ffq3[kk] # Convert FF to 1 - FF
      #
      if ChiNormalized:
        etaq1, etaq2, etaq3 = 1./4., 2./9., 3./16.
        Xrow = [nr_chieffq1['mid']/(1. - 76.*etaq1/113.),\
                nr_chieffq2['mid']/(1. - 76.*etaq2/113.),\
                nr_chieffq3['mid']/(1. - 76.*etaq3/113.)]
      else: Xrow = [nr_chieffq1['mid'], nr_chieffq2['mid'], nr_chieffq3['mid']]
      Yrow = [nr_ffq1['mid'], nr_ffq2['mid'], nr_ffq3['mid']]
      Yerror = [np.append( [nr_ffq1['mid'] - nr_ffq1['low']],\
                          [nr_ffq1['high'] - nr_ffq1['mid']], axis=0),\
                np.append( [nr_ffq2['mid'] - nr_ffq2['low']],\
                          [nr_ffq2['high'] - nr_ffq2['mid']], axis=0),\
                np.append( [nr_ffq3['mid'] - nr_ffq3['low']],\
                          [nr_ffq3['high'] - nr_ffq3['mid']], axis=0)]
      Xerror = None      
      #
      ###################################
      # Now make the plots
      if OneMinus:
        ymin, ymax = 1.e-3, 1.e-0
        ylabel = '1 - Fitting Factor'
        logy = True
        legendplacement = 'best'#'upper left'
        nameprefix = 'MM'
      else:
        ymin, ymax = 0.92, 1.
        ylabel = 'Fitting Factor' 
        logy = False
        legendplacement = 'lower left'
        nameprefix = 'FF'
      if ChiNormalized: xlabel = '$\chi_\mathrm{eff}/(1-\\frac{76}{113}\eta)$'
      else: xlabel = '$\chi_\mathrm{eff}$'
      print ymin, ymax, ylabel, logy, legendplacement, nameprefix
      make_2Dplot_errorbars( Xrow, Yrow, Xerrs=Xerror, Yerrs=Yerror,\
         xlabel=xlabel, ylabel=ylabel, title=app[:-4],\
         logy=logy, ymin=ymin, ymax=ymax, labels=['q=1','q=2','q=3'],\
         legendplacement=legendplacement,\
         savefig=self.plotdir+'/'+nameprefix+'vsChiEff_'+app[:-4]+'.'+figtype )
    return
    #}}}    
  def plot_parameterbias_vs_single_parameter(self, logy=False, \
            parameter='chieff', massmid=70., OneMinus=False, \
            ChiNormalized=True,\
            ApproxList=[], biasparameter='ChirpMass', ylabel='',\
            ylims=[], \
            onlyimr=True, bounds=None, azimuthal=30, alpha=0.8, figtype='pdf'):
    #{{{
    print "trying to make pb vs p plot"
    try:
      import matplotlib.pyplot as plt
      from pycbc.pnutils import mtotal_eta_to_mass1_mass2
    except:
      print "error import PyCBC modules / matplotlib"
      return
    #
    # which approximants to plot ?
    if type(ApproxList) == list and len(ApproxList) != 0:
      approx_present = True
      for app in ApproxList:
        if app not in self.ApproxList: approx_present = False
      if not approx_present: ApproxList = self.ApproxList
    else: ApproxList = self.ApproxList
    # 
    # Set the chi-effective flag
    if 'ChiEff' in biasparameter: chieffflag = True
    else: chieffflag = False
    #
    # Set the total-mass flag
    if 'TotalMass' in biasparameter: mtotalflag = True
    else: mtotalflag = False
    #
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
    #                    np.array([]), np.array([]), np.array([]), np.array([])
    titles = []
    for idx, app in enumerate(ApproxList):
      if onlyimr:
        if 'Taylor' in app: continue
      else:
        if 'Taylor' not in app: continue
      print "making pb vs p plot for %s" % app
      #################################################
      # q = 1
      inkey = 'q1'
      inq = 1.
      ineta = inq / (1. + inq)**2
      nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {},{},{},{},{}
      nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                        np.array([])
      #
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff = \
                    self.data.parameterbiases_vs_parameters(inkey=sim, \
                    approx=app, chieff=chieffflag, total_mass=mtotalflag)
        # Here replace ff with the pb (parameter bias)
        if 'ChirpMass' in biasparameter or 'TotalMass' in biasparameter:
          ff = mc_diff
        elif 'ChiEff' in biasparameter: ff = s1_diff
        elif 'Eta' in biasparameter: ff = eta_diff
        #
        idx['high']= np.where( ff == ff.max() )[0][0]
        idx['low'] = np.where( ff == ff.min() )[0][0]
        if massmid > 0: idx['mid'] = np.where( masses == massmid )[0][0]
        else: idx['mid'] = np.where( masses == masses.min() )[0][0]
        for kk in ['low', 'mid', 'high']:
          nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
          nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
          nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
          nr_ff[kk]     = np.append(nr_ff[kk],     ff[idx[kk]])
      nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      nr_mass1q1, nr_mass2q1, nr_chieffq1 = {}, {}, {}
      for kk in ['low', 'mid', 'high']:
        nr_mass1q1[kk], nr_mass2q1[kk] = \
            mtotal_eta_to_mass1_mass2(nr_masses[kk],\
                                      ineta * np.ones(len(nr_masses[kk])))
        nr_chieffq1[kk] = spins_to_PNeffective_spin(\
              nr_mass1q1[kk], nr_mass2q1[kk], nr_spin1zq1[kk], nr_spin2zq1[kk])
      #################################################
      # q = 2
      inkey = 'q2'
      inq = 2.
      ineta = inq / (1. + inq)**2
      nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {},{},{},{},{}
      nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                        np.array([])
      #
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff = \
                    self.data.parameterbiases_vs_parameters(inkey=sim, \
                    approx=app, chieff=chieffflag, total_mass=mtotalflag)
        # Here replace ff with the pb (parameter bias)
        if 'ChirpMass' in biasparameter or 'TotalMass' in biasparameter:
          ff = mc_diff
        elif 'ChiEff' in biasparameter: ff = s1_diff
        elif 'Eta' in biasparameter: ff = eta_diff
        #
        idx['high']= np.where( ff == ff.max() )[0][0]
        idx['low'] = np.where( ff == ff.min() )[0][0]
        print "PK: masses = ", masses
        if massmid > 0: idx['mid'] = np.where( masses == massmid )[0][0]
        else: idx['mid'] = np.where( masses == masses.min() )[0][0]
        for kk in ['low', 'mid', 'high']:
          nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
          nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
          nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
          nr_ff[kk]     = np.append(nr_ff[kk],     ff[idx[kk]])
      nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      nr_mass1q2, nr_mass2q2, nr_chieffq2 = {}, {}, {}
      for kk in ['low', 'mid', 'high']:
        nr_mass1q2[kk], nr_mass2q2[kk] = \
            mtotal_eta_to_mass1_mass2(nr_masses[kk],\
                                      ineta * np.ones(len(nr_masses[kk])))
        nr_chieffq2[kk] = spins_to_PNeffective_spin(\
              nr_mass1q2[kk], nr_mass2q2[kk], nr_spin1zq2[kk], nr_spin2zq2[kk])
      #################################################
      # q = 3
      inkey = 'q3'
      inq = 3.
      ineta = inq / (1. + inq)**2
      nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {},{},{},{},{}
      nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                        np.array([])
      #
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff = \
                    self.data.parameterbiases_vs_parameters(inkey=sim, \
                    approx=app, chieff=chieffflag, total_mass=mtotalflag)
        # Here replace ff with the pb (parameter bias)
        if 'ChirpMass' in biasparameter or 'TotalMass' in biasparameter:
          ff = mc_diff
        elif 'ChiEff' in biasparameter: ff = s1_diff
        elif 'Eta' in biasparameter: ff = eta_diff
        #
        idx['high']= np.where( ff == ff.max() )[0][0]
        idx['low'] = np.where( ff == ff.min() )[0][0]
        if massmid > 0: idx['mid'] = np.where( masses == massmid )[0][0]
        else: idx['mid'] = np.where( masses == masses.min() )[0][0]
        for kk in ['low', 'mid', 'high']:
          nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
          nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
          nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
          nr_ff[kk]     = np.append(nr_ff[kk],     ff[idx[kk]])
      nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      nr_mass1q3, nr_mass2q3, nr_chieffq3 = {}, {}, {}
      for kk in ['low', 'mid', 'high']:
        nr_mass1q3[kk], nr_mass2q3[kk] = \
            mtotal_eta_to_mass1_mass2(nr_masses[kk],\
                                      ineta * np.ones(len(nr_masses[kk])))
        nr_chieffq3[kk] = spins_to_PNeffective_spin(\
              nr_mass1q3[kk], nr_mass2q3[kk], nr_spin1zq3[kk], nr_spin2zq3[kk])
      #################################################
      #
      Yrow = [nr_ffq1['mid'], nr_ffq2['mid'], nr_ffq3['mid']]
      Yerror = [np.append( [nr_ffq1['mid'] - nr_ffq1['low']],\
                          [nr_ffq1['high'] - nr_ffq1['mid']], axis=0),\
                np.append( [nr_ffq2['mid'] - nr_ffq2['low']],\
                          [nr_ffq2['high'] - nr_ffq2['mid']], axis=0),\
                np.append( [nr_ffq3['mid'] - nr_ffq3['low']],\
                          [nr_ffq3['high'] - nr_ffq3['mid']], axis=0)]
      if ChiNormalized:
        etaq1, etaq2, etaq3 = 1./4., 2./9., 3./16.
        Xrow = [nr_chieffq1['mid']/(1. - 76.*etaq1/113.),\
                nr_chieffq2['mid']/(1. - 76.*etaq2/113.),\
                nr_chieffq3['mid']/(1. - 76.*etaq3/113.)]
        if chieffflag:
          Yrow = [Yrow[0]/(1. - 76.*etaq1/113.),\
                  Yrow[1]/(1. - 76.*etaq2/113.),\
                  Yrow[2]/(1. - 76.*etaq3/113.)]
          Yerror = [Yerror[0]/(1. - 76.*etaq1/113.),\
                    Yerror[1]/(1. - 76.*etaq2/113.),\
                    Yerror[2]/(1. - 76.*etaq3/113.)]
      else: Xrow = [nr_chieffq1['mid'], nr_chieffq2['mid'], nr_chieffq3['mid']]
      Xerror = None      
      #
      ###################################
      # Now make the plots
      if OneMinus: nameprefix = ''
      else:
        ymin, ymax = None, None
        if len(ylims)==2: ymin, ymax = ylims
        #ymin, ymax = np.array(Yrow).min(), np.array(Yrow).max()
        #ylabel = 'Fitting Factor'
        logy = False
        legendplacement = 'best'
        nameprefix = biasparameter+'Bias_'
      if ChiNormalized: xlabel = '$\chi_\mathrm{eff}/(1-\\frac{76}{113}\eta)$'
      else: xlabel = '$\chi_\mathrm{eff}$'
      print ymin, ymax, ylabel, logy, legendplacement, nameprefix
      make_2Dplot_errorbars( Xrow, Yrow, Xerrs=Xerror, Yerrs=Yerror,\
         xlabel=xlabel, ylabel=ylabel, title=app[:-4],\
         logy=logy, \
         ymin=ymin, ymax=ymax, \
         labels=['q=1','q=2','q=3'],\
         legendplacement=legendplacement,\
         savefig=self.plotdir+'/'+nameprefix+'vsChiEff_'+app[:-4]+'.'+figtype )
    return
    #}}}    
  def plot_recoveredparameter_vs_single_parameter(self, logy=False, \
            parameter='chieff', massmid=70., OneMinus=False, \
            ChiNormalized=True,\
            ApproxList=[], biasparameter='ChirpMass', ylabel='',\
            ylims=[], \
            onlyimr=True, bounds=None, azimuthal=30, alpha=0.8, figtype='pdf'):
    #{{{
    try:
      import matplotlib.pyplot as plt
      from pycbc.pnutils import mtotal_eta_to_mass1_mass2
    except:
      print "error import PyCBC modules / matplotlib"
      return
    #
    # which approximants to plot ?
    if type(ApproxList) == list and len(ApproxList) != 0:
      approx_present = True
      for app in ApproxList:
        if app not in self.ApproxList: approx_present = False
      if not approx_present: ApproxList = self.ApproxList
    else: ApproxList = self.ApproxList
    # 
    # Set the chi-effective flag
    if 'ChiEff' in biasparameter: chieffflag = True
    else: chieffflag = False
    #
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
    #                    np.array([]), np.array([]), np.array([]), np.array([])
    titles = []
    for idx, app in enumerate(ApproxList):
      if onlyimr:
        if 'Taylor' in app: continue
      else:
        if 'Taylor' not in app: continue
      #################################################
      # q = 1
      inkey = 'q1'
      inq = 1.
      ineta = inq / (1. + inq)**2
      nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {},{},{},{},{}
      nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                        np.array([])
      #
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff = \
                    self.data.parameterbiases_vs_parameters(inkey=sim, \
                    approx=app, chieff=chieffflag)
        # Here replace ff with the pb (parameter bias)
        if 'ChirpMass' in biasparameter:
          inj_mc = masses * (nr_q / (1. + nr_q)**2)
          ff = inj_mc * (1. + mc_diff)
        elif 'ChiEff' in biasparameter: ff = nr_s1 + s1_diff
        elif 'Eta' in biasparameter or 'Q' in biasparameter:
          inj_et = nr_q / (1. + nr_q)**2
          ff = inj_et * (1. + eta_diff)
          ff = eta_to_q( ff )
        #
        idx['high']= np.where( ff == ff.max() )[0][0]
        idx['low'] = np.where( ff == ff.min() )[0][0]
        if massmid > 0: idx['mid'] = np.where( masses == massmid )[0][0]
        else: idx['mid'] = np.where( masses == masses.min() )[0][0]
        for kk in ['low', 'mid', 'high']:
          nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
          nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
          nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
          nr_ff[kk]     = np.append(nr_ff[kk],     ff[idx[kk]])
      nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      nr_mass1q1, nr_mass2q1, nr_chieffq1 = {}, {}, {}
      for kk in ['low', 'mid', 'high']:
        nr_mass1q1[kk], nr_mass2q1[kk] = \
            mtotal_eta_to_mass1_mass2(nr_masses[kk],\
                                      ineta * np.ones(len(nr_masses[kk])))
        nr_chieffq1[kk] = spins_to_PNeffective_spin(\
              nr_mass1q1[kk], nr_mass2q1[kk], nr_spin1zq1[kk], nr_spin2zq1[kk])
      #################################################
      # q = 2
      inkey = 'q2'
      inq = 2.
      ineta = inq / (1. + inq)**2
      nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {},{},{},{},{}
      nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                        np.array([])
      #
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff = \
                    self.data.parameterbiases_vs_parameters(inkey=sim, \
                    approx=app, chieff=chieffflag)
        # Here replace ff with the pb (parameter bias)
        if 'ChirpMass' in biasparameter:
          inj_mc = masses * (nr_q / (1. + nr_q)**2)
          ff = inj_mc * (1. + mc_diff)
        elif 'ChiEff' in biasparameter: ff = nr_s1 + s1_diff
        elif 'Eta' in biasparameter or 'Q' in biasparameter:
          inj_et = nr_q / (1. + nr_q)**2
          ff = inj_et * (1. + eta_diff)
          ff = eta_to_q( ff )
        #
        idx['high']= np.where( ff == ff.max() )[0][0]
        idx['low'] = np.where( ff == ff.min() )[0][0]
        print "PK: masses = ", masses
        if massmid > 0: idx['mid'] = np.where( masses == massmid )[0][0]
        else: idx['mid'] = np.where( masses == masses.min() )[0][0]
        for kk in ['low', 'mid', 'high']:
          nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
          nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
          nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
          nr_ff[kk]     = np.append(nr_ff[kk],     ff[idx[kk]])
      nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      nr_mass1q2, nr_mass2q2, nr_chieffq2 = {}, {}, {}
      for kk in ['low', 'mid', 'high']:
        nr_mass1q2[kk], nr_mass2q2[kk] = \
            mtotal_eta_to_mass1_mass2(nr_masses[kk],\
                                      ineta * np.ones(len(nr_masses[kk])))
        nr_chieffq2[kk] = spins_to_PNeffective_spin(\
              nr_mass1q2[kk], nr_mass2q2[kk], nr_spin1zq2[kk], nr_spin2zq2[kk])
      #################################################
      # q = 3
      inkey = 'q3'
      inq = 3.
      ineta = inq / (1. + inq)**2
      nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {},{},{},{},{}
      nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                          np.array([]), np.array([])
      nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                        np.array([])
      #
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff = \
                    self.data.parameterbiases_vs_parameters(inkey=sim, \
                    approx=app, chieff=chieffflag)
        # Here replace ff with the pb (parameter bias)
        if 'ChirpMass' in biasparameter:
          inj_mc = masses * (nr_q / (1. + nr_q)**2)
          ff = inj_mc * (1. + mc_diff)
        elif 'ChiEff' in biasparameter: ff = nr_s1 + s1_diff
        elif 'Eta' in biasparameter or 'Q' in biasparameter:
          inj_et = nr_q / (1. + nr_q)**2
          ff = inj_et * (1. + eta_diff)
          ff = eta_to_q( ff )
        #
        idx['high']= np.where( ff == ff.max() )[0][0]
        idx['low'] = np.where( ff == ff.min() )[0][0]
        if massmid > 0: idx['mid'] = np.where( masses == massmid )[0][0]
        else: idx['mid'] = np.where( masses == masses.min() )[0][0]
        for kk in ['low', 'mid', 'high']:
          nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
          nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
          nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
          nr_ff[kk]     = np.append(nr_ff[kk],     ff[idx[kk]])
      nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      nr_mass1q3, nr_mass2q3, nr_chieffq3 = {}, {}, {}
      for kk in ['low', 'mid', 'high']:
        nr_mass1q3[kk], nr_mass2q3[kk] = \
            mtotal_eta_to_mass1_mass2(nr_masses[kk],\
                                      ineta * np.ones(len(nr_masses[kk])))
        nr_chieffq3[kk] = spins_to_PNeffective_spin(\
              nr_mass1q3[kk], nr_mass2q3[kk], nr_spin1zq3[kk], nr_spin2zq3[kk])
      #################################################
      #
      Yrow = [nr_ffq1['mid'], nr_ffq2['mid'], nr_ffq3['mid']]
      Yerror = [np.append( [nr_ffq1['mid'] - nr_ffq1['low']],\
                          [nr_ffq1['high'] - nr_ffq1['mid']], axis=0),\
                np.append( [nr_ffq2['mid'] - nr_ffq2['low']],\
                          [nr_ffq2['high'] - nr_ffq2['mid']], axis=0),\
                np.append( [nr_ffq3['mid'] - nr_ffq3['low']],\
                          [nr_ffq3['high'] - nr_ffq3['mid']], axis=0)]
      if ChiNormalized:
        etaq1, etaq2, etaq3 = 1./4., 2./9., 3./16.
        Xrow = [nr_chieffq1['mid']/(1. - 76.*etaq1/113.),\
                nr_chieffq2['mid']/(1. - 76.*etaq2/113.),\
                nr_chieffq3['mid']/(1. - 76.*etaq3/113.)]
        if chieffflag:
          Yrow = [Yrow[0]/(1. - 76.*etaq1/113.),\
                  Yrow[1]/(1. - 76.*etaq2/113.),\
                  Yrow[2]/(1. - 76.*etaq3/113.)]
          Yerror = [Yerror[0]/(1. - 76.*etaq1/113.),\
                    Yerror[1]/(1. - 76.*etaq2/113.),\
                    Yerror[2]/(1. - 76.*etaq3/113.)]
      else: Xrow = [nr_chieffq1['mid'], nr_chieffq2['mid'], nr_chieffq3['mid']]
      Xerror = None      
      #
      ###################################
      # Now make the plots
      if OneMinus: pass
      else:
        ymin, ymax = None, None
        if len(ylims)==2: ymin, ymax = ylims
        #ymin, ymax = np.array(Yrow).min(), np.array(Yrow).max()
        #ylabel = 'Fitting Factor'
        logy = False
        legendplacement = 'best'
        nameprefix = biasparameter+'Rec_'
      if ChiNormalized: xlabel = '$\chi_\mathrm{eff}/(1-\\frac{76}{113}\eta)$'
      else: xlabel = '$\chi_\mathrm{eff}$'
      print ymin, ymax, ylabel, logy, legendplacement, nameprefix
      make_2Dplot_errorbars( Xrow, Yrow, Xerrs=Xerror, Yerrs=Yerror,\
         xlabel=xlabel, ylabel=ylabel, title=app[:-4],\
         logy=logy, \
         ymin=ymin, ymax=ymax, \
         labels=['q=1','q=2','q=3'],\
         legendplacement=legendplacement,\
         savefig=self.plotdir+'/'+nameprefix+'vsChiEff_'+app[:-4]+'.'+figtype )
    return
    #}}}    
  def plot_recoveredparameters_vs_parameters_multrow(self, \
            logy=True, elevation=30,\
            ApproxList=[], onlyimr=True, chieff=False, \
            bounds=True, azimuthal=30, alpha=0.8, figtype='pdf'):
    #{{{
    try: import matplotlib.pyplot as plt
    except: return
    #
    # which approximants to plot ?
    if type(ApproxList) == list and len(ApproxList) != 0:
      approx_present = True
      for app in ApproxList:
        if app not in self.ApproxList: approx_present = False
      if not approx_present: ApproxList = self.ApproxList
    else: ApproxList = self.ApproxList
    ###############################
    # Read data
    if self.data == None: self.read_data_from_all_files()
    all_sims = self.data.data.keys()
    #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
    #                    np.array([]), np.array([]), np.array([]), np.array([])
    Xs, Ys, Zs, mcCs, etCs, qCs, s1Cs, s2Cs = [], [], [], [], [], [], [], []
    titles = []
    for idx, app in enumerate(ApproxList):
      if onlyimr:
        if 'Taylor' in app: continue
      # q = 1
      inkey = 'q1'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
      nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                  self.data.parameterbiases_vs_parameters(inkey=sim,\
                      chieff=chieff, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_mcdiff = np.append( nr_mcdiff, mc_diff )
        nr_etdiff = np.append( nr_etdiff, eta_diff )
        nr_s1diff = np.append( nr_s1diff, s1_diff )
        nr_s2diff = np.append( nr_s2diff, s2_diff )
        nr_ff = np.append( nr_ff, ff )
      nr_mcdiffq1, nr_etdiffq1, nr_s1diffq1, nr_s2diffq1 = nr_mcdiff,\
                    nr_etdiff, nr_s1diff, nr_s2diff
      nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      # q = 2
      inkey = 'q2'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
      nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                  self.data.parameterbiases_vs_parameters(inkey=sim,\
                      chieff=chieff, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_mcdiff = np.append( nr_mcdiff, mc_diff )
        nr_etdiff = np.append( nr_etdiff, eta_diff )
        nr_s1diff = np.append( nr_s1diff, s1_diff )
        nr_s2diff = np.append( nr_s2diff, s2_diff )
        nr_ff = np.append( nr_ff, ff )
      nr_mcdiffq2, nr_etdiffq2, nr_s1diffq2, nr_s2diffq2 = nr_mcdiff,\
                    nr_etdiff, nr_s1diff, nr_s2diff
      nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      # q = 3
      inkey = 'q3'
      nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
      nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([]),\
                      np.array([]), np.array([]), np.array([]), np.array([])
      for sim in all_sims:
        if inkey is not None and inkey not in str(sim): continue
        masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                  self.data.parameterbiases_vs_parameters(inkey=sim,\
                      chieff=chieff, approx=app)
        nr_masses = np.append( nr_masses, masses )
        nr_massratios = np.append( nr_massratios, nr_q )
        nr_spin1z = np.append( nr_spin1z, nr_s1 )
        nr_spin2z = np.append( nr_spin2z, nr_s2 )
        nr_mcdiff = np.append( nr_mcdiff, mc_diff )
        nr_etdiff = np.append( nr_etdiff, eta_diff )
        nr_s1diff = np.append( nr_s1diff, s1_diff )
        nr_s2diff = np.append( nr_s2diff, s2_diff )
        nr_ff = np.append( nr_ff, ff )
      nr_mcdiffq3, nr_etdiffq3, nr_s1diffq3, nr_s2diffq3 = nr_mcdiff,\
                    nr_etdiff, nr_s1diff, nr_s2diff
      nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                              nr_masses, nr_ff
      #
      Xrow = [nr_spin1zq1, nr_spin1zq2, nr_spin1zq3]
      Yrow = [nr_spin2zq1, nr_spin2zq2, nr_spin2zq3]
      Zrow = [nr_massesq1, nr_massesq2, nr_massesq3]
      mcCrow=[nr_mcdiffq1, nr_mcdiffq2, nr_mcdiffq3]
      etCrow=[nr_etdiffq1, nr_etdiffq2, nr_etdiffq3]
      s1Crow=[nr_s1diffq1, nr_s1diffq2, nr_s1diffq3]
      s2Crow=[nr_s2diffq1, nr_s2diffq2, nr_s2diffq3]
      
      etaq1, etaq2, etaq3 = 1./4., 2./9., 3./16.
      inj_mc = [nr_massesq1 * etaq1**0.6, nr_massesq2 * etaq2**0.6,\
                nr_massesq3 * etaq3**0.6]
      rec_mc = []
      for i in range(3): rec_mc.append( inj_mc[i] * (1. + mcCrow[i]) )
      
      rec_et = []
      inj_et = [etaq1, etaq2, etaq3]
      for i in range(3): rec_et.append( inj_et[i] * (1. + etCrow[i]) )
     
      rec_q = eta_to_q( rec_et )

      rec_s1 = []
      for i in range(3): rec_s1.append( Xrow[i] + s1Crow[i] )

      rec_s2 = []
      for i in range(3): rec_s2.append( Yrow[i] + s2Crow[i] )
      
      newmcCrow = rec_mc
      newetCrow = rec_et
      news1Crow = rec_s1
      news2Crow = rec_s2
      newqCrow  = rec_q

      #
      Xs.append( Xrow )
      Ys.append( Yrow )
      Zs.append( Zrow )
      mcCs.append( newmcCrow )
      etCs.append( newetCrow )
      qCs.append( newqCrow )
      s1Cs.append( news1Crow )
      s2Cs.append( news2Crow )
      #
      titles.append( app[:-4] )
    ###################################
    # Now make the plots
    #print "With ", nr_spin1z, nr_spin2z, nr_masses, nr_ff
    if type(bounds) != np.ndarray and type(bounds) != list:
      if bounds:
        print "SHOULD NOT HAPPEN"
        bounds = np.array([-0.05,-0.03,-0.01,-0.005,0.005,0.01,0.03,0.05])
    print "bounds before calling plotting function  = ", bounds
    make_scatter_plot3D_multrow( Xs, Ys, Zs, mcCs, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='Total Mass $(M_\odot)$',\
          clabel='Recovered $\mathcal{M}_c$',\
          title=titles,\
          logC=False,\
          bounds=bounds,\
          savefig=self.plotdir+'/ChirpMassRec_TotalMass_Spin1z_Spin2z.%s'\
                                                  %(figtype))
    #
    if type(bounds) != np.ndarray and type(bounds) != list:
      if bounds:
        print "SHOULD NOT HAPPEN"
	bounds = np.array([-0.5,-0.2,-0.1,-0.05,-0.02,0.02,0.05,0.1,0.2,0.5])
    make_scatter_plot3D_multrow( Xs, Ys, Zs, etCs, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='Total Mass $(M_\odot)$',\
          clabel='Recovered $\eta$',\
          title=titles,\
          logC=False,\
          bounds=bounds,\
          savefig=self.plotdir+'/EtaRec_TotalMass_Spin1z_Spin2z.%s'\
                                                  %(figtype))
    #
    if type(bounds) != np.ndarray and type(bounds) != list:
      if bounds:
        print "SHOULD NOT HAPPEN"
	bounds = np.array([-0.5,-0.2,-0.1,-0.05,-0.02,0.02,0.05,0.1,0.2,0.5])
    make_scatter_plot3D_multrow( Xs, Ys, Zs, qCs, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='Total Mass $(M_\odot)$',\
          clabel='Recovered $q$',\
          title=titles,\
          logC=False,\
          bounds=bounds,\
          savefig=self.plotdir+'/QRec_TotalMass_Spin1z_Spin2z.%s'\
                                                  %(figtype))
    #
    if type(bounds) != np.ndarray and type(bounds) != list:
      if bounds:
        print "SHOULD NOT HAPPEN"
	bounds = np.array([-0.5,-0.2,-0.1,-0.05,-0.02,0.02,0.05,0.1,0.2,0.5])
    if chieff:
      clabel='Recovered $\chi_\mathrm{eff}$'
      figtag='ChiEff'
    else:
      clabel='Recovered $\chi_1$'
      figtag='Chi1'
    make_scatter_plot3D_multrow( Xs, Ys, Zs, s1Cs, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='Total Mass $(M_\odot)$',\
          clabel=clabel, title=titles,\
          #clabel='$\Delta\chi_\mathrm{eff2PN}$', title=titles,\
          #clabel='$\Delta\chi_\mathrm{eff}$', title=titles,\
          #clabel='$\Delta\chi_\mathrm{mw}$', title=titles,\
          #clabel='$\Delta\chi_\mathrm{effD}$', title=titles,\
          logC=False,\
          bounds=bounds,\
          savefig=self.plotdir+'/'+figtag+'Rec_TotalMass_Spin1z_Spin2z.%s'\
                                                  %(figtype))
    #
    if type(bounds) != np.ndarray and type(bounds) != list:
      if bounds:
        print "SHOULD NOT HAPPEN"
	bounds = np.array([-0.5,-0.2,-0.1,-0.05,-0.02,0.02,0.05,0.1,0.2,0.5])
    make_scatter_plot3D_multrow( Xs, Ys, Zs, s2Cs, \
          elevation=elevation, azimuthal=azimuthal, alpha=alpha, \
          xlabel='$\chi_1$', ylabel='$\chi_2$',\
          zlabel='Total Mass $(M_\odot)$',\
          clabel='Recovered $\chi_2$', title=titles,\
          logC=False,\
          bounds=bounds,\
          savefig=self.plotdir+'/Chi2Rec_TotalMass_Spin1z_Spin2z.%s'\
                                                  %(figtype))
    #
    return
    #}}}    
  #}}}

def insert_min_max_into_array(arr, low, high):
  #{{{
  # Assume an ordered array is passed. Insert min and max and force that
  if low > arr.max() or high < arr.min(): return np.array([low, high])
  new_arr = arr
  mask = new_arr > low
  new_arr = np.append( low, new_arr[mask] )
  mask = new_arr < high
  new_arr = np.append( new_arr[mask], high )
  print "MAX,MIN inserted bounds = ", arr, " -> ", new_arr
  return new_arr
  #}}}

def make_2Dplot_errorbars(Xs, Ys, Xerrs=None, Yerrs=None,\
                  xlabel='', ylabel='', title='',\
                  logy=True, xmin=-0.9, xmax=0.9, ymin=None, ymax=None,\
                  addlines=True, legendplacement='best',\
                  labels=None, fmts=['bs','ko','r^'], savefig='plots/plot.png'):
  #{{{
  if len(Xs) != len(Ys):
    raise IOError("Length of lists to be plotted not the same")
  nrows, ncols = 1,1#np.shape(Xs)
  ncurves = len(Xs)
  print "No of rows = %d, columns = %d, curves = %d" % (nrows, ncols, ncurves)
  if labels == None: 
    labels = range(ncurves)
    for idx in range(len(labels)): labels[idx] = str( labels[idx] )
  #
  gmean = (5.**0.5-1)*0.5
  fig = plt.figure(int(1e7 * np.random.random()), figsize=(4*ncols/gmean,4*nrows))
  nplot = 0
  ax = fig.add_subplot(nrows, ncols, nplot)
  #
  pcolor = ['b', 'r', 'k']
  for curveid in range(3):
    print "Adding curve {}".format(curveid)
    X, Y, Xerr, Yerr = Xs[curveid], Ys[curveid], None, None
    if Xerrs is not None: Xerr = Xerrs[curveid]
    if Yerrs is not None: Yerr = Yerrs[curveid]
    # Make one marker hollow
    if curveid != 1:
      ax.errorbar( X, Y, yerr=Yerr, xerr=Xerr, \
                mfc=pcolor[curveid], mec=pcolor[curveid], \
                color=pcolor[curveid], ecolor=pcolor[curveid], \
                fmt=fmts[curveid], elinewidth=2, label=labels[curveid])
    else:
      ax.errorbar( X, Y, yerr=Yerr, xerr=Xerr, \
                mfc='none', mec=pcolor[curveid], \
                color=pcolor[curveid], ecolor=pcolor[curveid], \
                fmt=fmts[curveid], elinewidth=2, label=labels[curveid])
    
    ax.hold(True)
    if addlines:
      ax.plot([-1,1], [curveid+1, curveid+1], pcolor[curveid]+'--', lw=1.6)
      ax.hold(True)
  if logy: ax.set_yscale('log')
  ax.grid()
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_xlim(xmin, xmax)
  if ymin is not None and ymax is not None: ax.set_ylim(ymin, ymax)
  ax.set_title(title)
  ax.legend(ncol=ncurves,loc=legendplacement)
  fig.tight_layout()#rect=(0,0,0.93,1))
  fig.savefig(savefig,dpi=600)
  return
  #}}}

def make_scatter_plot3D_mult(X1, Y1, Z1, C1, X2, Y2, Z2, C2, X3, Y3, Z3, C3,\
          elevation=30, azimuthal=30, alpha=0.8,\
                      xlabel='', ylabel='', zlabel='', clabel='', title='',\
                      bounds=None, equal_mass=1, \
          label=None, logC=True, cmin=0.8, cmax=1., savefig='plots/plot.png'):
  #{{{
  S = 30#50*Z
  if logC:
    print >>sys.stderr, "Using logscale on Z"
    C1 = np.log10(np.abs(C1))
    C2 = np.log10(np.abs(C2))
    C3 = np.log10(np.abs(C3))
    if 'FF' in clabel: cmin = -2.3
    else: cmin = np.round(min(min(C1),min(C2),min(C3)) * 100)/100.
    cmax = np.round(max(max(C1),max(C2),max(C3)) * 100)/100.
    clabel = clabel + ' (Log)'
    # Insert logic here to ensure cmin and cmax are the limits on bounds
  else:
    print >>sys.stderr, "NOT using logscale on Z"
    if 'FF' in clabel: cmin = 10**-2.3
    else: cmin = np.round(min(min(C1),min(C2),min(C3)) * 100)/100.
    cmax = np.round(max(max(C1),max(C2),max(C3)) * 100)/100.
    # Insert logic here to ensure cmin and cmax are the limits on bounds
  #
  if bounds != None and (type(bounds) == np.ndarray or type(bounds) == list):
    bounds = insert_min_max_into_array(bounds, cmin, cmax)
  # 
  if bounds is None and logC:
    if 'FF' in clabel or 'mathcal{M}' in clabel:
      bounds = np.log10([0.0001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 1])
    elif '\mathcal{M}_c' in clabel:
      print "CHIRP MASS PLOT"
      bounds = np.log10([0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2])
    elif '\Delta' in clabel:
      bounds = np.log10([0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.])
    else:
      bounds = np.linspace(cmin, cmax, 10)
      #bounds = np.append( bounds - 0.1, 0 )
  elif bounds is None: 
    #raise IOError("Non-log default colorbar bounds not supported")
    print(" >> >> >> Non-log default colorbar bounds not supported")
  #xlabel = xlabel.replace('_', '-')
  #ylabel = ylabel.replace('_', '-')
  #zlabel = zlabel.replace('_', '-')
  #clabel = clabel.replace('_', '-')
  title  = title.replace('_', '-')
  fig = plt.figure(int(1e7 * np.random.random()), figsize=(12,4))
  #
  #cmap = plt.cm.Spectral
  cmap = plt.cm.RdYlGn#bwr
  #cmap = plt.get_cmap('jet', 20)
  #cmaplist = [cmap(i) for i in range(cmap.N)]
  #cmap = cmap.from_list('Custom map', cmaplist, cmap.N)
  #cmap.set_under('gray')
  print "bounds = ", bounds
  if type(bounds) == np.ndarray or type(bounds) == list:
    norm = mp.colors.BoundaryNorm(bounds, cmap.N)
  else:
    tmp_bounds = np.linspace(cmin, cmax, 10)
    norm = mp.colors.BoundaryNorm(tmp_bounds, cmap.N)
  # 1
  pltnum = 1
  ax = fig.add_subplot(131, projection='3d')
  ax.view_init(elev=elevation, azim=azimuthal)
  scat = ax.scatter(X1, Y1, Z1, c=C1, s=S, lw=0,\
                      alpha=alpha, vmin=cmin, vmax=cmax,\
                      cmap=cmap, norm=norm)
  # Add points in the bottom plane marking spins
  ax.plot( X1, Y1, (min(Z1)-5.) * np.ones(len(Z1)), 'kx', markersize=4 )
  # Add mirrored points if q == 1
  if equal_mass == pltnum:
    scat = ax.scatter(Y1, X1, Z1, c=C1, s=S, lw=0,\
                      alpha=alpha, vmin=cmin, vmax=cmax,\
                      cmap=cmap, norm=norm)
    ax.plot( Y1, X1, (min(Z1)-5.) * np.ones(len(Z1)), 'kx', markersize=4 )
  #
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_xlim([-1,1])
  ax.set_ylim([-1,1])
  ax.zaxis.set_rotate_label(False)
  ax.set_zlabel(zlabel, rotation=90)
  ax.set_title('$q = 1$', verticalalignment='bottom')
  ax.grid()
  # 2
  pltnum = 2
  ax = fig.add_subplot(132, projection='3d')
  ax.view_init(elev=elevation, azim=azimuthal)
  scat = ax.scatter(X2, Y2, Z2, c=C2, s=S, lw=0, \
                      alpha=alpha, vmin=cmin, vmax=cmax,\
                      cmap=cmap, norm=norm)
  if equal_mass == pltnum:
    scat = ax.scatter(Y1, X1, Z1, c=C1, s=S, lw=0,\
                      alpha=alpha, vmin=cmin, vmax=cmax,\
                      cmap=cmap, norm=norm)
  # Add points in the bottom plane marking spins
  ax.plot( X2, Y2, (min(Z2)-5.) * np.ones(len(Z2)), 'kx', markersize=4 )
  #
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_xlim([-1,1])
  ax.set_ylim([-1,1])
  ax.set_title(title+'\n$q = 2$')
  ax.grid()
  # 3
  pltnum = 3
  ax = fig.add_subplot(133, projection='3d')
  ax.view_init(elev=elevation, azim=azimuthal)
  scat = ax.scatter(X3, Y3, Z3, c=C3, s=S, lw=0,\
                      alpha=alpha, vmin=cmin, vmax=cmax,\
                      cmap=cmap, norm=norm)
  if equal_mass == pltnum:
    scat = ax.scatter(Y1, X1, Z1, c=C1, s=S, lw=0,\
                      alpha=alpha, vmin=cmin, vmax=cmax,\
                      cmap=cmap, norm=norm)
  # Add points in the bottom plane marking spins
  ax.plot( X3, Y3, (min(Z3)-5.) * np.ones(len(Z3)), 'kx', markersize=4 )
  #
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_xlim([-1,1])
  ax.set_ylim([-1,1])
  ax.grid()
  ax.set_title('$q = 3$', verticalalignment='bottom')
  #ax.set_zlabel(zlabel)
  #ax.set_title(title)
  #
  #ax2 = fig.add_axes([0.95, 0.1, 0.01, 0.7])
  ax2 = fig.add_axes([0.2, 0.05, 0.6, 0.02])
  # Make the colorbar
  if type(bounds) == np.ndarray or type(bounds) == list:
    cb = mp.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, \
          spacing='uniform', format='%.2f', orientation=u'horizontal',\
          ticks=bounds, boundaries=bounds)
  else:
    # How does this colorbar know what colors to span??
    cb = mp.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, \
          spacing='uniform', format='%.2f', orientation=u'horizontal',\
          ticks=tmp_bounds)
  # Add tick labels
  if logC and (type(bounds) == np.ndarray or type(bounds) == list):
    cb.set_ticklabels(np.round(10**bounds, decimals=3))
  elif type(bounds) == np.ndarray or type(bounds) == list:
    cb.set_ticklabels(np.round(bounds, decimals=3))
  #cb = fig.colorbar(scat, shrink=0.5, aspect=30, spacing='proportional',\
  #  ticks=[0,-0.5,-1,-1.3,-1.6,-1.9,-2.1,-2.4,-2.7,-3])
  #ax2.set_title(clabel, loc='left')
  cb.set_label(clabel, labelpad=-0.3,y=0)  
  #if max(C) < cmax and min(C) > cmin: cb.set_clim([cmin,cmax])
  fig.tight_layout()
  print savefig
  if '.png' in savefig:
    savefig = savefig.split('.png')[0]+'_q123.png'
  elif '.pdf' in savefig:
    savefig = savefig.split('.pdf')[0]+'_q123.pdf'
  print savefig
  fig.savefig(savefig)
  return
  #}}}

# WRITE A NEW FUNCTION THAT CAN PLOT 2-3 (rows) X 3 (COLUMN) plots:
def make_scatter_plot3D_multrow(Xs, Ys, Zs, Cs,\
          elevation=30, azimuthal=30, alpha=0.8,\
                      xlabel='', ylabel='', zlabel='', clabel='', title='',\
                      bounds=None, equal_mass=1, \
                      colormin=None, colormax=None, \
          label=None, logC=True, cmin=0.8, cmax=1., savefig='plots/plot.png'):
  #{{{
  if len(Xs) != len(Ys) or len(Xs) != len(Zs) or len(Xs) != len(Cs):
    raise IOError("Length of lists to be plotted not the same")
  nrows, ncols = np.shape(Xs) # Known failure mode: when all arrays in Xrows are of the same length
  print "No of rows = %d, columns = %d" % (nrows, ncols)
  #
  gmean = (5.**0.5-1)*0.5
  S = 30#50*Z
  if logC:
    bounds = np.log10(bounds)
    print >>sys.stderr, "Using logscale on Z"
    for ridx, C in enumerate(Cs):
      for cidx, R in enumerate(C):
        Cs[ridx][cidx] = np.log10(R)
    if 'FF' in clabel: cmin = -2.3
    else:
      Cs = np.array(Cs)
      print "Shape of C = ", np.shape(Cs), "dtype of C = ", type(Cs)
      cmin = np.inf
      for tmpr in Cs:
        for tmpc in tmpr:
          print np.shape(tmpc), np.shape(tmpr)
          if cmin > np.min(tmpc): cmin = np.min(tmpc)
      print "Min = ", cmin
      cmin = np.round(cmin, decimals=3)
    #clabel = clabel + ' (Log)'
  else:
    print >>sys.stderr, "NOT using logscale on Z"
    if 'FF' in clabel: cmin = 10**-2.3
    else:
      Cs = np.array(Cs)
      print "Shape of C = ", np.shape(Cs), "dtype of C = ", type(Cs)
      cmin = np.inf
      for tmpr in Cs:
        for tmpc in tmpr:
          print np.shape(tmpc), np.shape(tmpr)
          if cmin > np.min(tmpc): cmin = np.min(tmpc)
      print "Min = ", cmin
      cmin = np.round(cmin, decimals=3)
  cmax = -np.inf
  for tmpr in Cs:
    for tmpc in tmpr:
      if cmax < np.max(tmpc): cmax = np.max(tmpc)
  cmax = np.round(cmax, decimals=2)
  # After having computed cmin, cmax, check for user inputs
  if colormin is not None: cmin = colormin
  if colormax is not None: cmax = colormax
  #
  if bounds != None and (type(bounds) == np.ndarray or type(bounds) == list):
    print "bounds before insert  : ", bounds
    bounds = insert_min_max_into_array(bounds, cmin, cmax)
    print "bounds after insert  : ", bounds
  # 
  # Insert default values of bounds
  if bounds is None and logC:
    if 'FF' in clabel or 'mathcal{M}' in clabel:
      bounds = np.log10([0.0001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 1])
    elif '\mathcal{M}_c' in clabel:
      print "CHIRP MASS PLOT"
      bounds = np.log10([0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2])
    elif '\Delta' in clabel:
      bounds = np.log10([0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.])
    else:
      bounds = np.linspace(cmin, cmax, 10)
      #bounds = np.append( bounds - 0.1, 0 )
  elif bounds is None: 
    #raise IOError("Non-log default colorbar bounds not supported")
    print(">> >> Non-log default colorbar bounds not supported")
  #xlabel = xlabel.replace('_', '-')
  #ylabel = ylabel.replace('_', '-')
  #zlabel = zlabel.replace('_', '-')
  #clabel = clabel.replace('_', '-')
  for tid, t in enumerate(title): t[tid].replace('_', '-')
  fig = plt.figure(int(1e7 * np.random.random()), figsize=(4*ncols,4*nrows))
  #cmap = plt.cm.RdYlGn_r#gist_heat_r##winter#gnuplot#PiYG_r#RdBu_r#jet#rainbow#RdBu_r#Spectral_r
  cmap = plt.cm.jet#seismic#RdBu_r#RdYlGn_r#PiYG_r#jet#
  #cmap = plt.get_cmap('jet', 20)
  #cmaplist = [cmap(i) for i in range(cmap.N)]
  #cmap = cmap.from_list('Custom map', cmaplist, cmap.N)
  #cmap.set_under('gray')
  print "bounds = ", bounds
  if type(bounds) == np.ndarray or type(bounds) == list:
    norm = mp.colors.BoundaryNorm(bounds, cmap.N)
  else:
    tmp_bounds = np.linspace(cmin, cmax, 20)
    norm = mp.colors.BoundaryNorm(tmp_bounds, cmap.N)
  #
  #### Begin plotting loop
  nplot = 0
  for rowid in range( nrows ):
    for colid in range( ncols ):
      nplot += 1
      ax = fig.add_subplot(nrows, ncols, nplot, projection='3d')
      ax.view_init(elev=elevation, azim=azimuthal)
      X, Y, Z, C = Xs[rowid][colid], Ys[rowid][colid], Zs[rowid][colid], Cs[rowid][colid]
      # Add the actual color plot
      scat = ax.scatter(X, Y, Z, c=C, s=S, lw=0, alpha=alpha, \
                        vmin=cmin, vmax=cmax, cmap=cmap, norm=norm)
      # Add points in the bottom plane marking spins
      ax.plot(X, Y, (np.min(Z)-2.)*np.ones(len(Z)), 'kx', markersize=4)
      if equal_mass == colid+1:
        scat = ax.scatter(Y, X, Z, c=C, s=S, lw=0, alpha=alpha, \
                        vmin=cmin, vmax=cmax, cmap=cmap, norm=norm)
        ax.plot(Y, X, (np.min(Z)-2.)*np.ones(len(Z)), 'kx', markersize=4)
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)
      ax.set_xlim([-1,1])
      ax.set_ylim([-1,1])
      ax.zaxis.set_rotate_label(False)
      ax.set_zlabel(zlabel, rotation=108)
      ax.set_zlim(zmin=40)
      ax.locator_params(axis='z', nbins=5)
      ax.locator_params(axis='z', prune='upper')
      if colid == ncols/2:
        ax.set_title(title[rowid]+'\n $q=%d$' % (colid+1), verticalalignment='bottom')
      else:
        ax.set_title('$q=%d$' % (colid+1), verticalalignment='bottom')
      ax.grid()
  #
  #ax2 = fig.add_axes([0.9, 0.3, 0.01, 0.35]) # Short colorbar
  ax2 = fig.add_axes([0.9, 0.1, 0.01, 0.7])
  
  #ax2 = fig.add_axes([0.2, 0.05, 0.6, 0.02])
  # Make the colorbar
  if type(bounds) == np.ndarray or type(bounds) == list:
    cb = mp.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, \
          spacing='uniform', format='%.2f', orientation=u'vertical',\
          ticks=bounds, boundaries=bounds)    
  else:
    # How does this colorbar know what colors to span??
    cb = mp.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, \
          spacing='uniform', format='%.3f', orientation=u'vertical')#,\
    #      #ticks=tmp_bounds)
    #cb = mp.colorbar.Colorbar(ax2,\
    #      spacing='uniform', format='%.2f', orientation=u'vertical')#,\
  # Add tick labels
  if logC and (type(bounds) == np.ndarray or type(bounds) == list):
    cb.set_ticklabels(np.round(10**bounds, decimals=4))
  elif type(bounds) == np.ndarray or type(bounds) == list:
    cb.set_ticklabels(np.round(bounds, decimals=4))
  #cb = fig.colorbar(scat, shrink=0.5, aspect=30, spacing='proportional',\
  #  ticks=[0,-0.5,-1,-1.3,-1.6,-1.9,-2.1,-2.4,-2.7,-3])
  #ax2.set_title(clabel, loc='left')
  cb.set_label(clabel, verticalalignment='top', horizontalalignment='center', size=22)
  #,labelpad=-0.3,y=1.1,x=-0.5)  
  #if max(C) < cmax and min(C) > cmin: cb.set_clim([cmin,cmax])
  fig.tight_layout(rect=(0,0,0.9,1))
  if '.png' in savefig: savefig = savefig.split('.png')[0]+'_q123.png'
  elif '.pdf' in savefig: savefig = savefig.split('.pdf')[0]+'_q123.pdf'
  fig.savefig(savefig)
  return
  #}}}

# WRITE A NEW FUNCTION THAT CAN PLOT 2-3 (rows) X 3 (COLUMN) plots
def make_contour_plot_multrow(Xs, Ys, Cs,\
          elevation=30, azimuthal=30, alpha=0.8,\
          xlabel='', ylabel='', zlabel='', clabel='', title='', titles=[],\
          bounds=None, colors=[], equal_mass=1, colorbartype='simple',\
          label=None, logC=True, cmin=0.8, cmax=1., savefig='plots/plot.png'):
  #{{{
  if not len(Xs)==len(Ys)==len(Cs):
    raise IOError("Length of lists to be plotted not the same")
  nrows, ncols = np.shape(Xs) # Known failure mode: when all arrays in Xrows are of the same length
  print "No of rows = %d, columns = %d" % (nrows, ncols)
  #
  gmean = (5.**0.5-1)*0.5
  if logC:
    bounds = np.log10(bounds)
    print >>sys.stderr, "Using logscale on Z"
    for ridx, C in enumerate(Cs):
      for cidx, R in enumerate(C):
        Cs[ridx][cidx] = np.log10(R)
    if 'FF' in clabel: cmin = -2.3
    else:
      Cs = np.array(Cs)
      print "Shape of C = ", np.shape(Cs), "dtype of C = ", type(Cs)
      cmin = np.inf
      for tmpr in Cs:
        for tmpc in tmpr:
          print np.shape(tmpc), np.shape(tmpr)
          if cmin > np.min(tmpc): cmin = np.min(tmpc)
      print "Min = ", cmin
      cmin = np.round(cmin, decimals=3)
    #clabel = clabel + ' (Log)'
  else:
    print >>sys.stderr, "NOT using logscale on Z"
    if 'FF' in clabel: cmin = 10**-2.3
    else:
      Cs = np.array(Cs)
      print "Shape of C = ", np.shape(Cs), "dtype of C = ", type(Cs)
      cmin = np.inf
      for tmpr in Cs:
        for tmpc in tmpr: cmin = min(cmin, np.min(tmpc))
      print "Min = ", cmin
      cmin = np.round(cmin, decimals=3)
  cmax = -np.inf
  for tmpr in Cs:
    for tmpc in tmpr: cmax = max(cmax, np.max(tmpc))
  cmax = np.round(cmax, decimals=3)
  #
  if bounds != None and (type(bounds) == np.ndarray or type(bounds) == list):
    print "bounds before insert  : ", bounds
    bounds = insert_min_max_into_array(bounds, cmin, cmax)
    print "bounds after insert  : ", bounds
  # 
  # Insert default values of bounds
  if bounds is None and logC:
    if 'FF' in clabel or 'mathcal{M}' in clabel:
      bounds = np.log10([0.0001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 1])
    elif '\mathcal{M}_c' in clabel:
      print "CHIRP MASS PLOT"
      bounds = np.log10([0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2])
    elif '\Delta' in clabel:
      bounds = np.log10([0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.])
    else:
      bounds = np.linspace(cmin, cmax, 10)
      #bounds = np.append( bounds - 0.1, 0 )
  elif bounds is None: 
    raise IOError("Non-log default colorbar bounds not supported")
  for tid, t in enumerate(title): t[tid].replace('_', '-')
  fig = plt.figure(int(1e7 * np.random.random()), figsize=(4*ncols,4*nrows))
  #cmap = plt.cm.RdYlGn_r#gist_heat_r##winter#gnuplot#PiYG_r#RdBu_r#jet#rainbow#RdBu_r#Spectral_r
  #cmap = plt.cm.PiYG_r#rainbow#RdBu_r#RdYlGn_r
  #cmap = plt.get_cmap('jet_r', 20)
  cmap = plt.cm.OrRd
  #cmaplist = [cmap(i) for i in range(cmap.N)]
  #cmap = cmap.from_list('Custom map', cmaplist, cmap.N)
  #cmap.set_under('gray')
  print "bounds = ", bounds
  if type(bounds) == np.ndarray or type(bounds) == list:
    norm = mp.colors.BoundaryNorm(bounds, cmap.N)
  else:
    tmp_bounds = np.linspace(cmin, cmax, 10)
    norm = mp.colors.BoundaryNorm(tmp_bounds, cmap.N)
  #
  #### Begin plotting loop
  nplot = 0
  allaxes = []
  for rowid in range( nrows ):
    for colid in range( ncols ):
      nplot += 1
      ax = fig.add_subplot(nrows, ncols, nplot)
      allaxes.append( ax )
      #
      X, Y, C = Xs[rowid][colid], Ys[rowid][colid], Cs[rowid][colid]
      print np.shape(X), np.shape(Y), np.shape(C)
      # Add points in the bottom plane marking spins
      if equal_mass == colid+1:
        tmpX, tmpY = X, Y
        X = np.append(X, tmpY)
        Y = np.append(Y, tmpX)
        C = np.append(C, C)
      #
      Xrange = np.linspace( min(X), max(X), 2000 )
      Yrange = np.linspace( min(Y), max(Y), 2000 )
      #Xrange = np.linspace( -1, 1, 100 )
      #Yrange = np.linspace( -1, 1, 100)
      Xmap, Ymap = np.meshgrid( Xrange, Yrange )
      print np.shape(X), np.shape(Y), np.shape(C)
      colormap = plt.mlab.griddata( X, Y, C, Xmap, Ymap, interp='linear')
      #
      import scipy.interpolate as si
      rbfi = si.SmoothBivariateSpline(X, Y, C, kx=4, ky=4)
      #colormap = rbfi(Xrange, Yrange)
      #
      # New interpolation scheme
      #
      import scipy.ndimage
      #xyzData = np.append( np.append( [X], [Y], axis=0 ), [C], axis=0 )
      #xyzData = scipy.ndimage.zoom(xyzData, 3)
      #Xmap = xyzData[:,0]
      #Ymap = xyzData[:,1]
      #colormap = xyzData[:,2]
      print "Shape pof Xmap, Ymap, colormap = ", \
                        np.shape(Xmap), np.shape(Ymap), np.shape(colormap)
      #Xmap = scipy.ndimage.zoom(Xmap, 3)
      #Ymap = scipy.ndimage.zoom(Ymap, 3)
      #colormap = scipy.ndimage.zoom(colormap, 3)
      if len(colors)==(len(bounds)-1):
        CS = ax.contourf(Xmap, Ymap, colormap,\
                    levels=bounds, \
                    colors=colors,\
                    alpha=0.75,\
                    #cmap=plt.cm.spectral,\
                    linestyles='dashed')
        '''CS = ax.tricontourf(X,Y,C,\
                    levels=bounds,\
                    colors=colors,\
                    alpha=0.9)'''
        '''CS1 = ax.scatter(X, Y, c='k', s=5)#, \
                    #reduce_C_function=np.max)'''
      else:
        ax.contourf( Xmap, Ymap, colormap, \
                      bounds, cmap=cmap, linestyles='dashed')  
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)
      ax.set_xlim([-1,1])
      ax.set_ylim([-1,1])
      #ax.zaxis.set_rotate_label(False)
      #ax.set_zlabel(zlabel, rotation=90)
      print "Len(titles) = %d, NCOLS = %d" % (len(titles), ncols)
      if len(titles)==ncols:
        ax.set_title(titles[colid], verticalalignment='bottom')
      elif colid == ncols/2:
        ax.set_title(title[rowid]+'\n $q=%d$' % (colid+1), verticalalignment='bottom')
      else:
        ax.set_title('$q=%d$' % (colid+1), verticalalignment='bottom')
      ax.grid()
  #
  if colorbartype=='simple':
    ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.7])
    cb = plt.colorbar(CS, cax=ax2, orientation=u'vertical', format='%.3f')
    cb.set_label(clabel)
  else:
    ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.7])
    #ax2 = fig.add_axes([0.2, 0.05, 0.6, 0.02])
    # Make the colorbar
    if type(bounds) == np.ndarray or type(bounds) == list:
      cb = mp.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, \
            spacing='uniform', format='%.2f', orientation=u'vertical',\
            ticks=bounds, boundaries=bounds)
    else:
      # How does this colorbar know what colors to span??
      cb = mp.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, \
            spacing='uniform', format='%.2f', orientation=u'vertical',\
            ticks=tmp_bounds)
    # Add tick labels
    if logC and (type(bounds) == np.ndarray or type(bounds) == list):
      cb.set_ticklabels(np.round(10**bounds, decimals=4))
    elif type(bounds) == np.ndarray or type(bounds) == list:
      cb.set_ticklabels(np.round(bounds, decimals=4))
    #cb = fig.colorbar(scat, shrink=0.5, aspect=30, spacing='proportional',\
    #  ticks=[0,-0.5,-1,-1.3,-1.6,-1.9,-2.1,-2.4,-2.7,-3])
    #ax2.set_title(clabel, loc='left')
    cb.set_label(clabel, verticalalignment='top', horizontalalignment='center')#,labelpad=-0.3,y=1.1,x=-0.5)  
    #if max(C) < cmax and min(C) > cmin: cb.set_clim([cmin,cmax])
  fig.tight_layout(rect=(0,0,0.93,1))
  if '.png' in savefig: savefig = savefig.split('.png')[0]+'_q123.png'
  elif '.pdf' in savefig: savefig = savefig.split('.pdf')[0]+'_q123.pdf'
  fig.savefig(savefig)
  return
  #}}}


def make_scatter_plot(X, Y, Z, xlabel='', ylabel='', zlabel='', title='',\
                      logz=True, cmin=0.8, cmax=1., savefig='plots/plot.png'):
  #{{{
  S = 30#50*Z
  if logz:
    print >>sys.stderr, "Using logscale on Z"
    Z = np.log10(1. - Z)
    cmin, cmax = -2.75, max(Z)
    zlabel='Log[Mismatch]'
  xlabel = xlabel.replace('_', '-')
  ylabel = ylabel.replace('_', '-')
  zlabel = zlabel.replace('_', '-')
  title  = title.replace('_', '-')
  plt.figure(int(1e7 * np.random.random()))
  plt.scatter(X, Y, c=Z, s=S, lw=0, alpha=0.8)
  cb = plt.colorbar()
  cb.set_label(zlabel)
  if max(Z) < cmax and min(Z) > cmin: cb.set_clim([cmin,cmax])
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.grid()
  plt.savefig(savefig)
  return
  #}}}
  
def make_scatter_plot3D(X, Y, Z, C, elevation=30, azimuthal=30, alpha=0.8,\
                      xlabel='', ylabel='', zlabel='', clabel='', title='',\
          label=None, logC=True, cmin=0.8, cmax=1., savefig='plots/plot.png'):
  #{{{
  S = 30#50*Z
  if logC:
    print >>sys.stderr, "Using logscale on Z"
    C = np.log10(1. - C)
    cmin, cmax = -2.75, np.round(max(C) * 100)/100.
    clabel='Log[Mismatch]'
  xlabel = xlabel.replace('_', '-')
  ylabel = ylabel.replace('_', '-')
  zlabel = zlabel.replace('_', '-')
  clabel = clabel.replace('_', '-')
  title  = title.replace('_', '-')
  fig = plt.figure(int(1e7 * np.random.random()))
  ax = fig.add_subplot(111, projection='3d')
  ax.view_init(elev=elevation, azim=azimuthal)
  #
  cmap = plt.cm.jet
  #cmap = plt.get_cmap('jet', 20)
  cmaplist = [cmap(i) for i in range(cmap.N)]
  cmap = cmap.from_list('Custom map', cmaplist, cmap.N)
  cmap.set_under('gray')
  bounds = np.linspace(cmin, cmax, 10)
  bounds = np.append( bounds - 0.1, 0 )
  norm = mp.colors.BoundaryNorm(bounds, cmap.N)
  #
  scat = ax.scatter(X, Y, Z, c=C, s=S, lw=0, alpha=alpha, vmin=cmin, vmax=cmax,\
                      cmap=cmap, norm=norm)
  ax2 = fig.add_axes([0.9, 0.1, 0.01, 0.7])
  cb = mp.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, \
        spacing='uniform', format='%.1f', ticks=bounds, boundaries=bounds)
  #cb = fig.colorbar(scat, shrink=0.5, aspect=30, spacing='proportional',\
  #  ticks=[0,-0.5,-1,-1.3,-1.6,-1.9,-2.1,-2.4,-2.7,-3])
  ax2.set_label(clabel)
  #if max(C) < cmax and min(C) > cmin: cb.set_clim([cmin,cmax])
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_zlabel(zlabel)
  ax.set_title(title)
  ax.grid()
  fig.tight_layout()
  print savefig
  if label is not None: savefig = savefig.strip('png')[:-1]+('_%s.png' % label)
  fig.savefig(savefig)
  return
  #}}}

### This function has not been tested and in all likelihood it does not work
def make_contourf_mult(X1, Y1, C1, X2, Y2, C2, X3, Y3, C3,\
          elevation=30, azimuthal=30, alpha=0.8,\
                      xlabel='', ylabel='', zlabel='', clabel='', title='',\
          label=None, logC=True, cmin=0.8, cmax=1., savefig='plots/plot.png'):
  #{{{
  S = 30#50*Z
  if logC:
    print >>sys.stderr, "Using logscale on Z"
    C1 = np.log10(1. - C1)
    C2 = np.log10(1. - C2)
    C3 = np.log10(1. - C3)
    cmin, cmax = -2.75, np.round(max(max(C1),max(C2),max(C3)) * 100)/100.
    clabel='Log[1-FF]'
  xlabel = xlabel.replace('_', '-')
  ylabel = ylabel.replace('_', '-')
  zlabel = zlabel.replace('_', '-')
  clabel = clabel.replace('_', '-')
  title  = title.replace('_', '-')
  fig = plt.figure(int(1e7 * np.random.random()), figsize=(12,4))
  #
  cmap = plt.cm.jet
  cmaplist = [cmap(i) for i in range(cmap.N)]
  cmap = cmap.from_list('Custom map', cmaplist, cmap.N)
  cmap.set_under('gray')
  #bounds = np.linspace(cmin, cmax, 10)
  #bounds = np.append( bounds - 0.1, 0 )
  #norm = mp.colors.BoundaryNorm(bounds, cmap.N)
  # 1
  Xrange = np.linspace( min(X1), max(X1), 400 )
  Yrange = np.linspace( min(Y1), max(Y1), 400 )
  Xmap, Ymap = np.meshgrid( Xrange, Yrange )
  etamap, incmap = Xmap, Ymap
  colormap = plt.mlab.griddata( X1, Y1, C1, Xmap, Ymap)
  ax = fig.add_subplot(131)
  #pylab.contour(etamap, incmap, colormap,  [0.947, .965], colors='black', linestyles='dashed' )
  ax.contourf( etamap, incmap, colormap, \
        [.92,.93,.94,.947,.96,.965,.97,.98,.99,1.], cmap=cmap, \
        linestyles='dashed')  
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title('q = 1')
  ax.grid()
  # 2
  Xrange = np.linspace( min(X2), max(X2), 400 )
  Yrange = np.linspace( min(Y2), max(Y2), 400 )
  Xmap, Ymap = np.meshgrid( Xrange, Yrange )
  colormap = plt.mlab.griddata( X2, Y2, C2, Xmap, Ymap)
  ax = fig.add_subplot(132)
  ax.contourf( etamap, incmap, colormap, \
        [.92,.93,.94,.947,.96,.965,.97,.98,.99,1.], cmap=cmap, \
        linestyles='dashed')  
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title+'\n q = 2')
  ax.grid()
  # 3
  Xrange = np.linspace( min(X3), max(X3), 400 )
  Yrange = np.linspace( min(Y3), max(Y3), 400 )
  Xmap, Ymap = np.meshgrid( Xrange, Yrange )
  colormap = plt.mlab.griddata( X3, Y3, C3, Xmap, Ymap)
  ax = fig.add_subplot(133)
  ax.contourf( etamap, incmap, colormap, \
        [.92,.93,.94,.947,.96,.965,.97,.98,.99,1.], cmap=cmap, \
        linestyles='dashed')  
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.grid()
  ax.set_title('q = 3')
  #ax2 = fig.add_axes([0.95, 0.1, 0.01, 0.7])
  #cb = mp.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, \
  #      spacing='uniform', format='%.1f', ticks=bounds, boundaries=bounds)
  #cb = fig.colorbar(scat, shrink=0.5, aspect=30, spacing='proportional',\
  #  ticks=[0,-0.5,-1,-1.3,-1.6,-1.9,-2.1,-2.4,-2.7,-3])
  #ax2.set_title(clabel)
  cb = plt.colorbar(ax, cmap=cmap, format='%.2f')
  cb.set_title(clabel)
  #if max(C) < cmax and min(C) > cmin: cb.set_clim([cmin,cmax])
  fig.tight_layout()
  print savefig
  savefig = savefig.strip('png')[:-1]+'_q123.png'
  fig.savefig(savefig)
  return
  #}}}

def make_parameters_plot(X, Y, Z, elevation=30, azimuthal=30, \
                      xlabel='', ylabel='', zlabel='', title='',\
                      savefig='plots/plot.png'):
  #{{{
  S = 30#50*Z
  if logC:
    print >>sys.stderr, "Using logscale on Z"
    C = np.log10(1. - C)
    cmin, cmax = -2.75, np.round(max(C) * 100)/100.
    clabel='Log[Mismatch]'
  xlabel = xlabel.replace('_', '-')
  ylabel = ylabel.replace('_', '-')
  zlabel = zlabel.replace('_', '-')
  clabel = clabel.replace('_', '-')
  title  = title.replace('_', '-')
  fig = plt.figure(int(1e7 * np.random.random()))
  ax = fig.add_subplot(111, projection='3d')
  ax.view_init(elev=elevation, azim=azimuthal)
  #
  cmap = plt.cm.jet
  #cmap = plt.get_cmap('jet', 20)
  cmaplist = [cmap(i) for i in range(cmap.N)]
  cmap = cmap.from_list('Custom map', cmaplist, cmap.N)
  cmap.set_under('gray')
  bounds = np.linspace(cmin, cmax, 10)
  bounds = np.append( bounds - 0.1, 0 )
  norm = mp.colors.BoundaryNorm(bounds, cmap.N)
  #
  scat = ax.scatter(X, Y, Z, c=C, s=S, lw=0, alpha=alpha, vmin=cmin, vmax=cmax,\
                      cmap=cmap, norm=norm)
  ax2 = fig.add_axes([0.9, 0.1, 0.01, 0.7])
  cb = mp.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, \
        spacing='uniform', format='%.1f', ticks=bounds, boundaries=bounds)
  #cb = fig.colorbar(scat, shrink=0.5, aspect=30, spacing='proportional',\
  #  ticks=[0,-0.5,-1,-1.3,-1.6,-1.9,-2.1,-2.4,-2.7,-3])
  ax2.set_label(clabel)
  #if max(C) < cmax and min(C) > cmin: cb.set_clim([cmin,cmax])
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_zlabel(zlabel)
  ax.set_title(title)
  ax.grid()
  fig.tight_layout()
  print savefig
  if label is not None: savefig = savefig.strip('png')[:-1]+('_%s.png' % label)
  fig.savefig(savefig)
  return
  #}}}


#########################################################
####### FAITHFULNESS
#########################################################

class plot_mismatches_sim():
  def __init__(self, simdir=None, matchdirs=['matches'], plotdir='plots',\
                      verbose=True, debug=False):
    self.verbose = verbose
    self.debug   = debug
    self.simdir  = simdir
    self.simtag  = self.simdir.strip('/').split('/')[-1]
    self.data = errors_in_sim(simdir=simdir, matchdirs=matchdirs,\
                      verbose=self.verbose, debug=self.debug)
    for i in range(len(self.data.ccelevs)):
      self.data.ccelevs[i] = str(self.data.ccelevs[i])
    self.lines   = ["-","--","-.","-:"]
    self.markers = ["o","x","s","^","v","*",'.','<']
    self.colors  = ["blue","red","green","magenta","cyan","gold","black",\
                    "darkorange"]
    self.taperlabels = ["None","A","B","C","D","E"]
    self.plotdir = plotdir + matchdirs[0].lstrip('matches')
  #
  def plot_cce_mismatches_all(self, nsubplotrows=2, nsubplotcols=2,\
                              savedir=None, savefig=None):
    #{{{
    if savedir is None: savedir = self.plotdir
    #panel1,2,3,4
    #For each tapering: x5
    #    
    #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,sharex='col',sharey='row')
    fig = plt.figure(int(1e7 * np.random.random()))
    fig.set_size_inches(16,12)
    fig.suptitle(self.simtag, fontsize=14)
    self.data.ccelevs.sort()
    self.data.cceradii.sort()
    const_list = [self.data.ccelevs, self.data.cceradii, self.data.ccelevs,\
                  self.data.ccelevs]
    const_func = [self.data.ccer, self.data.ccelev, self.data.cceextrapolated,\
                  self.data.cceextrapolated]
    const_keys = [[],[],self.data.extraporders,self.data.cceradii]
    const_addtn= [[], [], [str(self.data.cceradii[0])],\
                    [str(self.data.cceradii[1])]]
    for idx in range(4):
      print "idx = ", idx
      additional_constraint = const_addtn[idx]
      print "additional constraing = ", additional_constraint
      ax = plt.subplot(nsubplotrows,nsubplotcols,idx + 1) # <<
      l_taper = self.taperlabels
      l_lev = []
      l_ccer= []
      plot_lines_all = []
      plot_lines_levs = []
      for lidx, lev in enumerate(const_list[idx]):#self.data.cceradii): #<<
        lev = str(lev)
        print "lev = ", lev
        l_lev.append( lev )
        const_key = lev
        #For each tapering: x6
        plot_lines = [ [],[],[],[],[],[] ]
        if idx >= 4:
          overlaps = {}
          for l2 in const_keys[idx]:
            print "l2, lev = ", l2, lev
            overlaps[l2] = const_func[idx](key=[l2]+[lev],noduplicate=True)
        else:
          overlaps = const_func[idx](key=additional_constraint+[lev],\
                                      noduplicate=True)
        print overlaps
        #self.data.ccelev(key=lev,noduplicate=True)#.values()[0] <<
        for i in range( overlaps.values()[0].nWindows ):
          for n in range( len(overlaps.keys()) ):
            print "i = %d/%d, n = %d/%d" % (i,overlaps.values()[0].nWindows,\
                                            n, len(overlaps.keys()))
            var_key = str(overlaps.keys()[n])
            print "var_key = ", var_key
            olap = overlaps[var_key]
            mass = olap.X()
            mismatch= 1. - olap.Y(i)
            #
            if i==0 and lidx==0: l_ccer.append( var_key.split('/')[-1] )
            pl, = plt.plot( mass, mismatch,\
                          linestyle=self.lines[lidx], lw=1,\
                          marker=self.markers[n], markersize=3,\
                          color=self.colors[i],
                          alpha=0.65 )
            plot_lines[i].append(pl,)
        plot_lines_all.append(plot_lines)
        plot_lines_levs.append(plot_lines[0])
      #
      print "l_taper,lev,ccer = ", l_taper, l_lev, l_ccer
      #
      ax.set_yscale('log')
      ax.grid(b=True,which='major')
      if idx+1 >= nsubplotcols*(nsubplotrows-1): ax.set_xlabel('mass (solar mass)')
      if idx%nsubplotcols == 0: ax.set_ylabel('Mismatches')
      legend_1 = plt.legend(zip(*plot_lines_levs)[0], l_lev,\
                            loc="upper left",prop={'size':8},\
                            fancybox=True)
      legend_2 = plt.legend(zip(*plot_lines)[0], l_taper, ncol=2,\
                            loc="lower right",prop={'size':8},\
                            fancybox=True)
      legend_3 = plt.legend(plot_lines[0], l_ccer, ncol=2, loc='lower left',\
                            prop={'size':8}, fancybox=True)
      ax.legend()
      ax.add_artist(legend_1)
      ax.add_artist(legend_2)
      ax.add_artist(legend_3)
      ax.set_ylim([1.e-5,0.01])
    #
    cmd.getoutput('mkdir -p %s' % self.simdir+'/'+savedir)
    savedir = self.simdir+'/'+savedir
    if savefig is not None: plt.savefig(savedir+'/'+savefig, dpi=500)
    else: plt.savefig(savedir+'/'+self.simtag + '.png', dpi=500)
    return
  #}}}
  #
  def plot_cce_mismatches_only(self, nsubplotrows=1, nsubplotcols=2,\
                              savedir=None, savefig=None):
    #{{{
    if savedir is None: savedir = self.plotdir
    #panel1,2,3,4
    #For each tapering: x5
    #    
    #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,sharex='col',sharey='row')
    fig = plt.figure(int(1e7 * np.random.random()))
    fig.set_size_inches(12,9)
    fig.suptitle(self.simtag, fontsize=14)
    self.data.ccelevs.sort()
    self.data.cceradii.sort()
    const_list = [self.data.ccelevs, self.data.cceradii, self.data.ccelevs,\
                  self.data.ccelevs]
    const_func = [self.data.ccer, self.data.ccelev, self.data.cceextrapolated,\
                  self.data.cceextrapolated]
    const_keys = [[],[],self.data.extraporders,self.data.cceradii]
    const_addtn= [[], [], [str(self.data.cceradii[0])],\
                    [str(self.data.cceradii[1])]]
    for idx in range(2):
      print "idx = ", idx
      additional_constraint = const_addtn[idx]
      print "additional constraing = ", additional_constraint
      ax = plt.subplot(nsubplotrows,nsubplotcols,idx + 1) # <<
      l_taper = self.taperlabels
      l_lev = []
      l_ccer= []
      plot_lines_all = []
      plot_lines_levs = []
      for lidx, lev in enumerate(const_list[idx]):#self.data.cceradii): #<<
        lev = str(lev)
        print "lev = ", lev
        l_lev.append( lev )
        const_key = lev
        #For each tapering: x6
        plot_lines = [ [],[],[],[],[],[] ]
        if idx >= 4:
          overlaps = {}
          for l2 in const_keys[idx]:
            print "l2, lev = ", l2, lev
            overlaps[l2] = const_func[idx](key=[l2]+[lev],noduplicate=True)
        else:
          overlaps = const_func[idx](key=additional_constraint+[lev],\
                                      noduplicate=True)
        print overlaps
        #self.data.ccelev(key=lev,noduplicate=True)#.values()[0] <<
        for i in range( overlaps.values()[0].nWindows ):
          for n in range( len(overlaps.keys()) ):
            print "i = %d/%d, n = %d/%d" % (i,overlaps.values()[0].nWindows,\
                                            n, len(overlaps.keys()))
            var_key = str(overlaps.keys()[n])
            print "var_key = ", var_key
            olap = overlaps[var_key]
            mass = olap.X()
            mismatch= 1. - olap.Y(i)
            #
            if i==0 and lidx==0: l_ccer.append( var_key.split('/')[-1] )
            pl, = plt.plot( mass, mismatch,\
                          linestyle=self.lines[lidx], lw=1,\
                          marker=self.markers[n], markersize=3,\
                          color=self.colors[i],
                          alpha=0.65 )
            plot_lines[i].append(pl,)
        plot_lines_all.append(plot_lines)
        plot_lines_levs.append(plot_lines[0])
      #
      print "l_taper,lev,ccer = ", l_taper, l_lev, l_ccer
      #
      ax.set_yscale('log')
      ax.grid(b=True,which='major')
      if idx+1 >= nsubplotcols*(nsubplotrows-1): ax.set_xlabel('mass (solar mass)')
      if idx%nsubplotcols == 0: ax.set_ylabel('Mismatches')
      legend_1 = plt.legend(zip(*plot_lines_levs)[0], l_lev,\
                            loc="upper left",prop={'size':8},\
                            fancybox=True)
      legend_2 = plt.legend(zip(*plot_lines)[0], l_taper, ncol=2,\
                            loc="lower right",prop={'size':8},\
                            fancybox=True)
      legend_3 = plt.legend(plot_lines[0], l_ccer, ncol=2, loc='lower left',\
                            prop={'size':8}, fancybox=True)
      ax.legend()
      ax.add_artist(legend_1)
      ax.add_artist(legend_2)
      ax.add_artist(legend_3)
      ax.set_ylim([1.e-5,0.01])
    #
    cmd.getoutput('mkdir -p %s' % self.simdir+'/'+savedir)
    savedir = self.simdir+'/'+savedir
    if savefig is not None: plt.savefig(savedir+'/'+savefig, dpi=400)
    else: plt.savefig(savedir+'/'+self.simtag + '_cceOnly' + '.png', dpi=400)
    return
  #}}}
  #
  def plot_cce_extrapolation_mismatches(self, nsubplotrows=2, nsubplotcols=3,\
                              savedir=None, savefig=None):
    #{{{
    if savedir is None: savedir = self.plotdir
    #panel1,2,3,4
    #For each tapering: x5
    #    
    #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,sharex='col',sharey='row')
    fig = plt.figure(int(1e7 * np.random.random()))
    fig.set_size_inches(16,12)
    fig.suptitle(self.simtag, fontsize=14)
    self.data.ccelevs.sort()
    self.data.cceradii.sort()
    const_list = []
    for i in range(len(self.data.ccelevs)):
      const_list.append( [str(self.data.ccelevs[i])] )
    const_list = const_list + const_list
    print const_list
    #const_list = self.data.ccelevs + self.data.ccelevs # One plot for each lev
    const_func = [self.data.cceextrapolated, self.data.cceextrapolated,\
                  self.data.cceextrapolated, self.data.cceextrapolated,\
                  self.data.cceextrapolated, self.data.cceextrapolated]
    #const_keys = [self.data.extraporders,self.data.cceradii]
    const_addtn= [[str(self.data.cceradii[0])], [str(self.data.cceradii[0])],\
                  [str(self.data.cceradii[0])], [str(self.data.cceradii[1])],\
                  [str(self.data.cceradii[1])], [str(self.data.cceradii[1])]]
    # subplot each for a unique Lev+CCER combination
    for idx in range(6):
      print "idx = ", idx
      additional_constraint = const_addtn[idx]
      print "additional constraing = ", additional_constraint
      ax = plt.subplot(nsubplotrows,nsubplotcols,idx + 1) # <<
      l_taper = self.taperlabels
      l_lev = []
      l_ccer= []
      plot_lines_all = []
      plot_lines_levs = []
      for lidx, lev in enumerate(const_list[idx]):#self.data.cceradii): #<<
        lev = str(lev)
        print "lev = ", lev
        l_lev.append( lev )
        const_key = lev
        #For each tapering: x6
        plot_lines = [ [],[],[],[],[],[] ]
        overlaps = const_func[idx](key=additional_constraint+[lev],\
                                      noduplicate=True)
        print overlaps
        #self.data.ccelev(key=lev,noduplicate=True)#.values()[0] <<
        for i in range( overlaps.values()[0].nWindows ):
          for n in range( len(overlaps.keys()) ):
            print "i = %d/%d, n = %d/%d" % (i,overlaps.values()[0].nWindows,\
                                            n, len(overlaps.keys()))
            var_key = str(overlaps.keys()[n])
            print "var_key = ", var_key
            olap = overlaps[var_key]
            mass = olap.X()
            mismatch= 1. - olap.Y(i)
            #
            if i==0 and lidx==0: l_ccer.append( var_key.split('/')[-1] )
            pl, = plt.plot( mass, mismatch,\
                          linestyle=self.lines[lidx], lw=1,\
                          marker=self.markers[n], markersize=3,\
                          color=self.colors[i],
                          alpha=0.65 )
            plot_lines[i].append(pl,)
        plot_lines_all.append(plot_lines)
        plot_lines_levs.append(plot_lines[0])
      #
      print "l_taper,lev,ccer = ", l_taper, l_lev, l_ccer
      #
      ax.set_yscale('log')
      ax.grid(b=True,which='major')
      if idx+1 >= nsubplotcols*(nsubplotrows-1): ax.set_xlabel('mass (solar mass)')
      if idx%nsubplotcols == 0: ax.set_ylabel('Mismatches')
      legend_1 = plt.legend(zip(*plot_lines_levs)[0], l_lev,\
                            loc="upper left",prop={'size':8},\
                            fancybox=True)
      legend_2 = plt.legend(zip(*plot_lines)[0], l_taper, ncol=3,\
                            loc="lower right",prop={'size':8},\
                            fancybox=True)
      legend_3 = plt.legend(plot_lines[0], l_ccer, ncol=1, loc='upper right',\
                            prop={'size':8}, fancybox=True)
      ax.legend()
      ax.add_artist(legend_1)
      ax.add_artist(legend_2)
      ax.add_artist(legend_3)
      ax.set_ylim([1.e-5,0.01])
    #
    cmd.getoutput('mkdir -p %s' % self.simdir+'/'+savedir)
    savedir = self.simdir+'/'+savedir
    if savefig is not None: plt.savefig(savedir+'/'+savefig, dpi=500)
    else: plt.savefig(savedir+'/'+self.simtag +'_Extraction'+ '.png', dpi=500)
    return
    #}}}
  #
  def plot_cce_max_mismatch(self, savedir=None, savefig=None):
    #{{{
    if savedir is None: savedir = self.plotdir
    #
    overlaps = self.data.get_max_cce_mismatch()
    X = overlaps.X()
    Y = overlaps.Y()
    fig = plt.figure(int(1e7 * np.random.random()))
    for i in range( overlaps.nWindows ):
      plt.semilogy( X, 1. - Y, label=self.taperlabels[i] )
    plt.grid()
    plt.xlabel('Total mass (solar masses)')
    plt.ylabel('Max of CCER, Lev, Extraction mismatches')
    plt.title(self.simtag)
    plt.legend(ncol=2, loc='lower left')
    if savefig: plt.savefig(savedir + '/' + savefig, dpi=400)
    else: plt.savefig(savedir + '/' + self.simtag + '_MAXNR.png', dpi=400)
    return
    #}}}

class plot_mismatches_sims():
  """ 
  This class makes population plots. This is done to find patterns between
  NR errors based on binary parameters."""
  def __init__(self, basedir=None, simdirs=None, matchdirs=['matches'],\
                      plotdir='plots',\
                      verbose=True, debug=False):
    self.verbose = verbose
    self.debug   = debug
    self.basedir = basedir
    self.simdirs  = simdirs
    self.simtag  = self.simdirs.strip('/').split('/')[-1]
    self.data = {}
    for simdir in self.simdirs:
      self.data[simdir] = plot_mismatches_sim(simdir=self.basedir+'/'+simdir,\
                                      matchdirs=matchdirs,\
                                      verbose=self.verbose, debug=self.debug)
      for i in range(len(self.data[simdir].ccelevs)):
        self.data.ccelevs[i] = str(self.data.ccelevs[i])
      for i in range(len(self.data[simdir].cceradii)):
        self.data[simdir].cceradii[i] = str(self.data[simdir].cceradii[i])
    self.lines   = ["-","--","-.","-:"]
    self.markers = ["o","x","s","^","v","*",'.','<']
    self.colors  = ["blue","red","green","magenta","cyan","gold","black",\
                    "darkorange"]
    self.taperlabels = ["None","A","B","C","D","E"]
    self.plotdir = plotdir + matchdirs[0].lstrip('matches')
  #
  def hist_cce_mismatch(self):
    """
    This function will make histograms of different sources of error, across
    the catalog. The catalog is given as a list of dir names. The errors considered here are:
    1. NR Res: For R = OUTER: Lev4 vs Lev5, Lev3 vs Lev4
    2. CCE Radius: For Lev = Lev5: R1 vs R2
    3. Extraction: For Lev=Lev5, R = OUTER vs N2, N3, N4
    """
    #{{{
    # Get all required information for each sim, and then combine the info
    overlaps_all = {}
    for d in self.simdirs:
      overlaps_all[d] = []
      spl_lev = 'Lev5'
      spl_extrap=['N2','N3','N4']
      spl_ccer= 'CceR%04d' % max([int(x[-4:]) for x in self.data[d].cceradii])
      #
      const_combinations = [[\
                    [spl_ccer, 'Lev4', 'Lev5']\
                    ,[spl_ccer, 'Lev3', 'Lev4']]\
                    ,[['Lev3']+self.data[d].cceradii\
                    ,[['Lev4']+self.data[d].cceradii]\
                    ,[['Lev5']+self.data[d].cceradii]]\
                    ,[['Lev3', spl_ccer, 'N2']\
                    ,['Lev3', spl_ccer, 'N3']\
                    ,['Lev3', spl_ccer, 'N4']\
                    ,['Lev4', spl_ccer, 'N2']\
                    ,['Lev4', spl_ccer, 'N3']\
                    ,['Lev4', spl_ccer, 'N4']\
                    ,['Lev5', spl_ccer, 'N2']\
                    ,['Lev5', spl_ccer, 'N3']\
                    ,['Lev5', spl_ccer, 'N4']]
                    ]
      const_funcs = [[\
                    self.data[d].ccelev\
                    ,self.data[d].ccelev]\
                    ,[self.data[d].ccer\
                    ,self.data[d].ccer\
                    ,self.data[d].ccer]\
                    ,[self.data[d].cceextrapolated\
                    ,self.data[d].cceextrapolated\
                    ,self.data[d].cceextrapolated\
                    ,self.data[d].cceextrapolated\
                    ,self.data[d].cceextrapolated\
                    ,self.data[d].cceextrapolated\
                    ,self.data[d].cceextrapolated\
                    ,self.data[d].cceextrapolated\
                    ,self.data[d].cceextrapolated]\
                    ]
      for kdx, key_combos in enumerate(const_combinations):
        overlaps_tmp = []
        for idx, key_combo in enumerate(key_combos):
          overlaps = const_funcs[jdx][idx](key=key_combo, noduplicate=True)
          if len(overlaps.values()) > 1:
            raise RuntimeError("keys %s gave %d resuls" %\
                            (key_combo, len(overlaps.keys())))
          overlaps_tmp.append( overlaps.values()[0] ) # proceed if only 1 dataset
        overlaps_all[d].append( overlaps_tmp )
    #
    # Now combine info for all sims. CONCATENATE, NOT MAXIMIZE!!!
    # TODO
    #}}}
  #
  def plot_cce_mismatches_all(self, nsubplotrows=2, nsubplotcols=2,\
                              savedir=None, savefig=None):
    #{{{
    if savedir is None: savedir = self.plotdir
    #panel1,2,3,4
    #For each tapering: x5
    #    
    #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,sharex='col',sharey='row')
    fig = plt.figure(int(1e7 * np.random.random()))
    fig.set_size_inches(16,12)
    fig.suptitle(self.simtag, fontsize=14)
    self.data.ccelevs.sort()
    self.data.cceradii.sort()
    const_list = [self.data.ccelevs, self.data.cceradii, self.data.ccelevs,\
                  self.data.ccelevs]
    const_func = [self.data.ccer, self.data.ccelev, self.data.cceextrapolated,\
                  self.data.cceextrapolated]
    const_keys = [[],[],self.data.extraporders,self.data.cceradii]
    const_addtn= [[], [], [str(self.data.cceradii[0])],\
                    [str(self.data.cceradii[1])]]
    for idx in range(4):
      print "idx = ", idx
      additional_constraint = const_addtn[idx]
      print "additional constraing = ", additional_constraint
      ax = plt.subplot(nsubplotrows,nsubplotcols,idx + 1) # <<
      l_taper = self.taperlabels
      l_lev = []
      l_ccer= []
      plot_lines_all = []
      plot_lines_levs = []
      for lidx, lev in enumerate(const_list[idx]):#self.data.cceradii): #<<
        lev = str(lev)
        print "lev = ", lev
        l_lev.append( lev )
        const_key = lev
        #For each tapering: x6
        plot_lines = [ [],[],[],[],[],[] ]
        if idx >= 4:
          overlaps = {}
          for l2 in const_keys[idx]:
            print "l2, lev = ", l2, lev
            overlaps[l2] = const_func[idx](key=[l2]+[lev],noduplicate=True)
        else:
          overlaps = const_func[idx](key=additional_constraint+[lev],\
                                      noduplicate=True)
        print overlaps
        #self.data.ccelev(key=lev,noduplicate=True)#.values()[0] <<
        for i in range( overlaps.values()[0].nWindows ):
          for n in range( len(overlaps.keys()) ):
            print "i = %d/%d, n = %d/%d" % (i,overlaps.values()[0].nWindows,\
                                            n, len(overlaps.keys()))
            var_key = str(overlaps.keys()[n])
            print "var_key = ", var_key
            olap = overlaps[var_key]
            mass = olap.X()
            mismatch= 1. - olap.Y(i)
            #
            if i==0 and lidx==0: l_ccer.append( var_key.split('/')[-1] )
            pl, = plt.plot( mass, mismatch,\
                          linestyle=self.lines[lidx], lw=1,\
                          marker=self.markers[n], markersize=3,\
                          color=self.colors[i],
                          alpha=0.65 )
            plot_lines[i].append(pl,)
        plot_lines_all.append(plot_lines)
        plot_lines_levs.append(plot_lines[0])
      #
      print "l_taper,lev,ccer = ", l_taper, l_lev, l_ccer
      #
      if idx+1 >= nsubplotcols*(nsubplotrows-1): ax.set_xlabel('mass (solar mass)')
      if idx%nsubplotcols == 0: ax.set_ylabel('Mismatches')
      legend_1 = plt.legend(zip(*plot_lines_levs)[0], l_lev,\
                            loc="upper left",prop={'size':8},\
                            fancybox=True)
      legend_2 = plt.legend(zip(*plot_lines)[0], l_taper, ncol=2,\
                            loc="lower right",prop={'size':8},\
                            fancybox=True)
      legend_3 = plt.legend(plot_lines[0], l_ccer, ncol=2, loc='lower left',\
                            prop={'size':8}, fancybox=True)
      ax.legend()
      ax.add_artist(legend_1)
      ax.add_artist(legend_2)
      ax.add_artist(legend_3)
      ax.set_ylim([1.e-5,0.01])
    #
    cmd.getoutput('mkdir -p %s' % self.simdir+'/'+savedir)
    savedir = self.simdir+'/'+savedir
    if savefig is not None: plt.savefig(savedir+'/'+savefig, dpi=500)
    else: plt.savefig(savedir+'/'+self.simtag + '.png', dpi=500)
    return
    #}}}

