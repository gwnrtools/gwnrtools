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

import h5py
import numpy as np
import os
import sys
import subprocess as cmd


# Overlap storage classes
class overlaps_vs_totalmass():
    # {{{
    def __init__(self, dataset=None, verbose=True):
        # {{{
        if dataset is None:
            raise IOError("Need a dataset to initialize")
        self.M, self.O = dataset[:, 0], dataset[:, 1:]
        self.nWindows = len(self.O[0, :])
        # }}}

    #
    def X(self):
        return self.M

    #

    def Y(self, idx):
        # {{{
        if idx >= self.nWindows:
            raise IOError("This column does not exist in data")
        return self.O[:, idx]
        # }}}

    def get_overlap_mass_taper(self, mass, taperid):
        # {{{
        X = self.M
        Y = self.O
        if taperid < 0 or taperid >= self.nWindows:
            raise IOError("only have %d(%d) taperwins" %
                          (self.nWindows, taperid))
        idx, = np.where(X == mass)
        if len(idx) == 1:
            return Y[idx, taperid]
        for idx, mm in enumerate(X):
            if (mm - mass) < 1.e-12:
                return Y[idx, taperid]
        raise IOError("This mass value not found")
        # }}}

    # }}}


class Overlaps():
    """ Hide the complexity of HDF5 file structure here.
        Provide an interable, corresponding to subdirs in HDF5 file,
        Provide overlaps, labels corresponding to each subdir
    """

    # {{{

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

    def string_from_dir(self, dirname):
        if '.dir' in dirname:
            return dirname.strip('.dir')
        else:
            return dirname

    #

    def dir_from_string(self, strname):
        return strname + '.dir'

    #

    def string_from_dset(self, dsetname):
        dtmp = dsetname.strip('.dat').split('_')
        if self.debug:
            print(dtmp)
        # Get index of first dir
        for idx in range(len(dtmp)):
            if '.dir' in dtmp[idx]:
                break
        if idx == len(dtmp) - 1 and idx == 1:
            idx = 0
        d1, d2 = add_strings(dtmp[:idx + 1]), add_strings(dtmp[idx + 1:])
        d1, d2 = self.string_from_dir(d1), self.string_from_dir(d2)
        print("d1, d2 = ", d1, d2)
        return d1 + '_' + d2

    #

    def dset_from_string(self, dsetname):
        dtmp = dsetname.strip('.dat').split('_')
        if self.debug:
            print(dtmp)
        # Get index of first dir
        for idx in range(len(dtmp)):
            if '.dir' in dtmp[idx]:
                break
        d1, d2 = add_strings(dtmp[:idx + 1]), add_strings(dtmp[idx + 1:])
        d1, d2 = string_from_dir(d1), string_from_dir(d2)
        return d1 + '_' + d2

    #

    def read_data(self):
        self.fin = h5py.File(self.fullfilename, 'r')
        for k in list(self.fin.keys()):
            self.filedirs[k] = []
            self.read_dir(l1dir=k, openfile=False)
        self.fin.close()

    #

    def read_dir(self, l1dir=None, openfile=True):
        # {{{
        if openfile:
            self.fin = h5py.File(self.fullfilename, 'r')
        if l1dir not in self.filedirs:
            self.filedirs[l1dir] = []
        keys = list(self.fin[l1dir].keys())
        for k in keys:
            self.filedirs[l1dir].append(k)
            self.read_dataset(l1dir=l1dir, dset=k, openfile=False)
        return
        # }}}

    #

    def read_dataset(self, l1dir=None, dset=None, openfile=False):
        # {{{
        if l1dir is None:
            raise IOError("No dir name given for reading dset")
        if dset is None:
            raise IOError("No dset name given for reading dset")
        if openfile:
            self.fin = h5py.File(self.fullfilename, 'r')
        itr = self.get_iterable(l1dir=l1dir, dsetname=dset)
        if itr not in list(self.filekeys.keys()):
            self.filekeys[itr] = [l1dir, dset]
        self.data[itr] = overlaps_vs_totalmass(
            dataset=self.fin[l1dir][dset].value)
        self.keys = list(self.data.keys())
        return
        # }}}

    #

    def get_dirdsetname_from_iterable(self, itr=None):
        # {{{
        if itr not in list(self.filekeys.keys()):
            raise RuntimeError("Iter %s not defined" % itr)
        return self.filekeys[itr]
        # }}}

    #

    def get_iterable(self, l1dir=None, dsetname=None):
        # {{{
        if l1dir == None:
            raise IOError("No dir name given for reading dset")
        if dsetname == None:
            raise IOError("No dir name given for reading dset")
        try:
            itr = self.string_from_dir(l1dir) + '/' + self.string_from_dset(
                dsetname)
        except:
            raise "Problem with get_iterable for %s, %s"
        self.filekeys[itr] = [l1dir, dsetname]
        return itr
        # }}}

    #

    def iterables(self):
        '''This is the main function to return the inner dataset'''
        return list(self.data.keys())

    #

    def overlaps_vs_totalmass(self, itr=None):
        # {{{
        if itr not in self.keys:
            raise IOError("Invalid iterable %s" % itr)
        return self.data[itr]
        # }}}

    #

    def overlaps(self, itr):
        # {{{
        return self.overlaps_vs_totalmass(itr=itr).O
        # }}}

    #

    def mismatches(self, itr):
        # {{{
        return 1. - self.overlaps_vs_totalmass(itr=itr).O
        # }}}

    # }}}


# Classes to manipulate overlap data for one simulation
class SimulationErrors():
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

    # {{{

    def __init__(self,
                 simdir=None,
                 matchdirs=['matches'],
                 verbose=True,
                 debug=False):
        # {{{
        self.verbose = verbose
        self.debug = debug
        self.simdir = simdir
        self.matchdirs = matchdirs
        if len(self.matchdirs):
            for d in self.matchdirs:
                if not os.path.exists(self.simdir + '/' + d):
                    raise IOError("Match directories do not exist for %s" %
                                  self.simdir)
        for d in self.matchdirs:
            self.read_all_overlaps(matchdir=d)
        # }}}

    #

    def read_all_overlaps(self, matchdir=None):
        # {{{
        try:
            if matchdir == None:
                matchdir = self.matchdirs[0]
        except IndexError:
            return
        pwd = cmd.getoutput('pwd')
        os.chdir(self.simdir)
        # Read in the matches
        self.read_ccer_overlaps(matchdir=matchdir)
        self.read_ccelev_overlaps(matchdir=matchdir)
        self.read_cceextrap_overlaps(matchdir=matchdir)
        if matchdir not in self.matchdirs:
            self.matchdirs.extend(matchdir)
        #
        os.chdir(pwd)
        return
        # }}}

    #

    def read_ccer_overlaps(self,
                           matchdir=None,
                           matchfile='OverlapsExtractionRadii.h5'):
        # {{{
        if self.verbose:
            print("Reading CCER overlaps frm %s/%s" % (matchdir, matchfile))
        self.ccer_overlaps = Overlaps(filename=matchfile,
                                      outdir=self.simdir + '/' + matchdir,
                                      verbose=self.verbose,
                                      debug=self.debug)
        self.ccer_dirnames = list(self.ccer_overlaps.filedirs.keys())
        self.ccer_dsetnames = {}
        for d in self.ccer_dirnames:
            self.ccer_dsetnames[d] = self.ccer_overlaps.filedirs[d]
        self.ccelevs =\
            [self.ccer_overlaps.string_from_dir(d) for d in self.ccer_dirnames]
        return
        # }}}

    #

    def read_ccelev_overlaps(self, matchdir=None, matchfile='OverlapsLevs.h5'):
        # {{{
        if self.verbose:
            print("Reading CCELev overlaps frm %s/%s" % (matchdir, matchfile))
        self.ccelev_overlaps = Overlaps(filename=matchfile,
                                        outdir=self.simdir + '/' + matchdir,
                                        verbose=self.verbose,
                                        debug=self.debug)
        self.ccelev_dirnames = list(self.ccelev_overlaps.filedirs.keys())
        self.ccelev_dsetnames = {}
        for d in self.ccelev_dirnames:
            self.ccelev_dsetnames[d] = self.ccelev_overlaps.filedirs[d]
        self.cceradii =\
            [self.ccer_overlaps.string_from_dir(
                d) for d in self.ccelev_dirnames]
        return
        # }}}

    #

    def read_cceextrap_overlaps(self,
                                matchdir=None,
                                matchfile='OverlapsExtrapolated.h5'):
        # {{{
        if self.verbose:
            print("Reading CCEExrapolation overlaps frm %s/%s" %
                  (matchdir, matchfile))
        self.cceextrap_overlaps = Overlaps(filename=matchfile,
                                           outdir=self.simdir + '/' + matchdir,
                                           verbose=self.verbose,
                                           debug=self.debug)
        #
        self.cceextrap_dirnames = list(self.cceextrap_overlaps.filedirs.keys())
        self.cceextrap_dsetnames = {}
        for d in self.cceextrap_dirnames:
            self.cceextrap_dsetnames[d] = self.cceextrap_overlaps.filedirs[d]
        #
        if self.debug:
            print(self.cceextrap_dsetnames)
        self.extraporders = []
        for dsetname in self.cceextrap_dsetnames[self.cceextrap_dirnames[0]]:
            tmpsplit = dsetname.split('_')
            # if self.debug == True: print "tmpsplit = ", tmpsplit
            for idx in range(len(tmpsplit)):
                if 'Extrapolated' in tmpsplit[idx]:
                    eo = tmpsplit[idx + 1].split('.')[0]
                    break
                elif 'Outer' in tmpsplit[idx]:
                    eo = tmpsplit[idx].split('.')[0]
                    break
            if eo not in self.extraporders:
                self.extraporders.append(eo)
            if self.debug == True:
                print("idx, tmpsplit = ", idx, tmpsplit[idx:])
            if self.debug:
                print("extraporders = ", self.extraporders)
        return
        # }}}

    #

    def ccer(self, key=None, noduplicate=True, onlyduplicate=False):
        # {{{
        if noduplicate and onlyduplicate:
            noduplicate = False
            if self.verbose:
                print("Only returning duplicate comparisons")
        # tmp = [[itr, self.ccer_overlaps.overlaps_vs_totalmass(itr=itr)]\
        #            for itr in self.ccer_overlaps.keys]
        tmp = {}
        for itr in self.ccer_overlaps.keys:
            # Filter by all keys in the list key
            if key is not None:
                if type(key) == str and key not in itr:
                    continue
                if type(key) == list:
                    contflag = False
                    for kk in key:
                        if kk not in itr:
                            contflag = True
                    if contflag:
                        continue
            #
            a, b = itr.split('/')[-1].split('_')
            if noduplicate and a == b:
                continue
            if onlyduplicate and a != b:
                continue
            tmp[itr] = self.ccer_overlaps.overlaps_vs_totalmass(itr=itr)
        return tmp
        # }}}

    #

    def ccelev(self, key=None, noduplicate=True, onlyduplicate=False):
        # {{{
        if noduplicate and onlyduplicate:
            noduplicate = False
            if self.verbose:
                print("Only returning duplicate comparisons")
        # tmp = [[itr, self.ccelev_overlaps.overlaps_vs_totalmass(itr=itr)]\
        #            for itr in self.ccelev_overlaps.keys]
        tmp = {}
        for itr in self.ccelev_overlaps.keys:
            # Filter by all keys in the list key
            if key is not None:
                if type(key) == str and key not in itr:
                    continue
                if type(key) == list:
                    contflag = False
                    for kk in key:
                        if kk not in itr:
                            contflag = True
                    if contflag:
                        continue
            #
            a, b = itr.split('/')[-1].split('_')
            if noduplicate and a == b:
                continue
            if onlyduplicate and a != b:
                continue
            tmp[itr] = self.ccelev_overlaps.overlaps_vs_totalmass(itr=itr)
        return tmp
        # }}}

    #

    def cceextrapolated(self, key=None, noduplicate=True, onlyduplicate=False):
        # {{{
        if noduplicate and onlyduplicate:
            noduplicate = False
            if self.verbose:
                print("Only returning duplicate comparisons")
        # tmp = [[itr, self.cceextrap_overlaps.overlaps_vs_totalmass(itr=itr)]\
        #            for itr in self.cceextrap_overlaps.keys]
        tmp = {}
        for itr in self.cceextrap_overlaps.keys:
            # Filter by all keys in the list key
            if key is not None:
                if type(key) == str and key not in itr:
                    continue
                if type(key) == list:
                    contflag = False
                    for kk in key:
                        if kk not in itr:
                            contflag = True
                    if contflag:
                        continue
            #
            a, b = itr.split('/')[-1].split('_')
            if noduplicate and a == b:
                continue
            if onlyduplicate and a != b:
                continue
            tmp[itr] = self.cceextrap_overlaps.overlaps_vs_totalmass(itr=itr)
        return tmp
        # }}}

    #

    def get_max_cce_mismatch(self,
                             spl_lev='Lev5',
                             spl_extrap=['N2', 'N3', 'N4'],
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
        # {{{
        if spl_ccer is None:
            spl_ccer = 'CceR%04d' % max([int(x[-4:]) for x in self.cceradii])
        #
        const_combinations = [[
            self.cceradii[0], spl_lev,
            'Lev%d' % (int(spl_lev[-1]) - 1)
        ], [self.cceradii[1], spl_lev,
            'Lev%d' % (int(spl_lev[-1]) - 1)],
                              [self.cceradii[0], self.cceradii[1], spl_lev],
                              [spl_ccer, spl_lev, 'N2'],
                              [spl_ccer, spl_lev, 'N3'],
                              [spl_ccer, spl_lev, 'N4']]
        const_funcs = [
            self.ccelev, self.ccelev, self.ccer, self.cceextrapolated,
            self.cceextrapolated, self.cceextrapolated
        ]
        if spl_lev != 'Lev5':
            const_combinations = [[
                self.cceradii[0], spl_lev,
                'Lev%d' % (int(spl_lev[-1]) + 1)
            ], [self.cceradii[1], spl_lev,
                'Lev%d' % (int(spl_lev[-1]) + 1)]] + const_combinations
            const_funcs = [self.ccelev, self.ccelev] + const_funcs
        #
        overlaps_all = []
        for idx, key_combo in enumerate(const_combinations):
            overlaps = const_funcs[idx](key=key_combo, noduplicate=True)
            if len(list(overlaps.values())) > 1:
                raise RuntimeError("keys %s gave %d resuls" %
                                   (key_combo, len(list(overlaps.keys()))))
            # proceed if only 1 dataset
            overlaps_all.append(list(overlaps.values())[0])
        #
        # Now to find the MIN overlaps from all data sets
        # Find the lowest total masses
        max_min_mtotal = -1e8
        for olap in overlaps_all:
            if max_min_mtotal < olap.M[0]:
                max_min_mtotal = max(max_min_mtotal, olap.M[0])
                min_masses = olap.X()
        num_taper_windows = olap.nWindows
        max_overlaps = np.ones(len(min_masses), 1 + num_taper_windows) * -1.
        for taperid in range(num_taper_windows):
            for mid, mass in enumerate(min_masses):
                olaps = [
                    obj.get_overlap_mass_taper(mass, taperid)
                    for obj in overlaps_all
                ]
                max_overlaps[mid, 1 + taperid] = max(olaps)
                max_overlaps[mid, 0] = mass
        #
        return overlaps_vs_totalmass(dataset=max_overlaps)
        # }}}

    # }}}


# Fitting-Factor storage classes
class EffectualnessAndBias():
    '''
    storage: tag/approximant/
    '''

    # {{{

    def __init__(self, outdir='.', verbose=True):
        # {{{
        self.verbose = verbose
        self.outdir = outdir
        self.data = {}
        # }}}

    #

    def read_data_from_combined_file(self, simtags=None, filename=None):
        """ Reads in the effectualness and parameter bias information from 
          a single file containing this information. Keep the data structure
          as before when reading from different files
        """
        # {{{
        if self.verbose:
            print("Reading ", filename, file=sys.stderr)
        f = h5py.File(os.path.join(self.outdir, filename), 'r')
        sims = list(f.keys())
        if simtags != None and len(simtags) > 0:
            newsims = []
            for s in sims:
                for st in simtags:
                    if st in s:
                        newsims.append(s)
                        break
            sims = newsims
        if len(sims) == 0:
            print("File %s does not contain data for " % filename, simtags)
            f.close()
            return
        for sim in sims:
            sim = str(sim)
            if sim not in list(self.data.keys()):
                self.data[sim] = {}
            simdata = f[sim]
            approxes = list(simdata.keys())
            for approx in approxes:
                approx = str(approx)
                if approx not in list(self.data[sim].keys()):
                    self.data[sim][approx] = {}
                data = simdata[approx].value
                for mtot, nr_q, nr_s1, nr_s2, ff, mc_diff, et_diff, s1_diff, s2_diff in data:
                    if mtot not in list(self.data[sim][approx].keys()):
                        # convert parameter bias data parameter values
                        nr_et = nr_q / (1. + nr_q)**2
                        nr_mc = mtot * nr_et**0.6
                        sig_mc = nr_mc - nr_mc * mc_diff
                        sig_et = nr_et - nr_et * et_diff
                        sig_s1 = nr_s1 - s1_diff
                        sig_s2 = nr_s2 - s2_diff
                        # out_row = np.array([mtot, sig_et, sig_s1, sig_s2,\
                        #                tmp_mc, tmp_et, tmp_s1, tmp_s2, olap])
                        out_row = np.array([
                            mtot, nr_et, nr_s1, nr_s2, sig_mc, sig_et, sig_s1,
                            sig_s2, ff
                        ])
                        self.data[sim][approx][mtot] = np.array([out_row])
        return
        # }}}

    #

    def read_data_from_file(self,
                            simtags=None,
                            filename=None,
                            keepalldata=False):
        """ Reads in data for ALL simulations and ALL approxes in a single output
            file. Stores them so they can be fetched from a simulation's nametag.
            If simtags are given (as a list), then only those simulations' data is 
            read, and nothing else is.
        """
        # {{{
        if self.verbose:
            print("Reading ", filename, file=sys.stderr)
        num_of_aux_cols, num_of_data_cols = 10, 9
        # NR signal parameters and clumns
        # mtotal num_of_aux_cols, eta 1, spin1z 2, spin2z 3
        # template parameters and columns
        # mchirp 5, eta 6, spin1z 7, spin2z 8
        f = h5py.File(self.outdir + '/' + filename, 'r')
        sims = list(f.keys())
        if simtags != None and len(simtags) > 0:
            newsims = []
            for s in sims:
                for st in simtags:
                    if st in s:
                        newsims.append(s)
                        break
            sims = newsims
        if len(sims) == 0:
            #raise IOError("File %s is empty!" % filename)
            print("File %s does not contain data for " % filename, simtags)
            f.close()
            return
        for sim in sims:
            sim = str(sim)
            if sim not in list(self.data.keys()):
                self.data[sim] = {}
            simdata = f[sim]
            approxes = list(simdata.keys())
            for approx in approxes:
                approx = str(approx)
                if approx not in list(self.data[sim].keys()):
                    self.data[sim][approx] = {}
                data = simdata[approx].value
                rowidx = 0
                for row in data:
                    # print "\n\n RowID = ", rowidx
                    rowidx += 1
                    for midx in np.arange(num_of_aux_cols, len(row), 2):
                        mtot = np.round(row[midx] * 100.) / 100.
                        olap = row[midx + 1]
                        if mtot < 0 or olap < 0:
                            continue
                        tmp_mc, tmp_et, tmp_s1, tmp_s2 = row[5:9]
                        sig_et, sig_s1, sig_s2 = row[1:4]
                        out_row = np.array([
                            mtot, sig_et, sig_s1, sig_s2, tmp_mc, tmp_et,
                            tmp_s1, tmp_s2, olap
                        ])
                        #
                        if mtot not in list(self.data[sim][approx].keys()):
                            self.data[sim][approx][mtot] = out_row
                        else:
                            try:
                                nr, nc = np.shape(self.data[sim][approx][mtot])
                                self.data[sim][approx][mtot] = \
                                    np.append(self.data[sim][approx][mtot], [
                                              out_row], axis=0)
                            except:
                                self.data[sim][approx][mtot] = \
                                    np.append([self.data[sim][approx][mtot]], [
                                              out_row], axis=0)
                # Keep only the point with the maximum overlap
                if not keepalldata:
                    for mtot in list(self.data[sim][approx].keys()):
                        tmp_data = self.data[sim][approx][mtot]
                        max_idx = np.where(
                            tmp_data[:, -1] == max(tmp_data[:, -1]))[0][0]
                        self.data[sim][approx][mtot] = np.array(
                            [tmp_data[max_idx, :]])
            #
        f.close()
        return
        # }}}

    #

    def effectualness_vs_totalmass(self, inkey=None, approx=None):
        # {{{
        for kk in list(self.data.keys()):
            if inkey in kk:
                break
        for app in list(self.data[kk].keys()):
            if approx in app:
                break
        masses = np.array(list(self.data[inkey][app].keys()))
        masses.sort()
        ff = np.array([max(self.data[inkey][app][mm][:, -1]) for mm in masses])
        return masses, ff
        # }}}

    #

    def best_match_parameters(self, inkey=None, approx=None):
        # {{{
        for kk in list(self.data.keys()):
            if inkey in kk:
                break
        for app in list(self.data[kk].keys()):
            if approx in app:
                break
        masses = np.array(list(self.data[inkey][app].keys()))
        masses.sort()
        ff = np.array([max(self.data[inkey][app][mm][:, -1]) for mm in masses])
        sig_mc = np.array([
            self.data[inkey][app][mm][np.where(
                self.data[inkey][app][mm][:, -1] ==
                max(self.data[inkey][app][mm][:, -1]))[0][0], -5]
            for mm in masses
        ])
        sig_et = np.array([
            self.data[inkey][app][mm][np.where(
                self.data[inkey][app][mm][:, -1] ==
                max(self.data[inkey][app][mm][:, -1]))[0][0], -4]
            for mm in masses
        ])
        sig_s1 = np.array([
            self.data[inkey][app][mm][np.where(
                self.data[inkey][app][mm][:, -1] ==
                max(self.data[inkey][app][mm][:, -1]))[0][0], -3]
            for mm in masses
        ])
        sig_s2 = np.array([
            self.data[inkey][app][mm][np.where(
                self.data[inkey][app][mm][:, -1] ==
                max(self.data[inkey][app][mm][:, -1]))[0][0], -2]
            for mm in masses
        ])
        # NR parameters are fixed for a simulation, so they dont need to be
        # accumulated
        nr_et = self.data[inkey][app][mm][0, 1]
        nr_q = (1. + (1. - 4. * nr_et)**0.5 - 2. * nr_et) / (2. * nr_et)
        nr_q = np.ones(len(ff)) * nr_q  # Its constant
        nr_et = np.ones(len(ff)) * nr_et
        nr_mc = masses * nr_et**0.6
        nr_s1 = np.ones(len(ff)) * \
            self.data[inkey][app][mm][0, 2]  # Its constant
        nr_s2 = np.ones(len(ff)) * \
            self.data[inkey][app][mm][0, 3]  # Its constant
        #
        return ff, nr_mc, nr_et, nr_s1, nr_s2, sig_mc, sig_et, sig_s1, sig_s2
        # }}}

    #

    def parameterbiases_vs_parameters(self,
                                      inkey=None,
                                      approx=None,
                                      chieff=False,
                                      total_mass=False):
        """
        Computes the relative/absolute differences between maximum-overlap
        parameters and those of the injections themselves. Default behavior
        is to return biases in 
        - chirp mass, 
        - eta, 
        - spin1, 
        - spin2, but
        the following input flags alter this : 
        - total_mass = true ==> instead of chirp mass, total mass diffs're returned
        """
        # {{{
        if chieff:
            from pycbc import pnutils
        else:
            print("Not using effective spin", file=sys.stdout)
        for kk in list(self.data.keys()):
            if inkey in kk:
                break
        for app in list(self.data[kk].keys()):
            if approx in app:
                break
        masses = np.array(list(self.data[inkey][app].keys()))
        masses.sort()
        ff = np.array([max(self.data[inkey][app][mm][:, -1]) for mm in masses])
        sig_mc = np.array([
            self.data[inkey][app][mm][np.where(
                self.data[inkey][app][mm][:, -1] ==
                max(self.data[inkey][app][mm][:, -1]))[0][0], -5]
            for mm in masses
        ])
        sig_et = np.array([
            self.data[inkey][app][mm][np.where(
                self.data[inkey][app][mm][:, -1] ==
                max(self.data[inkey][app][mm][:, -1]))[0][0], -4]
            for mm in masses
        ])
        if chieff:
            sig_m1, sig_m2 = pnutils.mchirp_eta_to_mass1_mass2(sig_mc, sig_et)
        if total_mass:
            sig_mt = sig_mc * sig_et**-0.6
        sig_s1 = np.array([
            self.data[inkey][app][mm][np.where(
                self.data[inkey][app][mm][:, -1] ==
                max(self.data[inkey][app][mm][:, -1]))[0][0], -3]
            for mm in masses
        ])
        sig_s2 = np.array([
            self.data[inkey][app][mm][np.where(
                self.data[inkey][app][mm][:, -1] ==
                max(self.data[inkey][app][mm][:, -1]))[0][0], -2]
            for mm in masses
        ])
        # NR parameters are fixed for a simulation, so they dont need to be
        # accumulated
        nr_et = self.data[inkey][app][mm][0, 1]
        nr_q = (1. + (1. - 4. * nr_et)**0.5 - 2. * nr_et) / (2. * nr_et)
        nr_q = np.ones(len(ff)) * nr_q  # Its constant
        nr_et = np.ones(len(ff)) * nr_et
        nr_mc = masses * nr_et**0.6
        if chieff:
            nr_m1, nr_m2 = pnutils.mchirp_eta_to_mass1_mass2(nr_mc, nr_et)
        nr_s1 = np.ones(len(ff)) * \
            self.data[inkey][app][mm][0, 2]  # Its constant
        nr_s2 = np.ones(len(ff)) * \
            self.data[inkey][app][mm][0, 3]  # Its constant
        #
        if chieff:
            # Compute PN effective spins
            #nr_seff = spins_to_PNeffective_spin(nr_m1, nr_m2, nr_s1, nr_s2)
            #sig_seff= spins_to_PNeffective_spin(sig_m1, sig_m2, sig_s1, sig_s2)
            nr_seff = spins_to_massweighted_spin(nr_m1, nr_m2, nr_s1, nr_s2)
            sig_seff = spins_to_massweighted_spin(sig_m1, sig_m2, sig_s1,
                                                  sig_s2)
        #mc_diff = (nr_mc-sig_mc)/nr_mc
        #eta_diff = (nr_et-sig_et)/nr_et
        #
        # NB: here 'mc_diff' really can contain either mchirp or mtotal differences,
        # please do not pay attention to the nomenclature 'mc'
        if total_mass:
            mc_diff = (sig_mt - masses) / masses
        else:
            mc_diff = (sig_mc - nr_mc) / nr_mc
        eta_diff = (sig_et - nr_et) / nr_et
        # Handle spins specially for q = 1
        if 'q1' not in inkey:
            if chieff:
                s1_diff = sig_seff - nr_seff
            else:
                s1_diff = sig_s1 - nr_s1  # nr_s1-sig_s1
            s2_diff = sig_s2 - nr_s2  # nr_s2-sig_s2
        else:
            s1_diff, s2_diff = np.zeros(len(nr_s1)), np.zeros(len(nr_s2))
            s11d, s12d = sig_s1 - nr_s1, sig_s2 - nr_s1  # nr_s1 - sig_s1, nr_s1 - sig_s2
            s21d, s22d = sig_s1 - nr_s2, sig_s2 - nr_s2  # nr_s2 - sig_s1, nr_s2 - sig_s2
            s1122rms = (s11d**2 + s22d**2)**0.5
            s1221rms = (s12d**2 + s21d**2)**0.5
            mask = s1122rms < s1221rms
            s1_diff[mask] = s11d[mask]
            s2_diff[mask] = s22d[mask]
            mask = s1122rms >= s1221rms
            s1_diff[mask] = s12d[mask]
            s2_diff[mask] = s21d[mask]
            if chieff:
                s1_diff = sig_seff - nr_seff
        return masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff
        # }}}

    #

    def effectualness_vs_parameters(self, inkey=None, approx=None):
        # {{{
        for kk in list(self.data.keys()):
            if inkey in kk:
                break
        for app in list(self.data[kk].keys()):
            if approx in app:
                break
        masses = np.array(list(self.data[inkey][app].keys()))
        masses.sort()
        ff = np.array([max(self.data[inkey][app][mm][:, -1]) for mm in masses])
        nr_et = self.data[inkey][app][mm][0, 1]
        nr_q = (1. + (1. - 4. * nr_et)**0.5 - 2. * nr_et) / (2. * nr_et)
        nr_q = np.ones(len(ff)) * nr_q  # Its constant
        nr_s1 = np.ones(len(ff)) * \
            self.data[inkey][app][mm][0, 2]  # Its constant
        nr_s2 = np.ones(len(ff)) * \
            self.data[inkey][app][mm][0, 3]  # Its constant
        return masses, nr_q, nr_s1, nr_s2, ff
        # }}}

    # }}}
