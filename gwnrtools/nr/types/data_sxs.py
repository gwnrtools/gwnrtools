#!/usr/bin/env python
# Copyright (C) 2018 Prayush Kumar
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
from __future__ import absolute_import

import os
import h5py

import numpy as np

from .single_mode import nr_mode
######################################################################
verbose = False


## @nr_data pyexample
#  Documentation for this module.
#
## Alias "nr_wave" to "nr_strain". This way we have a complete set of
## - "nr_data" as raw data containers for NR data
## - "nr_mode" as manipulation class for single NR modes
## - "nr_strain" / "nr_wave" as manipulation class for GW strain
## Each class depends on the all previous ones.
#
######################################################################
######################################################################
#
#   Make a class to read NR data
#
######################################################################
######################################################################
class nr_data():
    #{{{
    def __init__(self,
                 filename,
                 filetype='HDF',
                 wavetype='Auto',
                 ex_order=3,
                 group_name=None,
                 UNDERSTOOD_TYPES=[
                     'CCE', 'Extrapolated', 'FiniteRadius', 'NoGroup'
                 ],
                 modeLmin=2,
                 modeLmax=4,
                 skipM0=True,
                 delta_t=1.0,
                 verbose=0):
        """
#### Input Options:
### 1. filename = FULL PATH to NR data file
### 2. filetypes passed should be : 'HDF5' , 'ASCII' or 'DataSet'

#### 2.1 For HDF5 files:-
#### Between "wavetype", "ex_order", and "group_name", provide:
           a) for CCE waveforms, wavetype=CCE. It uses highest-R data.
           b) for extrapolated waveforms, wavetype='Extrapolated' and ex_order=?
           c) for finite-radii waveforms, wavetype='FiniteRadius'
           d) for datasets without groups, wavetype='NoGroup'
           e) (experimental) Determine automatically: wavetype='Auto' (default)
           f) for non-SXS waveforms, group_name='...'
    Note: group_name overwrites other two options. So one can also provide
                group_name = 'Extrapolated_N3.dir' OR
                group_name = 'CceR0350.dir', etc.

#### 2.2 For ASCII files:-
####  () Provide wavetype = 'regex'. This assumes:
           a) filename is a REGEX expression that can be formatted with (modeL, modeM) integer tuples

#### 2.3 For DataSet:-
####  () Provide wavetype = 'dict'. This assumes:
           a) This dictionary should have [l][m] modes as Nx2 or Nx3 matrices

### 6. modeLmin, modeLmax: Range of l-modes of strain to use
            (cannot use arbitrary ones yet)
### 7. skipM0: Skip m=0 (DC) modes (Default: True)

        """
        ## Check inputs
        if not os.path.exists(filename):
            raise IOError(
                "HDF file {} cannot be read: does not exist".format(filename))
        if os.path.getsize(filename) == 0:
            raise IOError(
                "HDF file {} cannot be read: file empty".format(filename))

        self.verbose = verbose
        self.skipM0 = skipM0
        self.UNDERSTOOD_TYPES = UNDERSTOOD_TYPES
        self.filename = filename
        self.filetype = filetype
        self.wavetype = wavetype
        self.ex_order = ex_order
        self.group_name = group_name

        self.modeLmin = modeLmin
        self.modeLmax = modeLmax
        self.delta_t = delta_t

        self.fin = h5py.File(self.filename, 'r')
        self.modes = {}
        self.read_nr_data()
        return

    def read_nr_data(self):
        ##{{{
        if 'HDF' in self.filetype:
            self.read_nr_data_hdf5()
            return self
        elif 'DataSet' in self.filetype:
            self.read_nr_data_dataset()
            return self
        elif 'ASCII' in self.filetype:
            self.read_nr_data_ascii()
            return self
        raise IOError("Could not decipher filetype {}".format(self.filetype))
        ##}}}
    def read_nr_data_ascii(self):
        ##{{{
        if 'regex' in self.wavetype:
            if self.verbose > 0:
                print("Reading NR data in ASCII from {}".format(self.filename))
            ####
            ## Read modes
            MAXLEN, MINLEN = -1, 1e100
            ## Create output dictionary
            self.modes = {}
            ## Loop over modes
            for modeL in np.arange(2, self.modeLmax + 1):
                self.modes[modeL] = {}
                for modeM in np.arange(-1 * modeL, modeL + 1):
                    if self.skipM0 and modeM == 0: continue
                    ## Get dataset for this mode
                    try:
                        mdata = np.loadtxt(self.filename % (modeL, modeM))
                        if self.verbose > 2:
                            print("\t\t\tShape of data read is ",
                                  np.shape(mdata))
                    except:
                        if self.verbose > 0:
                            print("WARNING: Ignoring mode ({},{})".format(
                                modeL, modeM))
                        continue
                    ## Initialize a nr_mode class with this dataset
                    self.modes[modeL][modeM] = nr_mode(mdata,
                                                       delta_t=self.delta_t,
                                                       verbose=self.verbose)
                    ## Measure maximum duration of any mode
                    MINLEN = np.minimum(
                        MINLEN, self.modes[modeL][modeM].data_duration())
                    MAXLEN = np.maximum(
                        MAXLEN, self.modes[modeL][modeM].data_duration())
                self.MAX_DURATION_M = MAXLEN
                self.MIN_DURATION_M = MINLEN
                ## If no (l,m) mode is found for a given (l), pop it
                if len(self.modes[modeL]) == 0: self.modes.pop(modeL)
                ## If no (l) item exists in self.modes, no data has been read at ALL!
                if len(self.modes) == 0: raise IOError("No (ASCII) data READ!")
        else:
            raise IOError(
                "ASCII datafile not accepted in this format. Read HELP.")
        return self
        ##}}}
    def read_nr_data_dataset(self):
        ##{{{
        if 'dict' in self.wavetype:
            if self.verbose > 0:
                print("Reading NR data in ASCII from {}".format(self.filename))
            ## filename is NOT really filename, its a DICTIONARY
            dataset = self.filename
            ## Read Modes
            MAXLEN, MINLEN = -1, 1e100
            ## Create output dictionary
            self.modes = {}
            for modeL in dataset:
                modeL = int(modeL)
                self.modes[modeL] = {}
                for modeM in dataset[modeL]:
                    modeM = int(modeM)
                    if self.skipM0 and modeM == 0: continue
                    ## Get dataset for this mode
                    try:
                        mdata = dataset[modeL][modeM]
                        if self.verbose > 2:
                            print("\t\t\tShape of data read is ",
                                  np.shape(mdata))
                    except:
                        if self.verbose > 0:
                            print("WARNING: Ignoring mode ({},{})".format(
                                modeL, modeM))
                        continue
                    ## Initialize a nr_mode class with this dataset
                    self.modes[modeL][modeM] = nr_mode(mdata,
                                                       delta_t=self.delta_t,
                                                       verbose=self.verbose)
                    ## Measure maximum duration of any mode
                    MINLEN = np.minimum(
                        MINLEN, self.modes[modeL][modeM].data_duration())
                    MAXLEN = np.maximum(
                        MAXLEN, self.modes[modeL][modeM].data_duration())
            self.MAX_DURATION_M = MAXLEN
            self.MIN_DURATION_M = MINLEN
            ###############
        else:
            raise IOError("DataSets not accepted in this format. Read HELP.")
        return self
        ##}}}
    def get_nr_data_hdf5_wavetype(self):
        ##{{{
        wavetype = self.wavetype
        _wavetype = None
        if str(wavetype) in self.UNDERSTOOD_TYPES:
            _wavetype = wavetype
        elif str(wavetype) == 'Auto':
            # Decide the wavetype from filename
            fname = self.filename.split('/')[-1]
            if '_Asymptotic_GeometricUnits' in fname:
                _wavetype = 'Extrapolated'
            elif 'Cce' in fname:
                _wavetype = 'CCE'
            elif 'FiniteRad' in fname:
                _wavetype = 'FiniteRadius'
            elif 'HDF' in self.filetype:
                fgrps = [str(grptmp) for grptmp in list(self.fin.keys())]
                if 'Y_l2_m2.dat' in fgrps: _wavetype = 'NoGroup'
        else: raise IOError("Could not figure out wavetype")
        self.wavetype = _wavetype
        return self
        ##}}}
    def get_nr_data_hdf5_groupname(self):
        ##{{{
        if self.group_name is not None and len(self.group_name) != 0:
            return self
        self.get_nr_data_hdf5_wavetype()
        f = self.fin
        if self.wavetype == 'CCE':
            grp = 'CceR%04d.dir' % max(
                [int(k.split('.dir')[0][-4:]) for k in list(f.keys())])
            self.group_name = grp
            return self
        elif self.wavetype == 'Extrapolated':
            for kdx, k in enumerate(f.keys()):
                if 'Extrapolated' not in str(k): continue
                last_k = k
                try:
                    n = int(k[-1])
                except:
                    try:
                        n = int(k.split('.dir')[0][-1])
                    except:
                        if self.verbose > 1:
                            print(
                                "\t\t.. tested (wavetype is extrapolated) groupname: {}"
                                .format(k))
                        raise IOError(
                            "Could not find the group for extrapolated waveforms"
                        )
                if self.ex_order == n:
                    self.group_name = k
                    return self
            self.group_name = last_k
            return self
        elif self.wavetype == 'FiniteRadius':
            grp = 'R%04d.dir' % max(
                [int(k.split('.dir')[0][-4:]) for k in list(f.keys())])
            self.group_name = grp
            return self
        ## If all fails ...
        raise KeyError("Groupname not found")
        ##}}}
    def read_nr_data_hdf5(self):
        ##{{{
        ## Get prerequisites
        wavetype = self.get_nr_data_hdf5_wavetype().wavetype
        group_name = self.get_nr_data_hdf5_groupname().group_name
        ##
        if self.verbose > 1:
            print("Reading NR data in HDF5 from {}".format(self.filename))
        if wavetype != 'NoGroup':
            if self.verbose > 1:
                print("\tReading from {} out of ".format(group_name),
                      list(self.fin.keys()))
            wavedata = self.fin[group_name]
        else:
            wavedata = self.fin
        ##
        self.read_nr_mode_data_hdf5_group(wavedata)
        return self
        ##}}}
    def read_nr_mode_data_hdf5_group(self, wavedata):
        ##{{{
        ## Create output dictionary
        self.modes = {}
        if True:
            ## Read modes
            MAXLEN, MINLEN = -1, 1e100
            for modeL in np.arange(self.modeLmin, self.modeLmax + 1):
                self.modes[modeL] = {}
                for modeM in np.arange(modeL, -1 * modeL - 1, -1):
                    if self.skipM0 and modeM == 0: continue
                    ## Get dataset for this mode
                    try:
                        if self.verbose > 2:
                            print("\t\tTrying to read: %d,%d mode" %
                                  (modeL, modeM))
                        mdata = wavedata['Y_l{}_m{}.dat'.format(modeL,
                                                                modeM)].value
                        if self.verbose > 2:
                            print("\t\tShape of data read is ",
                                  np.shape(mdata))
                    except:
                        if self.verbose > 0:
                            print("WARNING: Ignoring mode ({},{})".format(
                                modeL, modeM))
                        continue
                    ## Initialize a nr_mode class with this dataset
                    self.modes[modeL][modeM] = nr_mode(mdata,
                                                       delta_t=self.delta_t,
                                                       verbose=self.verbose)
                    ## Measure maximum duration of any mode
                    MINLEN = np.minimum(
                        MINLEN, self.modes[modeL][modeM].data_duration())
                    MAXLEN = np.maximum(
                        MAXLEN, self.modes[modeL][modeM].data_duration())
            self.MAX_DURATION_M = MAXLEN
            self.MIN_DURATION_M = MINLEN
        return self
        ##}}}

    #}}}
