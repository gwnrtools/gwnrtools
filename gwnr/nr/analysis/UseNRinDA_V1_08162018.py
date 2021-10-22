#!/usr/bin/env python
# Copyright (C) 2015 Prayush Kumar
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

import matplotlib
try:
    matplotlib.use('Agg')
except:
    pass
import os
import sys

from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.misc import derivative
from scipy.optimize import bisect, brentq
from scipy.integrate import simps, cumtrapz

import subprocess as cmd
import string
from numpy import *
import numpy as np
from numpy.random import random
import h5py

from math import pow

from utils import phase_from_polarizations, frequency_from_polarizations, amplitude_from_polarizations
from pycbc.types import FrequencySeries, TimeSeries, real_same_precision_as, complex_same_precision_as, Array
import lal

verbose = False


def convert_TimeSeries_to_lalREAL8TimeSeries(h, name=None):
    tmp = lal.CreateREAL8Sequence(len(h))
    tmp.data = np.array(h.data)
    hnew = lal.REAL8TimeSeries()
    hnew.data = tmp
    hnew.deltaT = h.delta_t
    hnew.epoch = h._epoch
    if name is not None:
        hnew.name = name
    return hnew


def convert_lalREAL8TimeSeries_to_TimeSeries(h):
    return TimeSeries(h.data.data,
                      delta_t=h.deltaT,
                      copy=True,
                      epoch=h.epoch,
                      dtype=h.data.data.dtype)


def zero_pad_beginning(h, steps=1):
    h.data = np.roll(h.data, steps)
    return h


######################################################################
######################################################################

#     Make a class for basic manipulations of SpEC NR waveforms      #


######################################################################
######################################################################
class nr_wave():
    # {{{
    def __init__(self,
                 filename=None,
                 filetype='HDF5',
                 wavetype='Auto',
                 ex_order=3,
                 group_name=None,
                 modeLmin=2,
                 modeLmax=8,
                 skipM0=True,
                 sample_rate=8192,
                 time_length=32,
                 rawdelta_t=-1,
                 totalmass=None,
                 inclination=0,
                 phi=0,
                 distance=1.e6,
                 verbose=False,
                 debug=False):
        """
    #### Assumptions:
    ### 1. The nr waveform file is uniformly sampled
    ### 2. wavetypes passed should be :
    ###     CCE , Extrapolated , FiniteRadius , NoGroup, Auto
    ###     Auto : from filename figure out the file type
    ### 3. filetypes passed should be : ASCII , HDF5 , DataSet

    ###### About conventions:
    ### 1. Modes themselves are not amplitude-scaled. Only their time-axis is
    ###    rescaled.
    ### 2. ...

        """
        # {{{
        ##################################################################
        #   0. Ensure inputs are correct
        ##################################################################
        if 'dataset' not in filetype:
            if filename is not None and not os.path.exists(filename):
                raise IOError("Please provide data file!")
            if verbose:
                print("\n Reading From Filename=%s" % filename,
                      file=sys.stderr)

        # Datafile details
        self.verbose = verbose
        self.debug = debug
        self.filename = filename
        self.filetype = filetype
        self.modeLmax = modeLmax
        self.modeLmin = modeLmin
        self.skipM0 = skipM0

        # Extraction parameters
        self.ex_order = ex_order
        self.group_name = group_name

        # Data analysis parameters
        self.sample_rate = sample_rate
        self.time_length = time_length
        self.dt = 1. / self.sample_rate
        self.df = 1. / self.time_length
        self.n = int(self.sample_rate * self.time_length)
        if self.verbose:
            print("self.sample-rate & time_len = ",
                  self.sample_rate,
                  self.time_length,
                  file=sys.stderr)
        if self.verbose:
            print("self.n = ", self.n, file=sys.stderr)

        # Binary parameters
        self.totalmass = None
        self.inclination = inclination
        self.phi = phi
        self.distance = distance
        if self.verbose:
            print(" >> Input mass, inc, phi, dist = ",
                  totalmass,
                  self.inclination,
                  self.phi,
                  self.distance,
                  file=sys.stderr)

        sys.stderr.flush()
        ##################################################################
        #   1. Figure out what the data-storage type (wavetype) is
        ##################################################################
        self.wavetype = None
        if str(wavetype) in ['CCE', 'Extrapolated', 'FiniteRadius', 'NoGroup']:
            self.wavetype = wavetype
        elif str(wavetype) == 'Auto':
            # Decide from filename the wavetype
            fname = self.filename.split('/')[-1]
            if 'rhOverM_Asymptotic_GeometricUnits' in fname:
                self.wavetype = 'Extrapolated'
            elif 'Cce' in fname:
                self.wavetype = 'CCE'
            elif 'FiniteRadii' in fname:
                self.wavetype = 'FiniteRadius'
            elif 'HDF' in self.filetype:
                ftmp = h5py.File(self.filename, 'r')
                fgrps = [str(grptmp) for grptmp in fgrpstmp]
                if 'Y_l2_m2.dat' in fgrps:
                    self.wavetype = 'NoGroup'
        #
        if self.wavetype == None:
            raise IOError("Could not figure out wavetype")
        if self.verbose:
            print(self.wavetype, file=sys.stderr)

        sys.stderr.flush()
        ##################################################################
        #   2. Read the data from the file. Read all modes.
        ##################################################################
        #
        if 'HDF' in self.filetype:
            if self.verbose:
                print(" > Reading NR data in HDF5 from %s" % self.filename,
                      file=sys.stderr)

            # Read data file
            self.fin = h5py.File(self.filename, 'r')
            if self.wavetype != 'NoGroup':
                grp = self.get_groupname()
                print("FOUND GROUP = ", grp, type(grp))
                if self.verbose:
                    print(("From %s out of " % grp),
                          list(self.fin.keys()),
                          file=sys.stderr)
                wavedata = self.fin[grp]
            else:
                wavedata = self.fin

            # Read modes
            self.rawtsamples, self.rawmodes_real, self.rawmodes_imag = {}, {}, {}
            for modeL in np.arange(2, self.modeLmax + 1):
                self.rawtsamples[modeL] = {}
                self.rawmodes_real[modeL], self.rawmodes_imag[modeL] = {}, {}
                for modeM in np.arange(modeL, -1 * modeL - 1, -1):
                    if self.skipM0 and modeM == 0:
                        continue
                    mdata = wavedata['Y_l%d_m%d.dat' % (modeL, modeM)].value
                    if self.verbose:
                        print("Reading %d,%d mode" % (modeL, modeM),
                              file=sys.stderr)
                    #
                    ts = mdata[:, 0]
                    hp_int = InterpolatedUnivariateSpline(ts, mdata[:, 1])
                    hc_int = InterpolatedUnivariateSpline(ts, mdata[:, 2])

                    # Hard-coded to re-sample at initial dt
                    if rawdelta_t <= 0:
                        self.rawdelta_t = ts[1] - ts[0]
                    else:
                        self.rawdelta_t = rawdelta_t

                    #
                    self.rawtsamples[modeL][modeM] = TimeSeries(
                        np.arange(ts.min(), ts.max(), self.rawdelta_t),
                        delta_t=self.rawdelta_t,
                        epoch=0)
                    self.rawmodes_real[modeL][modeM] = TimeSeries(
                        hp_int(self.rawtsamples[2][2]),
                        delta_t=self.rawdelta_t,
                        epoch=0)
                    self.rawmodes_imag[modeL][modeM] = TimeSeries(
                        hc_int(self.rawtsamples[2][2]),
                        delta_t=self.rawdelta_t,
                        epoch=0)
        #
        elif 'dataset' in self.filetype:
            raise IOError("datasets not supported yet!")
            data = self.filename
            ts = data[:, 0] - data[0, 0]
            hp_int = InterpolatedUnivariateSpline(ts, data[:, 1])
            hc_int = InterpolatedUnivariateSpline(ts, data[:, 2])
            # Hard-coded to re-sample at dt = 1M
            if rawdelta_t <= 0:
                self.rawdelta_t = 1.
            else:
                self.rawdelta_t = rawdelta_t
            self.rawtsamples = TimeSeries(arange(0, ts.max()),
                                          delta_t=self.rawdelta_t,
                                          epoch=0)
            self.rawhp = TimeSeries(array(
                [hp_int(t) for t in self.rawtsamples]),
                                    delta_t=self.rawdelta_t,
                                    epoch=0)
            self.rawhc = TimeSeries(array(
                [hc_int(t) for t in self.rawtsamples]),
                                    delta_t=self.rawdelta_t,
                                    epoch=0)
            if self.verbose:
                print("times go from %f to %f" % (min(ts), max(ts)),
                      file=sys.stderr)
                print("rawhp Min = %e, Max = %e" %
                      (min(self.rawhp), max(self.rawhp)),
                      file=sys.stderr)
            #
        elif 'ASCII' in self.filetype:
            raise IOError("ASCII datafile not accepted yet!")
            if self.verbose:
                print("Reading NR data in ASCII from %s" % self.filename,
                      file=sys.stderr)

            # Read modes
            self.rawtsamples, self.rawmodes_real, self.rawmodes_imag = {}, {}, {}
            for modeL in np.arange(2, self.modeLmax + 1):
                self.rawtsamples[modeL] = {}
                self.rawmodes_real[modeL], self.rawmodes_imag[modeL] = {}, {}
                for modeM in np.arange(-1 * modeL, modeL + 1):
                    if self.skipM0 and modeM == 0:
                        continue
                    mdata = np.loadtxt(self.filename % (modeL, modeM))
                    if self.verbose:
                        print(np.shape(mdata), file=sys.stderr)
                    #
                    ts = mdata[:, 0]
                    hp_int = InterpolatedUnivariateSpline(ts, mdata[:, 1])
                    hc_int = InterpolatedUnivariateSpline(ts, mdata[:, 2])

                    # Hard-coded to re-sample at initial dt
                    if rawdelta_t <= 0:
                        self.rawdelta_t = ts[1] - ts[0]
                    else:
                        self.rawdelta_t = rawdelta_t

                    #
                    self.rawtsamples[modeL][modeM] = TimeSeries(
                        np.arange(ts.min(), ts.max(), self.rawdelta_t),
                        delta_t=self.rawdelta_t,
                        epoch=0)
                    self.rawmodes_real[modeL][modeM] = TimeSeries(
                        array([hp_int(t) for t in self.rawtsamples]),
                        delta_t=self.rawdelta_t,
                        epoch=0)
                    self.rawmodes_imag[modeL][modeM] = TimeSeries(
                        array([hc_int(t) for t in self.rawtsamples]),
                        delta_t=self.rawdelta_t,
                        epoch=0)
        #
        self.rescaled_hp = None
        self.rescaled_hc = None
        if totalmass:
            self.rescale_to_totalmass(totalmass)
            self.totalmass = totalmass

        try:
            self.fin.close()
        except:
            pass
        # }}}

    #

    def get_groupname(self):
        # {{{
        if self.group_name is not None:
            return self.group_name
        print("WAVETYPE = ", self.wavetype)
        #
        f = self.fin
        if self.wavetype == 'CCE':
            grp = 'CceR%04d.dir' % max(
                [int(k.split('.dir')[0][-4:]) for k in list(f.keys())])
            return grp
        #
        elif self.wavetype == 'Extrapolated':
            for k in list(f.keys()):
                if self.debug:
                    print("Going to test: ", k)
                try:
                    n = int(k[-1])
                except:
                    try:
                        n = int(k.split('.dir')[0][-1])
                    except:
                        if self.verbose:
                            print(" .. Extrapolated groupname is %" % k,
                                  file=sys.stderr)
                        raise IOError(
                            "Could not find the group for extrapolated waveforms"
                        )
                if self.debug:
                    print("Found n = ", n, " for ", k)
                if self.ex_order == n:
                    return k
        #
        elif self.wavetype == 'FiniteRadius':
            grp = 'R%04d.dir' % max(
                [int(k.split('.dir')[0][-4:]) for k in list(f.keys())])
            return grp
        #
        raise KeyError("Groupname not found")
        # }}}

    # ##################################################################
    # Basic waveform manipulation
    # ##################################################################

    def rescale_mode(self, M=None, distance=None, modeL=2, modeM=2):
        """ This function rescales the given mode to input mass value. No distance
            scaling is done. This function is meant for usage in
            amplitude-scaling-invariant calculations.
            Note that this function does NOT reset internal total-mass value,
            since it operates on a single mode, and reseting mass/distance etc
            would make things inconsistent
        """
        # {{{
        if (self.totalmass == M or M is None) and self.totalmass != None:
            return [
                self.rescaledmodes_real[modeL][modeM],
                self.rescaledmodes_imag[modeL][modeM]
            ]

        MinSecs = M * lal.MTSUN_SI
        scaleFac = 1

        if self.verbose:
            print(" Rescaling mode %d, %d for M = %f" % (modeL, modeM, M),
                  file=sys.stderr)
        rawmode_time = self.rawtsamples[modeL][modeM]
        rawmode_real = self.rawmodes_real[modeL][modeM]
        rawmode_imag = self.rawmodes_imag[modeL][modeM]

        end_t = rawmode_time.data[-1] * MinSecs
        start_t = rawmode_time.data[0] * MinSecs
        end_t_n = int((end_t - start_t) / self.dt)

        rescaled_hpI = InterpolatedUnivariateSpline(
            rawmode_time.data * MinSecs - start_t, rawmode_real.data, k=3)
        rescaled_hcI = InterpolatedUnivariateSpline(
            rawmode_time.data * MinSecs - start_t, rawmode_imag.data, k=3)

        tmp_rescaled_hp = rescaled_hpI(np.arange(end_t_n) * self.dt)
        tmp_rescaled_hc = rescaled_hcI(np.arange(end_t_n) * self.dt)

        try:
            tmp_rescaled_hp = np.concatenate(
                (tmp_rescaled_hp, np.zeros(self.n - end_t_n)))
            tmp_rescaled_hc = np.concatenate(
                (tmp_rescaled_hc, np.zeros(self.n - end_t_n)))
        except ValueError:
            raise ValueError("self.n = %d, end_t_n = %d" % (self.n, end_t_n))

        if self.verbose:
            print(self.n, end_t_n, file=sys.stderr)

        return [
            TimeSeries(tmp_rescaled_hp * scaleFac, delta_t=self.dt, epoch=0),
            TimeSeries(tmp_rescaled_hc * scaleFac, delta_t=self.dt, epoch=0)
        ]
        # }}}

    #

    def rescale_wave(self, M=None, inclination=None, phi=None, distance=None):
        """ Rescale modes and polarizations to given mass, angles, distance.
            Note that this function re-sets the stored values of binary parameters
            and so all future calculations will assume new values unless otherwise
            specified.
        """
        # {{{
        #
        # If MASS has changed, rescale modes
        #
        if self.totalmass != M and M is not None:
            # Rescale the time-axis for all modes
            self.rescaledmodes_real, self.rescaledmodes_imag = {}, {}
            for modeL in np.arange(2, self.modeLmax + 1):
                self.rescaledmodes_real[modeL], self.rescaledmodes_imag[
                    modeL] = {}, {}
                for modeM in np.arange(-1 * modeL, modeL + 1):
                    if self.skipM0 and modeM == 0:
                        continue
                    self.rescaledmodes_real[modeL][modeM], \
                        self.rescaledmodes_imag[modeL][modeM] = \
                        self.rescale_mode(M, modeL=modeL, modeM=modeM)
            self.totalmass = M
        elif self.totalmass == None and M == None:
            raise IOError("Please provide a total-mass value to rescale")

        #
        # Now rescale with distance and ANGLES
        #
        if inclination is not None:
            self.inclination = inclination
        if phi is not None:
            self.phi = phi
        if distance is not None:
            self.distance = distance

        # Mass / distance scaling pre-factor
        scalefac = self.totalmass * lal.MRSUN_SI / self.distance / lal.PC_SI

        # Orbital phase at the time of merger (time of amplitude peak)
        amp22 = self.get_mode_amplitude(totalmass=self.totalmass,
                                        modeL=2,
                                        modeM=2,
                                        dimensionless=False)
        iPeak, aPeak = self.get_peak_amplitude(amp=amp22)
        #phase22 = self.get_mode_phase(totalmass=self.totalmass, dimensionless=False)
        #phiOrbMerger = phase22[iPeak] / 2.

        # Combine all modes
        hp, hc = np.zeros(self.n, dtype=float), np.zeros(self.n, dtype=float)

        for modeL in np.arange(2, self.modeLmax + 1):
            for modeM in np.arange(-1 * modeL, modeL + 1):
                if self.skipM0 and modeM == 0:
                    continue
                # h+ - \ii hx = \Sum Ylm * hlm
                ArcTan2Merger = np.arctan2(
                    self.rescaledmodes_imag[modeL][modeM].data[iPeak],
                    self.rescaledmodes_real[modeL][modeM].data[iPeak])

                if self.verbose:
                    print("arctan2 at merger after: ", ArcTan2Merger)

                #curr_ylm = np.exp(1 * modeM * phiOrbMerger * 1j)
                curr_ylm = np.exp(-1 * ArcTan2Merger * 1j)
                if self.debug and curr_ylm:
                    print(curr_ylm / np.abs(curr_ylm))
                curr_ylm *= lal.SpinWeightedSphericalHarmonic(
                    self.inclination, self.phi, -2, modeL, modeM)
                if self.debug and curr_ylm:
                    print(curr_ylm / np.abs(curr_ylm))
                #
                if self.debug:
                    print((np.arctan2(curr_ylm.imag, curr_ylm.real) +
                           (modeM * phiOrbMerger)) / np.pi,
                          (np.arctan2(curr_ylm.imag, curr_ylm.real) +
                           (ArcTan2Merger)) / np.pi)
                    print((np.arctan2(curr_ylm.imag, curr_ylm.real) -
                           (modeM * phiOrbMerger)) / np.pi,
                          (np.arctan2(curr_ylm.imag, curr_ylm.real) -
                           (ArcTan2Merger)) / np.pi)
                ##
                hp += self.rescaledmodes_real[modeL][modeM].data * curr_ylm.real - \
                    self.rescaledmodes_imag[modeL][modeM].data * curr_ylm.imag
                hc -= self.rescaledmodes_real[modeL][modeM].data * curr_ylm.imag + \
                    self.rescaledmodes_imag[modeL][modeM].data * curr_ylm.real
        if self.debug:
            print("END\n\n")

        # Scale amplitude by mass and distance factors
        self.rescaled_hp = TimeSeries(scalefac * hp, delta_t=self.dt, epoch=0)
        self.rescaled_hc = TimeSeries(scalefac * hc, delta_t=self.dt, epoch=0)

        return [self.rescaled_hp, self.rescaled_hc]
        # }}}

    #

    def rescale_to_totalmass(self, M):
        """ Rescales the waveform to a different total-mass than currently. The
        values for different angles are set to internal values provided earlier, e.g.
        during object initialization.
        """
        if not hasattr(self, 'inclination') or self.inclination is None:
            raise RuntimeError(
                "Cannot rescale total-mass without setting inclination")
        elif not hasattr(self, 'phi') or self.phi is None:
            raise RuntimeError("Cannot rescale total-mass without setting phi")
        elif not hasattr(self, 'distance') or self.distance is None:
            raise RuntimeError(
                "Cannot rescale total-mass without setting distance")
        print("DEBUG rescale_to_totalmass: Rescaling to M=%.2f" % M)
        return self.rescale_wave(M,
                                 inclination=self.inclination,
                                 phi=self.phi,
                                 distance=self.distance)

    #

    def rescale_to_distance(self, distance):
        """ Rescales the waveform to a different distance than currently. The
        values for different angles, masses are set to internal values provided
        earlier, e.g. during object initialization.
        """
        if not hasattr(self, 'inclination') or self.inclination is None:
            raise RuntimeError(
                "Cannot rescale distance without setting inclination")
        elif not hasattr(self, 'phi') or self.phi is None:
            raise RuntimeError("Cannot rescale distance without setting phi")
        elif not hasattr(self, 'totalmass') or self.totalmass is None:
            raise RuntimeError(
                "Cannot rescale distance without setting total-mass")

        return self.rescale_wave(self.totalmass,
                                 inclination=self.inclination,
                                 phi=self.phi,
                                 distance=distance)

    #

    def rotate(self, inclination=0, phi=0):
        """ Rotates waveforms to different inclination and initial-phase angles,
        with the total-mass and distance set to internal values, provided earlier,
        e.g. during object initialization.
        """
        if not hasattr(self, 'totalmass') or self.totalmass is None:
            raise RuntimeError("Cannot rotate without setting total mass")
        elif not hasattr(self, 'distance') or self.distance is None:
            raise RuntimeError(
                "Cannot rescale total-mass without setting distance")

        return self.rescale_wave(self.totalmass,
                                 inclination=inclination,
                                 phi=phi,
                                 distance=self.distance)

    #

    def get_polarizations(self,
                          M=None,
                          inclination=None,
                          phi=None,
                          distance=None):
        if M is None:
            M = self.totalmass
        if inclination is None:
            inclination = self.inclination
        if phi is None:
            phi = self.phi
        if distance is None:
            distance = self.distance
        return self.rescale_wave(M,
                                 inclination=inclination,
                                 phi=phi,
                                 distance=distance)

    #
    ####################################################################
    ####################################################################
    # Functions to operate on individual modes
    ####################################################################
    ####################################################################
    #
    # Read mode amplitude as a function of t (in s or M)
    #

    def get_mode_amplitude(self,
                           totalmass=None,
                           modeL=2,
                           modeM=2,
                           dimensionless=False):
        """ compute the amplitude of a given mode. If dimensionless amplitude as a
        function of dimensionless time is not needed, make sure totalmass is set
        either in this function, or in the object earlier.
        """
        if dimensionless:
            hre = self.rawmodes_real[modeL][modeM]
            him = self.rawmodes_imag[modeL][modeM]
        else:
            # If a physical mass has been provided, returned rescaled amplitude
            if totalmass is not None:
                hre, him = self.rescale_mode(totalmass,
                                             modeL=modeL,
                                             modeM=modeM)
            elif self.totalmass != None:
                hre = self.rescaledmodes_real[modeL][modeM]
                him = self.rescaledmodes_imag[modeL][modeM]
            elif self.totalmass is None:
                raise IOError("Please provide total-mass to rescale modes to")
        return amplitude_from_polarizations(hre, him)

    #
    # Returns frequency (in Hz or 1/M) as a function of t (in s or M)
    #

    def get_mode_frequency(self,
                           totalmass=None,
                           modeL=2,
                           modeM=2,
                           dimensionless=False,
                           startIdx=0,
                           stopIdx=None):
        if dimensionless:
            hre = self.rawmodes_real[modeL][modeM]
            him = self.rawmodes_imag[modeL][modeM]
        else:
            # If a physical mass has been provided, return rescaled amplitude
            if totalmass is not None:
                hre, him = self.rescale_mode(totalmass,
                                             modeL=modeL,
                                             modeM=modeM)
            elif self.totalmass != None:
                hre = self.rescaledmodes_real[modeL][modeM]
                him = self.rescaledmodes_imag[modeL][modeM]
            elif self.totalmass is None:
                raise IOError("Please provide total-mass to rescale modes to")

        hre = hre[startIdx:stopIdx]
        him = him[startIdx:stopIdx]

        return frequency_from_polarizations(hre, -1 * him)

    #
    # Returns MODE PHASE (in radians) as a function of t (in s or M)
    #

    def get_mode_phase(self,
                       totalmass=None,
                       modeL=2,
                       modeM=2,
                       dimensionless=False,
                       startIdx=0,
                       stopIdx=None):
        if dimensionless:
            hre = self.rawmodes_real[modeL][modeM]
            him = self.rawmodes_imag[modeL][modeM]
        else:
            # If a physical mass has been provided, return rescaled amplitude
            if totalmass is not None:
                hre, him = self.rescale_mode(totalmass,
                                             modeL=modeL,
                                             modeM=modeM)
            elif self.totalmass != None:
                hre = self.rescaledmodes_real[modeL][modeM]
                him = self.rescaledmodes_imag[modeL][modeM]
            elif self.totalmass is None:
                raise IOError("Please provide total-mass to rescale modes to")

        hre = hre[startIdx:stopIdx]
        him = him[startIdx:stopIdx]

        return phase_from_polarizations(hre, him)

    #
    ####################################################################
    ####################################################################
    # Functions to operate on polarizations
    ####################################################################
    ####################################################################
    #
    ####################################################################
    # Functions related to wave-frequency
    ####################################################################
    #
    # Get 2,2-mode GW_frequency in Hz at a given time (M)
    #

    def get_frequency_t(self, t, totalmass=None):
        """ Get 2,2-mode GW_frequency in Hz at a given time (M) """
        if totalmass is None and self.totalmass > 0:
            totalmass = self.totalmass
        elif totalmass is None:
            raise IOError("Please set total-mass")
        self.rescale_to_totalmass(totalmass)
        #
        index = int(np.round(t * totalmass * lal.MTSUN_SI * self.sample_rate))
        #
        NEWindex = 50
        mf = self.get_mode_frequency(dimensionless=False,
                                     startIdx=index - NEWindex,
                                     stopIdx=index + NEWindex)
        freq = (mf.data[NEWindex] + mf.data[NEWindex - 1]) / 2.

        if self.verbose:
            print("> get_orbital_frequency:: index = %d, freq = %f" %
                  (index, freq),
                  file=sys.stderr)
        return totalmass, freq

    #

    def get_lowest_binary_mass(self, t, f_lower):
        """This function gives the total mass corresponding to a given starting time
        in the NR waveform, and a desired physical lower frequency cutoff.
        t = units of Total Mass
        """
        rescaled_mass, orbit_freq1 = self.get_frequency_t(t)
        m_lower = orbit_freq1 * rescaled_mass / f_lower
        return m_lower

    #
    ###################################################################
    # Functions related to wave-amplitude
    ###################################################################
    #
    #
    # Get the 2,2-mode GW amplitude at the peak of |h22|
    #

    def get_peak_amplitude(self, amp=None, totalmass=None):
        """ Get the 2,2-mode GW amplitude at the peak of |h22|. """
        if totalmass is None:
            if self.totalmass is not None:
                totalmass = self.totalmass
            else:
                raise IOError("Need to set the total-mass first")

        if amp is None:
            amp = self.get_mode_amplitude(totalmass=totalmass,
                                          dimensionless=False)

        iStart = int(len(amp.data) * 2. / 4.)
        aMax, iMax = amp[iStart:].abs_max_loc()
        if iMax == 0 or iMax == len(amp[iStart:]) - 1:
            iStart = 0
            aMax, iMax = amp[iStart:].abs_max_loc()
        else:
            iMax += iStart

        if self.verbose:
            print(" Amplitude MAX found at %d, value = %e" % (iMax, aMax))
        return [iMax, aMax]

    #
    ###################################################################
    ####################################################################
    # Strain conditioning
    ####################################################################
    ###################################################################
    #

    def taper_filter_waveform(self,
                              hpsamp=None,
                              hcsamp=None,
                              tapermethod='planck',
                              ttaper1=100,
                              ttaper2=1000,
                              ftaper3=0.1,
                              ttaper4=100.,
                              npad=00,
                              f_filter=-1.,
                              verbose=False):
        """Tapers using a Plank-taper window and high-pass filters.
        **IMPORTANT** The domain of the window is passed as ttaper{1,2,3,4} values.
        ttaper1 : time (in total-mass units) from the start where the window starts
        ttaper2 : width of start window
        ftaper3 : fraction by which amplitude should fall after its peak,
                    marking where the rolldown window starts
        ttaper4 : width of the rolldown window

        Currently supported tapermethods: 'planck' [default], 'cosine'
        """
        # {{{
        if hpsamp and hcsamp:
            hp0, hc0 = [hpsamp, hcsamp]
        elif self.rescaled_hp is not None and self.rescaled_hc is not None:
            totalmass = self.totalmass
            hp0, hc0 = [self.rescaled_hp, self.rescaled_hc]
        else:
            raise IOError("Please provide either the total mass (and strain)")
        # Check windowing extents
        if ttaper1 > ttaper2 or \
                (ttaper1+ttaper2+ttaper4) > (len(hp0)*hp0.delta_t/self.totalmass/lal.MTSUN_SI):
            raise IOError("Invalid window configuration with [%f,%f,%f,%f]" %
                          (ttaper1, ttaper2, ftaper3, ttaper4))
        #
        hp = TimeSeries(hp0.data,
                        dtype=hp0.dtype,
                        delta_t=hp0.delta_t,
                        epoch=hp0._epoch)
        hc = TimeSeries(hc0.data,
                        dtype=hc0.dtype,
                        delta_t=hc0.delta_t,
                        epoch=hc0._epoch)
        # Get actual waveform length
        for idx in np.arange(len(hp) - 1, 0, -1):
            if hp[idx] == 0 and hc[idx] == 0:
                break
        N = idx  # np.where( hp.data == 0 )[0][0]
        # Check npad
        if abs(len(hp) - N) < npad:
            print("Ignoring npad..\n", file=sys.stdout)
            npad = 0
        else:
            # Prepend some zeros to the waveform (assuming there are ''npad'' zeros at the end)
            hp = zero_pad_beginning(hp, steps=npad)
            hc = zero_pad_beginning(hc, steps=npad)
        #
        # ##########   Construct the taper window    #############
        #
        # Get the amplitude peak
        amp = amplitude_from_polarizations(hp, hc)
        max_a, nPeak = amp.abs_max_loc()

        # First get the starting-half indices
        ttapers = np.array([ttaper1, ttaper2], dtype=np.float128)
        ntapers = np.int64(
            np.round(ttapers * self.totalmass * lal.MTSUN_SI *
                     self.sample_rate))
        ntaper1, ntaper2 = ntapers
        ntaper2 += ntaper1

        if ntaper2 < ntaper1:
            raise RuntimeError("Could not configure starting taper-window")

        # Next get the next-half indices
        amp_after_peak = amp[nPeak:]
        iA, vA = min(enumerate(amp_after_peak),
                     key=lambda x: abs(x[1] - ftaper3 * max_a))
        ntaper3 = nPeak + iA

        #iB, vB = min(enumerate(amp_after_peak),key=lambda x:abs(x[1] - 0.01*max_a))
        #iB = iA + ntaper4
        ntaper4 = np.int64(
            np.round(ttaper4 * self.totalmass * lal.MTSUN_SI *
                     self.sample_rate))
        ntaper4 += ntaper3

        if ntaper3 <= nPeak or ntaper4 < ntaper3:
            tmp_data = amp_after_peak.data
            for idx in range(len(amp_after_peak)):
                if tmp_data[idx] < ftaper3 * max_a:
                    break
            ntaper3 = nPeak + idx
            for idx in range(len(amp_after_peak)):
                if tmp_data[idx] < ftaper4 * max_a:
                    break
            ntaper4 = nPeak + idx

        if ntaper3 <= nPeak or ntaper4 < ntaper3:
            raise RuntimeError("Could not configure ringdown tapering window")

        ntapers = np.array([ntaper1, ntaper2, ntaper3, ntaper4],
                           dtype=np.int64)
        ttapers = np.float128(ntapers) / self.sample_rate
        time_array = hp.sample_times.data - np.float(hp._epoch)
        #
        # Actual windowing function
        #
        region1 = np.zeros(ntaper1)
        region2 = np.zeros(ntaper2 - ntaper1)
        region3 = np.ones(ntaper3 - ntaper2)
        region4 = np.zeros(ntaper4 - ntaper3)
        region5 = np.zeros(len(hp) - ntaper4)
        #
        if 'planck' in tapermethod:
            np.seterr(divide='raise',
                      over='raise',
                      under='ignore',
                      invalid='raise')
            t1, t2, t3, t4 = ttapers
            i1, i2, i3, i4 = ntapers
            if verbose:
                print("window times = ", t1, t2, t3, t4, " idxs = ", i1, i2,
                      i3, i4)
            #
            for i in range(len(region2)):
                if i == 0:
                    region2[i] = 0
                    continue
                try:
                    region2[i] = 1. / (np.exp(((t2 - t1) /
                                               (time_array[i + i1] - t1)) +
                                              ((t2 - t1) /
                                               (time_array[i + i1] - t2))) + 1)
                except:
                    if time_array[i + i1] > 0.9 * t1 and time_array[
                            i + i1] < 1.1 * t1:
                        region2[i] = 0
                    if time_array[i + i1] > 0.9 * t2 and time_array[
                            i + i1] < 1.1 * t2:
                        region2[i] = 1.
            #
            for i in range(len(region4)):
                if i == 0:
                    region4[i] = 1.
                    continue
                try:
                    region4[i] = 1. / (np.exp(((t3 - t4) /
                                               (time_array[i + i3] - t3)) +
                                              ((t3 - t4) /
                                               (time_array[i + i3] - t4))) + 1)
                except:
                    if time_array[i + i3] > 0.9 * t3 and time_array[
                            i + i3] < 1.1 * t3:
                        region4[i] = 1.
                    if time_array[i + i3] > 0.9 * t4 and time_array[
                            i + i3] < 1.1 * t4:
                        region4[i] = 0
            #
            if verbose and False:
                import matplotlib.pyplot as plt
                plt.plot(np.arange(len(region1)) * hp.delta_t, region1)
                offset = len(region1)
                plt.plot((offset + np.arange(len(region2))) * hp.delta_t,
                         region2)
                offset += len(region2)
                plt.plot((offset + np.arange(len(region3))) * hp.delta_t,
                         region3)
                offset += len(region3)
                plt.plot((offset + np.arange(len(region4))) * hp.delta_t,
                         region4)
                plt.grid()
                plt.show()

            win = np.concatenate((region1, region2, region3, region4, region5))
        elif 'cos' in tapermethod:
            win = region1
            win12 = 0.5 + 0.5 * np.array([
                np.cos(np.pi *
                       (float(j - ntaper1) / float(ntaper2 - ntaper1) - 1))
                for j in np.arange(ntaper1, ntaper2)
            ])
            win = np.append(win, win12)
            win = np.append(win, region3)
            win34 = 0.5 - 0.5 * np.array([
                np.cos(np.pi *
                       (float(j - ntaper3) / float(ntaper4 - ntaper3) - 1))
                for j in np.arange(ntaper3, ntaper4)
            ])
            win = np.append(win, win34)
            win = np.append(win, region5)
        else:
            raise IOError("Please specify valid taper-method")
        #
        # ########### Taper & Filter the waveform ##############
        #
        hp.data *= win
        hc.data *= win
        #
        # High pass filter the waveform
        if f_filter > 0:
            hplal = convert_TimeSeries_to_lalREAL8TimeSeries(hp)
            hclal = convert_TimeSeries_to_lalREAL8TimeSeries(hc)
            lal.HighPassREAL8TimeSeries(hplal, f_filter, 0.9, 8)
            lal.HighPassREAL8TimeSeries(hclal, f_filter, 0.9, 8)
            hp = convert_lalREAL8TimeSeries_to_TimeSeries(hplal)
            hc = convert_lalREAL8TimeSeries_to_TimeSeries(hclal)

        return hp, hc
        # }}}

    #
    # }}}
