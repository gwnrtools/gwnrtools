#!/usr/bin/env python
#
# Copyright (C) 2018 Prayush Kumar
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
import os
import sys

from numpy import *
import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline

import lal
from pycbc.types import *
#from pycbc.types import TimeSeries
from pycbc.waveform import phase_from_polarizations

######################################################################
__author__ = "Prayush Kumar <prayush@astro.cornell.edu>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])
verbose = False


# @nr_mode pyexample
#  Documentation for this module.
#
# Alias "nr_wave" to "nr_strain". This way we have a complete set of
# - "nr_data" as raw data containers for NR data
# - "nr_mode" as manipulation class for single NR modes
# - "nr_strain" / "nr_wave" as manipulation class for GW strain
# Each class depends on the all previous ones.
#
######################################################################
######################################################################
#
#   Make a class for basic manipulations of NR multipoles
#
######################################################################
######################################################################
class nr_mode():
    # {{{
    def __init__(self, mode_data, delta_t=1.0, verbose=0):
        """
Container for a single NR mode.

Takes in one mode and its time samples,
as an N x 3 numpy.FLOAT64 array, or an N x 2 numpy.COMPLEX128 array

Takes in time-sampling rate, **in units of total mass M**

Returns mode amplitude as a numpy.FLOAT64 array
Returns mode phase     as a numpy.FLOAT64 array
Returns mode frequency as a numpy.FLOAT64 array

[OPTIONAL]

        """
        self.verbose = verbose
        self.delta_t = delta_t
        self.totalmass = None
        self.distance = None

        # Check input for shape
        r, c = np.shape(mode_data)
        if c > r:
            if verbose > 1:
                print(
                    "Transposing input mode array, r={}, c={} before.".format(
                        r, c))
            mode_data = np.transpose(mode_data)

        # Store time & mode samples
        t_samples = mode_data[:, 0]
        if np.shape(mode_data)[-1] == 2:
            mode_samples = mode_data[:, 1]
        elif np.shape(mode_data)[-1] == 3:
            mode_samples = mode_data[:, 1] + mode_data[:, 2] * 1.0j
        else:
            raise IOError("Check input data")
        self.t_samples = t_samples
        self.mode_samples = mode_samples

        # Create interpolation splines for the mode
        re_int = InterpolatedUnivariateSpline(t_samples, np.real(mode_samples))
        im_int = InterpolatedUnivariateSpline(t_samples, np.imag(mode_samples))
        self.mode_real_interp = re_int
        self.mode_imag_interp = im_int

        # Create a resampled COMPLEX mode array TimeSeries
        self.dimLess = True
        _ = self.resample(delta_t)
        return

    ##

    def resample(self, delta_t):
        """
Resample all data to a new sample rate.

Takes in the new sampling time step, in units of total mass M
        """
        if delta_t != self.delta_t or not hasattr(self, "mode_array"):
            if verbose > 0:
                print("Resampling mode data to sample rate: {} (1/M)".format(
                    1. / delta_t))
            self.delta_t = delta_t
            t_array = np.arange(np.min(self.t_samples), np.max(self.t_samples),
                                delta_t)
            mode_array = self.mode_real_interp(
                t_array) + self.mode_imag_interp(t_array) * 1.0j
            self.mode_array = TimeSeries(mode_array,
                                         delta_t=delta_t,
                                         copy=True)
            find_max_start = len(self.mode_array) * 4 / 5
            max_idx = find_max_start + \
                self.mode_array[find_max_start:].abs_max_loc()[-1]
            if self.verbose > 1:
                print("\t\tMax of mode found at index: {}".format(max_idx))
            # Set epoch of mode to place amplitude peak at t=0
            self.mode_array = TimeSeries(
                self.mode_array,
                epoch=lal.LIGOTimeGPS(-1. *
                                      self.mode_array.sample_times[max_idx]),
                copy=True)
        self.dimLess = True
        self.totalmass = None
        self.distance = None
        return self

    ##

    def resample_to_Hz(self, delta_t, total_mass, distance=1.0e6):
        """
Resample all data to a new sample rate (in Hz).

Takes in the new sampling time step, in Hz
Takes in the total mass M (in units of solar masses)

[NOTE] This function changes units for all subsequent computation
        """
        delta_t_dimless = delta_t / lal.MTSUN_SI / total_mass
        if total_mass != self.totalmass or delta_t != self.delta_t or distance != self.distance:
            self.resample(delta_t_dimless)
            ampl_scaling = total_mass * lal.MRSUN_SI / (distance * lal.PC_SI)
            time_scaling = total_mass * lal.MTSUN_SI
            self.mode_array = TimeSeries(self.mode_array * ampl_scaling,
                                         delta_t=delta_t,
                                         epoch=lal.LIGOTimeGPS(
                                             float(self.mode_array._epoch) *
                                             time_scaling),
                                         copy=True)
            self.totalmass = total_mass
            self.distance = distance
            self.delta_t = delta_t
        else:
            if self.verbose > 1:
                print(
                    "\tNo need to resample, as delta_t, total_mass and distance are same as before."
                )
        self.dimLess = False
        if self.verbose > 2:
            print(
                "WARNING: After resampling to Hz, make sure to resample before using dimensionless units"
            )
        return self

    ##
    def data(self):
        return self.mode_array

    ##

    def amplitude(self, startIdx=0, stopIdx=-1):
        """
Return the amplitude TimeSeries of the mode
        """
        return TimeSeries(np.abs(self.mode_array[startIdx:stopIdx]),
                          delta_t=self.mode_array.delta_t,
                          epoch=self.mode_array._epoch,
                          copy=True)

    ##

    def phase(self, startIdx=0, stopIdx=-1):
        """
Return the phase TimeSeries of the mode
        """
        re_array = self.mode_array.real()[startIdx:stopIdx]
        im_array = self.mode_array.imag()[startIdx:stopIdx]
        ph_array = phase_from_polarizations(re_array, -1 * im_array)
        return TimeSeries(
            ph_array,
            delta_t=self.mode_array.delta_t,
            epoch=self.mode_array.
            _epoch,  # FIXME: BUG HERE IF USING PART OF ARRAY?
            copy=True)

    ##

    def angular_velocity(self, startIdx=0, stopIdx=-1):
        """
Return the angular velocity TimeSeries of the mode, in units (radians per time Unit)
        """
        return (2. * np.pi) * self.frequency(startIdx=startIdx,
                                             stopIdx=stopIdx)

    ##

    def frequency(self, startIdx=0, stopIdx=-1):
        """
Return the frequency TimeSeries of the mode, in units (cycles per time Unit)
        """
        _phase = self.phase(startIdx=startIdx, stopIdx=stopIdx)
        phase_deriv_interp = InterpolatedUnivariateSpline(
            _phase.sample_times, _phase.data).derivative(n=1)
        _frequency = phase_deriv_interp(_phase.sample_times) / 2. / np.pi
        return TimeSeries(_frequency,
                          delta_t=_phase.delta_t,
                          epoch=_phase._epoch,
                          copy=True)

    ##

    def data_duration_in_time(self):
        return [self.t_samples[0], self.t_samples[-1]]

    def data_end_time(self):
        return self.data_duration_in_time[0]

    def data_start_time(self):
        return self.data_duration_in_time[-1]

    def data_duration(self):
        x, y = self.data_duration_in_time()
        return y - x

    # }}}
