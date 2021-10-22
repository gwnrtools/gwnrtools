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
from __future__ import print_function

import os
import sys

from numpy import *
import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import cumtrapz

import lal
from pycbc.waveform import *
from pycbc.types import *
from pycbc.filter import *

from gwnr.nr.types import nr_data
from gwnr.utils import zero_pad_beginning
from gwnr.waveform.utils import get_time_at_frequency
######################################################################
__author__ = "Prayush Kumar <prayush@astro.cornell.edu>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])
verbose = False


## @nr_wave pyexample
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
#     Make a class for basic manipulations of SpEC NR waveforms      #
#
######################################################################
######################################################################
class nr_strain():
    #{{{
    def __init__(self,
                 filename,\
                 filetype='HDF5',\
                 wavetype='Auto', \
                 wave_uniformly_sampled=False,\
                 ex_order = 3,\
                 group_name = None,\
                 modeLmin = 2,\
                 modeLmax = 4,\
                 skipM0 = True,\
                 which_modes = [],\
                 dimless_sample_rate=1.0,\
                 sample_rate=4096,\
                 time_length=16,\
                 totalmass=None,\
                 inclination=0.0,\
                 phi=0.0,\
                 distance=1.0e6,\
                 verbose=0):
        """
##################################################
######        ASSUMPTIONS
##################################################
1: All modes share a common time stencil

##################################################
######        CONVENTIONS
##################################################
1: Modes themselves are not amplitude-scaled. Only their time-axis is
    rescaled.
2: Following <https://arxiv.org/pdf/0709.0093.pdf> we will follow the convention where:

 $$h_+ - i h_\times = \sum_{l,m} Y^{l,m}_{-2}(\iota, \phi_c) h_{l,m}(M_i, S_i, \cdots)$$

 and each *(l,m)* multipole is expanded in amplitude and phase as:

 $$h_{l,m} := A_{l,m} \mathrm{e}^{i\phi_{l,m}}$$

 where $$\phi_{l,m}\propto -m\times\Phi (=\phi_\mathrm{orbital})$$,

 i.e. $$h_{l,m} := A_{l,m} \mathrm{e}^{-i m\Phi}$$, and

 $$\mathcal{Re}[h_{l,m}] := +A_{l,m} \cos(m\Phi)$$
 $$\mathcal{Im}[h_{l,m}] := -A_{l,m} \sin(m\Phi)$$


##################################################
###           INPUTS
##################################################
1: filename = FULL PATH to NR data file

2: filetypes passed should be : 'HDF5' , 'ASCII' or 'DataSet'
  2.1: For HDF5 files:-
  Between "wavetype", "ex_order", and "group_name", provide:
           a) for CCE waveforms, wavetype=CCE. It uses highest-R data.
           b) for extrapolated waveforms, wavetype='Extrapolated' and ex_order=?
           c) for finite-radii waveforms, wavetype='FiniteRadius'
           d) for datasets without groups, wavetype='NoGroup'
           e) (experimental) Determine automatically: wavetype='Auto' (default)
           f) for non-SXS waveforms, group_name='...'
    Note: group_name overwrites other two options. So one can also provide
                group_name = 'Extrapolated_N3.dir' OR
                group_name = 'CceR0350.dir', etc.
  2.2: For ASCII files:-
   () Provide wavetype = 'regex'. This assumes:
           a) filename is a REGEX expression that can be formatted with (modeL, modeM) integer tuples
  2.3: For DataSet:-
   () Provide wavetype = 'dict'. This assumes:
           a) This dictionary should have [l][m] modes as Nx2 or Nx3 matrices

6: modeLmin, modeLmax: Range of l-modes of strain to use
            (cannot use arbitrary ones yet)
7: skipM0: Skip m=0 (DC) modes (Default: True)
8: sample_rate, time_length: length params for data structures
9: dimless_sample_rate: length params in dimensionless units
10: totalmass: total mass to rescale NR waveform to (Solar Masses)
11: inclination, phi: Inclination and initial phase angles (rad)
12: distance: distance to source (Pc, Default=1e6 or 1Mpc)

##################################################
###           OBJECT STATE : CLASS S1
##################################################
A lot of functionality of this class depends on the internal state of the
object that sets how the NR modes are to be scaled. This specification of
state comes under class S1. Two variables set this state of the object:

I)
M - Total mass of the binary [in Solar Mass] at the start. This can be
    set at instantiation through the input option:
        totalmass=None,
    or as input to various member functions.

II)
Sample Rates - They are set by input options:
    dimless_sample_rate=1.0,
    sample_rate=4096,
that take inputs in units [Solar Mass, Hz]. Once specified, they cannot
be changed [Note that the base data-container class supports resampling,
just not this wrapper around it].

======= STATES =======
Both states are defined by the value of member var "totalmass". If it is
None then the object is in the "dimensionless" state. In this state:
(a) No amplitude scaling is applied to the raw modes (i.e. no effect of
    totalmass or source distance).
(b) No scaling of the time scale. Dimensionless time will be used in the unit
    of Solar Mass. Sampling rate will be set by "dimless_sample_rate".
(c) Frequency is generally expected in the dimensionless units of (1/M), while
    time in the dimensionless units of (M).
If "totalmass" is a floating point number the object's S1 state is 
"dimensionfull". In this state:
(a) Amplitude is scaled by (M/R), where the ratio is dimensionless.
(b) Time scale is scaled by (1/M), where M is in units of seconds.
(c) Frequencies in general are expected in Hz and time in seconds
  
======= SWITCHING BETWEEN STATES =======
There are two methods designed to switch to each of the two states. They are:

(a) make_modes_dimensionless(self, dimless_delta_t=-1),
(b) rescale_modes(self, delta_t=None, M=None, distance=None).

Function (a) switches the object to "dimensionless" state. It sets "totalmass"
to None, etc. Here one can change the "dimless_sample_rate" provided at object
creation. 

Function (b) switches the object to "dimensionfull" state. To do so,
it needs variables (delta_t, M, distance) that define the state. 

======= AUTO-SWITCHING BETWEEN STATES =======
In addition, there are many other functions that switch between states. The 
general philosophy is that member functions should now allow switching to 
the "dimensionfull" state, that has to be done explicitly. They can allow
switching to "dimensionless" state though.

These   **If dimensionless=True, object's S1 state is switched to "dimensionless"**
        **If dimensionless=False, object's state is NOT switched explicitly.     **
        **   You get results in whatever the object's current state is           **
(a) get_mode_amplitude(**dimensionless=False**)
(b) get_mode_frequency(**dimensionless=False**)
(c) get_mode_phase(**dimensionless=False**)

These are designed to switch to desired state:
(d) make_modes_dimensionless(self, dimless_delta_t=-1),
(e) rescale_modes(self, delta_t=None, M=None, distance=None).

These switch to and only work in ONE state:
(f) get_polarizations()     [switches to "dimensionfull" state]
(g) rescale_to_totalmass()  [switches to "dimensionfull" state]
(h) rescale_to_distance()   [switches to "dimensionfull" state]
(i) rotate(self)            [switches to "dimensionfull" state]
(j) taper_filter_waveform() [works only in 'dimensionfull' state]
(k) amplitude(self)         [works only in 'dimensionfull' state]
(l) phase(self)             [works only in 'dimensionfull' state]
(m) frequency(self)         [works only in 'dimensionfull' state]

The following do NOT change the state:
(n) get_t_frequency(self, f, totalmass=None, dimless=False)
(o) get_frequency_t(self, t, totalmass=None, dimless=False)
(p) get_amplitude_peak_h22(self, amp=None)

The following do NOT change OR care about the state:
(q) orbital_frequency(self)
(r) get_strain_modes_amplitudes()
(s) get_bondi_news_modes()
(t) get_psi4_modes()
(u) dEdt()
(v) dEdtfunc()
(w) E()
(x) J()
        """
        self.verbose = verbose
        ##################################################################
        #   0. Ensure inputs are correct
        ##################################################################
        if 'DataSet' not in filetype:
            if filename is not None and not os.path.exists(filename):
                raise IOError("Please provide data file!")
            if self.verbose > 0:
                print("Init nr_wave: Reading From Filename=%s" % filename)
        ##################################################################
        # 1. Store various things
        ##################################################################
        # Datafile details
        self.modeLmax = modeLmax
        self.modeLmin = modeLmin
        self.skipM0 = skipM0
        self.which_modes = which_modes

        # Binary parameters
        self.totalmass = totalmass
        if self.totalmass:
            self.totalmass_secs = self.totalmass * lal.MTSUN_SI
        self.inclination = inclination
        self.phi = phi
        self.distance = distance
        if self.verbose > 2:
            print("\t\tInput mass, inc, phi, dist = ", totalmass,\
                                self.inclination, self.phi, self.distance)

        # Data analysis parameters
        self.sample_rate = sample_rate
        self.time_length = time_length
        self.delta_t = 1. / self.sample_rate
        self.dimless_delta_t = 1.0 / dimless_sample_rate
        self.df = 1. / self.time_length
        self.n = int(self.sample_rate * self.time_length)
        if self.verbose > 1:
            print("self.sample-rate & time_len = ", self.sample_rate,
                  self.time_length)
            print("self.n = ", self.n)

        ##################################################################
        #   2. Read the data from the file. Read all modes.
        ##################################################################
        if self.verbose > 1:
            print("Init nr_wave: Reading data....")
        self.data = nr_data(filename,
                            filetype=filetype,
                            wavetype=wavetype,
                            ex_order=ex_order,
                            group_name=group_name,
                            modeLmin=self.modeLmin,
                            modeLmax=self.modeLmax,
                            skipM0=self.skipM0,
                            delta_t=1.0 / dimless_sample_rate,
                            verbose=self.verbose)
        self.which_modes_to_read()
        self.rescaled_hp, self.rescaled_hc = None, None
        if self.verbose > 1:
            print("Init nr_wave: Data reading successful.")
            if self.verbose > 2:
                print("\t\t Read in modes: ", self.which_modes)
        ##################################################################
        #   3. Preprocessing
        ##################################################################
        ## self.rescale_modes()
        if self.totalmass > 1.0: self.rescale_to_totalmass(self.totalmass)
        if self.verbose > 1:
            print("Init nr_wave: Successful.")
        return

    ##
    def which_modes_to_read(self):
        """[This function is agnostic to object's S1 state]"""
        if len(self.which_modes) == 0:
            for modeL in self.data.modes:
                for modeM in self.data.modes[modeL]:
                    self.which_modes.append((modeL, modeM))
        return self.which_modes

    ##
    ####################################################################
    ####################################################################
    ##### Functions to operate on individual modes
    ####################################################################
    ####################################################################
    #
    # Read mode amplitude as a function of t (in s or M)
    def get_mode_amplitude(self,
                           modeL=2,
                           modeM=2,
                           startIdx=0,
                           stopIdx=-1,
                           dimensionless=False):
        """
        **If dimensionless=True, object's S1 state is switched to "dimensionless"**
        **If dimensionless=False, object's state is NOT switched explicitly.     **
        **   You get results in whatever the object's current state is           **
        
        Compute the amplitude of a given mode. If dimensionless amplitude as a
        function of dimensionless time is not needed, make sure totalmass is set
        either in this function, or in the object earlier.
        """
        if dimensionless: self.make_modes_dimensionless()
        return self.data.modes[modeL][modeM].amplitude(startIdx=startIdx,
                                                       stopIdx=stopIdx)

    #
    # Returns frequency (in Hz or 1/M) as a function of t (in s or M)
    def get_mode_frequency(self,
                           modeL=2,
                           modeM=2,
                           startIdx=0,
                           stopIdx=-1,
                           dimensionless=False):
        """
        **If dimensionless=True, object's S1 state is switched to "dimensionless"**
        **If dimensionless=False, object's state is NOT switched explicitly.     **
        **   You get results in whatever the object's current state is           **
        """
        if dimensionless: self.make_modes_dimensionless()
        return self.data.modes[modeL][modeM].frequency(startIdx=startIdx,
                                                       stopIdx=stopIdx)

    #
    # Returns MODE PHASE (in radians) as a function of t (in s or M)
    def get_mode_phase(self,
                       modeL=2,
                       modeM=2,
                       startIdx=0,
                       stopIdx=-1,
                       dimensionless=False):
        """
        **If dimensionless=True, object's S1 state is switched to "dimensionless"**
        **If dimensionless=False, object's state is NOT switched explicitly.     **
        **   You get results in whatever the object's current state is           **
        """
        if dimensionless: self.make_modes_dimensionless()
        return self.data.modes[modeL][modeM].phase(startIdx=startIdx,
                                                   stopIdx=stopIdx)

    #
    ####################################################################
    # Functions related to wave-frequency
    ####################################################################
    #
    def get_t_frequency(self, f, totalmass=None, dimless=False):
        """
** Object's S1 state is NOT CHANGED. "dimless" flag sets the units of output**
** Will not work for f < 1Hz **
        
Provide f (frequency) in Hz. Or provide f (dimension-less) if dimless=True.
Returns t (time) in seconds. Or returns t (dimension-less) if dimless=True.
        """
        ## Figure out if provided frequency is dimensionless or not
        ## Assume that we don't care about <1Hz.
        if f < 1.0:
            f_is_dimless = True
        else:
            f_is_dimless = False
        ##
        if totalmass is None:
            totalmass = self.totalmass
        #else:
        #    self.totalmass = totalmass
        #    self.rescale_modes(M=totalmass)

        ##
        ## Get frequency timeSeries for the (2,2) mode
        freq = self.get_mode_frequency(modeL=2, modeM=2)

        ## Find time and use correct units
        if self.data.modes[2][2].dimLess:
            if not f_is_dimless and totalmass is None:
                raise IOError(
                    "Since you haven't rescaled this wave yet, provide dimensionless frequency instead of {}"
                    .format(f))
            elif not f_is_dimless:
                f *= (totalmass * lal.MTSUN_SI)  # Make frequency dimensionless
            t = get_time_at_frequency(freq,
                                      f)  # This will give dimensionless time
            if not dimless and totalmass != None:
                t *= (totalmass * lal.MTSUN_SI)
        else:
            if f_is_dimless:
                f /= (totalmass * lal.MTSUN_SI)
            t = get_time_at_frequency(freq, f)  # This will be time in seconds
            if dimless:
                t /= (totalmass * lal.MTSUN_SI)
        return t

    #
    # Get 2,2-mode GW_frequency in Hz at a given time (M)
    #
    def get_frequency_t(self, t, totalmass=None, dimless=False):
        """
Get 2,2-mode GW_frequency in Hz at a given time (in M)

** Object's S1 state is NOT CHANGED. "dimless" flag sets the units of output    **
** If Object is "dimensionless", provide t in units of (M). If not, then in (s).**

** Must provide "totalmass" if trying to get output in "dimensionfull" units from **
** an object in "dimensionless" state, or vice-versa.                             **

** Will not work for f < 1Hz **        
        """
        if totalmass is None:
            totalmass = self.totalmass
        #else:
        #    self.totalmass = totalmass
        #    self.rescale_to_totalmass(totalmass)

        freq = self.get_mode_frequency(modeL=2, modeM=2)
        freqI = InterpolatedUnivariateSpline(freq.sample_times, freq)

        f_is_dimless = self.data.modes[2][2].dimLess

        # If requested operation uses dimensionless Units, we
        # assume the t given is in units of M, and
        if f_is_dimless:
            # Convert t if its in seconds
            if totalmass is not None:
                t /= (totalmass * lal.MTSUN_SI)
            else:
                #raise IOError("Please provide totalmass to convert time to dimenLess Units")
                pass
        else:
            # Assume now the time given t is in seconds
            pass

        try:
            fvalue = freqI(t)
        except:
            raise IOError("""
            Time provided = {} and times of frequencyTimeSEries: {},{}
            Request to use Dimless Units: {}.
            Are these consistent?
            """.format(t, freq.sample_times[0], freq.sample_times[-1],
                       dimless))

        if self.verbose > 1:
            print("\tget_frequency_t: t = {}, freq = {}".format(t, fvalue))

        if dimless:
            if not f_is_dimless:
                if totalmass is not None:
                    fvalue *= (totalmass * lal.MTSUN_SI)
                else:
                    pass
        else:
            if f_is_dimless:
                if totalmass is not None:
                    fvalue /= (totalmass * lal.MTSUN_SI)
                else:
                    pass

        return fvalue

    ###################################################################
    # Functions related to wave-amplitude,phase,frequency
    ###################################################################
    #
    # Get the 2,2-mode GW amplitude at the peak of |h22|
    def get_amplitude_peak_h22(self, amp=None):
        """
        Get the 2,2-mode GW amplitude at the peak of |h22|.
        
        ** Object's S1 state is NOT CHANGED. **
        """
        if amp is None:
            amp = self.data.modes[2][2].amplitude()
        iMax = np.where(
            np.abs(amp.sample_times.data) == np.min(
                np.abs(amp.sample_times.data)))[0][0]
        aMax = amp[iMax]
        return [aMax, iMax]

    #
    # Get the amplitude of + x polarizations
    def amplitude(self):
        """
        ** Object's S1 state is NOT CHANGED. **
        """
        if self.rescaled_hp is None or self.rescaled_hc is None:
            raise IOError("Please call `get_polarizations` first. ")
        #return np.abs(self.rescaled_hp**2 + self.rescaled_hc**2)**0.5
        return amplitude_from_polarizations(self.rescaled_hp, self.rescaled_hc)

    #
    # Get the phase of + x polarizations
    def phase(self):
        """
        ** Object's S1 state is NOT CHANGED. **
        """
        if self.rescaled_hp is None or self.rescaled_hc is None:
            raise IOError("Please call `get_polarizations` first. ")
        return phase_from_polarizations(self.rescaled_hp, self.rescaled_hc)

    #
    # Get the frequency of + x polarizations
    def frequency(self):
        """
        ** Object's S1 state is NOT CHANGED. **
        """
        if self.rescaled_hp is None or self.rescaled_hc is None:
            raise IOError("Please call `get_polarizations` first. ")
        return frequency_from_polarizations(self.rescaled_hp, self.rescaled_hc)

    #
    def orbital_frequency(self):
        """
        ** Object's S1 state is NOT CHANGED. **
        """
        return self.data.modes[2][2].frequency() / 2.0

    #
    def get_lowest_binary_mass(self,
                               f_lower,
                               t_start,
                               totalmass=None,
                               dimless=True):
        """
Gives the Lowest possible binary total mass that the waveform can/should be
scaled to to start at **f_lower** at the time-sample at **t_start**

Choose t_start after Junk

f_lower can be in Hz of 1/M

t_start is dimensionless IFF dimless = True, else its in seconds
[MERGER is at t=0]


** Does not change the S1 state of the object. Tag "dimless=True" implies the **
**   units of t_start.                                                        **
        """
        ##{{{
        if totalmass is None:
            totalmass = self.totalmass

        UNDO_SCALING = False
        if totalmass is None:
            totalmass = 40.0
            self.rescale_to_totalmass(totalmass)
            self.totalmass = totalmass
            UNDO_SCALING = True

        if dimless:
            t_start *= (totalmass * lal.MTSUN_SI)

        orbit_freq = self.get_frequency_t(t_start,
                                          totalmass=totalmass,
                                          dimless=False)

        if f_lower < 1.0:
            f_lower /= (totalmass * lal.MTSUN_SI)

        if self.verbose > 1:
            print("\t orbit_freq found: {}, f_lower = {}".format(
                orbit_freq, f_lower))

        if UNDO_SCALING:
            self.make_modes_dimensionless()
            if self.verbose > 3:
                print("WARNING: Waveform were rescaled to M={}, Now UNSCALED.".
                      format(totalmass))
        ##
        return (orbit_freq / f_lower) * totalmass
        ##}}}

    # ##################################################################
    # Basic waveform manipulation & State Changing
    # ##################################################################
    ##
    def make_modes_dimensionless(self, dimless_delta_t=-1):
        if dimless_delta_t < 0: dimless_delta_t = self.dimless_delta_t
        _ = self.which_modes_to_read()
        for (modeL, modeM) in self.which_modes:
            self.data.modes[modeL][modeM].resample(dimless_delta_t)
        self.totalmass = None
        self.rescaled_hp = None
        self.rescaled_hc = None
        return self

    ##
    def rescale_modes(self, delta_t=None, M=None, distance=None):
        """
        This function rescales ALL modes to input mass value.
        This function is meant for usage in amplitude-scaling-invariant calculations.

        Note that this function RESETS internal total-mass values for all modes
        consistently
        """
        ##{{{
        if delta_t is None: delta_t = self.delta_t
        else: self.delta_t = delta_t

        if M is None: M = self.totalmass
        else: self.totalmass = M

        if distance is None: distance = self.distance
        else: self.distance = distance

        if self.verbose > 1:
            print("\tRescaling modes to: delta_t={}, M={}, dist={}".format(
                delta_t, M, distance))

        if delta_t is None or M is None or distance is None:
            raise IOError("One of delta_t={}, M={}, dist={} is None.\
            Please provide valid parameters to rescale modes".format(
                delta_t, M, distance))

        which_modes = self.which_modes_to_read()
        if self.verbose > 2:
            print("\t\tWill use modes: ", which_modes)

        for (modeL, modeM) in which_modes:
            self.data.modes[modeL][modeM].resample_to_Hz(delta_t,
                                                         M,
                                                         distance=distance)
        return
        ##}}}

    ##
    ####################################################################
    ####################################################################
    ##### Functions to operate on polarizations
    ####################################################################
    ####################################################################
    #
    ##
    def get_polarizations(self,
                          delta_t=None,
                          M=None,
                          distance=None,
                          inclination=None,
                          phi=None):
        """
Return plus and cross polarizations.

** Object's S1 state is CHANGED to "dimensionfull. **        
        """
        ##{{{
        #########################################################
        #### INPUT CHECKING
        #########################################################
        if delta_t is None: delta_t = self.delta_t
        else: self.delta_t = delta_t

        if M is None: M = self.totalmass
        else: self.totalmass = M

        if distance is None: distance = self.distance
        else: self.distance = distance

        if phi is None: phi = self.phi
        else: self.phi = phi

        if inclination is None: inclination = self.inclination
        else: self.inclination = inclination

        if delta_t is None or M is None or distance is None or inclination is None or phi is None:
            raise IOError(
                "One of delta_t={}, M={}, dist={}, incl={}, phi={} is None.\
            Please provide valid parameters to obtain polarizations".format(
                    delta_t, M, distance, inclination, phi))
        if self.verbose > 1:
            print("\tComputing polarizations for: delta_t={}, M={}, dist={}, incl={}, phi={}".format(\
                        delta_t, M, distance, inclination, phi))
        #########################################################
        #### RESCALE AND RESAMPLE INDIVIDUAL MODES
        #########################################################
        # First rescale all modes to required physical parameters
        self.rescale_modes(delta_t=delta_t, M=M, distance=distance)

        #########################################################
        #### ENSURE CORRECTNESS OF COALESCENCE-PHASE !!!!
        #########################################################
        # Orbital phase at the time of merger (time of amplitude peak for (2,2) mode)
        aPeak, iPeak = self.get_amplitude_peak_h22()
        if self.verbose > 3:
            print(
                "\t\t\tFound peak of amplitude of h22 at (index, ampl): {}, {}"
                .format(iPeak, aPeak))
        phase22 = self.get_mode_phase(2, 2)
        #phiOrbMerger = phase22[iPeak] / 2.
        phiOrbMerger = np.angle(self.data.modes[2][2].data()[iPeak]) / -2

        #########################################################
        #### COMBINE MODES TO GET POLARIZATIONS
        #########################################################
        # Create an empty complex array for (+ , x) polarizations
        hpols = TimeSeries(np.zeros(self.n, dtype=float) +
                           np.zeros(self.n, dtype=float) * 1.0j,
                           delta_t=delta_t,
                           epoch=phase22._epoch)
        curr_h22 = self.data.modes[2][2].data()
        # Loop over all modes to be included
        for (modeL, modeM) in self.which_modes_to_read():
            if self.skipM0 and modeM == 0: continue
            #phiOrbMerger = np.angle(self.data.modes[modeL][modeM].data()[iPeak]) * (1.0 / modeM) ## FIXME!!
            #print type(hpols), hpols.dtype
            # Compute spin -2 weighted Ylm for (inclination, PHI??)
            curr_ylm_lal = lal.SpinWeightedSphericalHarmonic(
                inclination,
                phiOrbMerger - 0 * np.pi / 4. - phi,
                #phiOrbMerger  - phi,
                -2,
                modeL,
                modeM)
            #print type(curr_ylm_lal)
            curr_ylm = np.complex128(curr_ylm_lal.real +
                                     curr_ylm_lal.imag * 1.0j)
            #print np.abs(curr_ylm)
            #print "Difference: {}",format(np.abs(curr_ylm_lal - curr_ylm))

            # h+ - \ii hx = \Sum Ylm * hlm
            tmp_hlm = self.data.modes[modeL][modeM].data()
            ## FIXME DELME
            if self.verbose > 2:
                null_ylm = np.exp(1 * (phiOrbMerger - 0 * np.pi / 4. - phi) *
                                  modeM * 1.0j)
                print("\t\t\tPhaseRemovalTerm: ", null_ylm)
                print(
                    "\t\t\tUSENR mode at peak: {}, after removing phase: {} ({})"
                    .format(tmp_hlm[iPeak], null_ylm * tmp_hlm[iPeak],
                            np.angle(null_ylm * tmp_hlm[iPeak])))

            curr_hlm = TimeSeries(tmp_hlm * curr_ylm,
                                  epoch=curr_h22._epoch,
                                  dtype=hpols.dtype,
                                  copy=True)
            if self.verbose > 1:
                print("\tShifted ({},{}) in time by {} units ({} samples)".
                      format(
                          modeL, modeM,
                          float(curr_hlm._epoch - tmp_hlm._epoch),
                          float(curr_hlm._epoch - tmp_hlm._epoch) /
                          curr_hlm.delta_t))
            hpols[:len(curr_hlm)] += curr_hlm
        # h+ - \ii hx = \Sum Ylm * hlm
        self.rescaled_hp = TimeSeries(hpols.real(),
                                      delta_t=hpols.delta_t,
                                      epoch=hpols._epoch)
        self.rescaled_hc = TimeSeries(-1 * hpols.imag(),
                                      delta_t=hpols.delta_t,
                                      epoch=hpols._epoch)
        # Return polarizations
        return [self.rescaled_hp, self.rescaled_hc]
        ##}}}

    ##
    def rescale_to_totalmass(self, M):
        """ Rescales the waveform to a different total-mass than currently. The
        values for different angles are set to internal values provided earlier, e.g.
        during object initialization.
        
        ** Object's S1 state is CHANGED to "dimensionfull. **
        """
        ##{{{
        if not hasattr(self, 'inclination') or self.inclination is None:
            raise RuntimeError(
                "Cannot rescale total-mass without setting inclination")
        elif not hasattr(self, 'phi') or self.phi is None:
            raise RuntimeError("Cannot rescale total-mass without setting phi")
        elif not hasattr(self, 'distance') or self.distance is None:
            raise RuntimeError(
                "Cannot rescale total-mass without setting distance")
        return self.get_polarizations(M=M,
                                      inclination=self.inclination,
                                      phi=self.phi,
                                      distance=self.distance)
        ##}}}

    ##
    def rescale_to_distance(self, distance):
        """ Rescales the waveform to a different distance than currently. The
        values for different angles, masses are set to internal values provided
        earlier, e.g. during object initialization.
        
        ** Object's S1 state is CHANGED to "dimensionfull. **
        """
        ##{{{
        if not hasattr(self, 'inclination') or self.inclination is None:
            raise RuntimeError(
                "Cannot rescale distance without setting inclination")
        elif not hasattr(self, 'phi') or self.phi is None:
            raise RuntimeError("Cannot rescale distance without setting phi")
        elif not hasattr(self, 'totalmass') or self.totalmass is None:
            raise RuntimeError(
                "Cannot rescale distance without setting total-mass")
        return self.get_polarizations(M=self.totalmass,
                                      inclination=self.inclination,
                                      phi=self.phi,
                                      distance=distance)
        ##}}}

    ##
    def rotate(self, inclination=None, phi=None):
        """ Rotates waveforms to different inclination and initial-phase angles,
        with the total-mass and distance set to internal values, provided earlier,
        e.g. during object initialization.
        
        ** Object's S1 state is CHANGED to "dimensionfull. **
        """
        ##{{{
        if not hasattr(self, 'totalmass') or self.totalmass is None:
            raise RuntimeError("Cannot rotate without setting total mass")
        elif not hasattr(self, 'distance') or self.distance is None:
            raise RuntimeError(
                "Cannot rescale total-mass without setting distance")
        if inclination is None: inclination = self.inclination
        else: self.inclination = inclination
        if phi is None: phi = self.phi
        else: self.phi = phi
        return self.get_polarizations(M=self.totalmass,
                                      inclination=inclination,
                                      phi=phi,
                                      distance=self.distance)
        ##}}}

    ###################################################################
    ####################################################################
    # Strain conditioning
    ####################################################################
    ###################################################################
    #
    def taper_filter_waveform(self,
                              hpsamp=None, hcsamp=None, totalmass=None,\
                              tapermethod='planck', \
                              ttaper1=100,\
                              ttaper2=1000,\
                              ftaper3=0.1,\
                              ttaper4=100.,\
                              npad=0, f_filter=-1.):
        """
Tapers using a Plank-taper window and high-pass filters.
**IMPORTANT** The domain of the window is passed as ttaper{1,2,3,4} values.

    ttaper1 : time (in total-mass units) from the start where the window starts
    ttaper2 : width of start window
    ftaper3 : fraction by which amplitude should fall after its peak,
                    marking where the rolldown window starts
    ttaper4 : width of the rolldown window

Currently supported tapermethods: 'planck' [default], 'cosine'

** Works ONLY in "dimensionfull" state **
        """
        #{{{
        ## Choose polarizations: Either User inputs here or use the objects internal polz.
        if hpsamp and hcsamp:
            hp0, hc0 = [hpsamp, hcsamp]
            if totalmass is None:
                raise IOError(
                    "If providing polarizations, also provide total mass!")
        elif self.rescaled_hp is not None and self.rescaled_hc is not None:
            totalmass = self.totalmass
            hp0, hc0 = self.rescaled_hp, self.rescaled_hc
        else:
            raise IOError(
                "Please call `get_polarizations` first. Or provide polarizations as input."
            )

        ## Check windowing extents
        if (ttaper1 + ttaper2 + ttaper4) > (len(hp0) * hp0.delta_t /
                                            self.totalmass / lal.MTSUN_SI):
            raise IOError("Invalid window configuration with [%f,%f,%f,%f] for wave of length %fM" %\
                            (ttaper1, ttaper2, ftaper3, ttaper4, (len(hp0)*hp0.delta_t/self.totalmass/lal.MTSUN_SI)))

        ## Copy over polarizations to fresh memory
        hp = TimeSeries(hp0.data,
                        dtype=hp0.dtype,
                        delta_t=hp0.delta_t,
                        epoch=hp0._epoch,
                        copy=True)
        hc = TimeSeries(hc0.data,
                        dtype=hc0.dtype,
                        delta_t=hc0.delta_t,
                        epoch=hc0._epoch,
                        copy=True)

        ## Get actual waveform length (minus padding at the end)
        N = np.minimum(
            np.where(hp.data == 0)[0][0],
            np.where(hc.data == 0)[0][0])

        ## Check if npad can be inserted at the start of the polarization TimeSeries.
        ## If there is no space ignore the pad completely
        if abs(len(hp) - N) < npad:
            print("WARNING: Cannot pad {} zeros at the end. len(hp)={} & N={}".
                  format(npad, len(hp), N))
            npad = 0
        else:
            # Prepend some zeros to the waveform (assuming there are ''npad'' zeros at the end)
            hp = zero_pad_beginning(hp, steps=npad)
            hc = zero_pad_beginning(hc, steps=npad)

        ##########################################################
        # ########   Construct the taper window configuration ####
        ##########################################################
        # Get the amplitude peak
        amp = amplitude_from_polarizations(hp, hc)
        # N + npad == total length
        # start from 0.8 * (N + npad)
        iStart = int(0.8 * (N + npad))
        max_a, nPeak = amp[iStart:].abs_max_loc()
        nPeak += iStart

        # First get the starting-half indices
        ttapers = np.array([ttaper1, ttaper2], dtype=np.float128)
        ntaper1, ntaper2 = np.int64(
            np.round(ttapers * totalmass * lal.MTSUN_SI / hp.delta_t))
        ntaper2 += ntaper1

        if ntaper2 < ntaper1:
            raise RuntimeError("Could not configure starting taper-window")

        ## Get the itime/index where the polarization amplitude = `ftaper3` x peakAmplitude
        amp_after_peak = amp[nPeak:]
        iA, vA = min(enumerate(amp_after_peak),
                     key=lambda x: abs(x[1] - ftaper3 * max_a))
        ntaper3 = nPeak + iA
        #iB, vB = min(enumerate(amp_after_peak),key=lambda x:abs(x[1] - 0.01*max_a))
        #iB = iA + ntaper4

        ## End of RD window is given by `width=ttaper4` + `start = ntaper3`
        ntaper4 = ntaper3
        ntaper4 += np.int64(\
              np.round(ttaper4 * totalmass * lal.MTSUN_SI / hp.delta_t))

        ## If the above minimization to find the RD window fails, try brute force!
        if ntaper3 <= nPeak or ntaper4 < ntaper3:
            tmp_data = amp_after_peak.data
            for idx in range(len(amp_after_peak)):
                if tmp_data[idx] < ftaper3 * max_a: break
            ntaper3 = nPeak + idx
            for idx in range(len(amp_after_peak)):
                if tmp_data[idx] < ftaper4 * max_a: break
            ntaper4 = nPeak + idx

        ## If the RD window is STILL NOT CONFIGURED, FAIL!
        if ntaper3 <= nPeak or ntaper4 < ntaper3 or ntaper3 < ntaper2 or ntaper2 < ntaper1:
            raise RuntimeError(
                "Could not configure ringdown tapering window: {},{},{},{}".
                format(ntaper1, ntaper2, ntaper3, ntaper4))

        #######################################################################
        # ################  Make final tapering window ########################
        #######################################################################
        # NOTE: ntaper1, ntaper2, ntaper3, ntaper4 are all measured    ########
        #        w.r.t. the start of hp, instead of w.r.t. each other. ########
        #######################################################################
        #
        # Combine window configuration
        ntapers = np.array([ntaper1, ntaper2, ntaper3, ntaper4],
                           dtype=np.int64)
        ttapers = np.float128(ntapers) * hp.delta_t
        time_array = hp.sample_times.data - np.float(hp._epoch)

        if self.verbose > 2:
            print("ntapers = ", ntapers)
            print("ttapers = ", ttapers)
        #
        # Windowing function time-series
        #
        region1 = np.zeros(ntaper1)
        region2 = np.zeros(ntaper2 -
                           ntaper1)  ## Modified later to chosen taperfunction
        region3 = np.ones(ntaper3 - ntaper2)
        region4 = np.zeros(ntaper4 -
                           ntaper3)  ## Modified later to chosen taperfunction
        region5 = np.zeros(len(hp) - ntaper4)
        #
        ## Modify region2 and region3 to chosen taperfunction
        if 'planck' in tapermethod:
            np.seterr(divide='raise',
                      over='raise',
                      under='ignore',
                      invalid='raise')
            t1, t2, t3, t4 = ttapers
            i1, i2, i3, i4 = ntapers
            if self.verbose > 2:
                print("\t\twindow times = ", t1, t2, t3, t4)
                print("\t\tidxs = ", i1, i2, i3, i4)
            #
            for i in range(len(region2)):
                if i == 0:
                    region2[i] = 0
                    continue
                try:
                    region2[i] = 1./(np.exp( ((t2-t1)/(time_array[i+i1]-t1)) + \
                                      ((t2-t1)/(time_array[i+i1]-t2)) ) + 1)
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
                    region4[i] = 1./(np.exp( ((t3-t4)/(time_array[i+i3]-t3)) + \
                                      ((t3-t4)/(time_array[i+i3]-t4)) ) + 1)
                except:
                    if time_array[i + i3] > 0.9 * t3 and time_array[
                            i + i3] < 1.1 * t3:
                        region4[i] = 1.
                    if time_array[i + i3] > 0.9 * t4 and time_array[
                            i + i3] < 1.1 * t4:
                        region4[i] = 0
            #
            if self.verbose > 3:
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
                plt.savefig('DEBUG-TaperingWindow.png')
            #
            win = np.concatenate((region1, region2, region3, region4, region5))
        elif 'cos' in tapermethod:
            win = region1
            win12 = 0.5 + 0.5*np.array([np.cos( np.pi*(float(j-ntaper1)/float(ntaper2-\
                                  ntaper1) - 1)) for j in np.arange(ntaper1,ntaper2)])
            win = np.append(win, win12)
            win = np.append(win, region3)
            win34 = 0.5 - 0.5*np.array([np.cos( np.pi*(float(j-ntaper3)/float(ntaper4-\
                                  ntaper3) - 1)) for j in np.arange(ntaper3,ntaper4)])
            win = np.append(win, win34)
            win = np.append(win, region5)
        else:
            raise IOError("Please specify valid taper-method")
        ##########################################################
        # ##########   Taper & Filter the waveform   #############
        ##########################################################
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
        #}}}

    #
    ###################################################################
    ####################################################################
    #
    # Mode amplitude, Energy, Flux of Energy, Orbital Angular Momentum
    #
    ####################################################################
    ###################################################################
    #
    def get_strain_modes_amplitudes(self, recalculate=False):
        """
Precompute amplitudes of all waveform modes.

** Does not change the S1 state of the object **
        """
        self.amplitudes = {}
        # Loop over all modes to be included
        for (modeL, modeM) in self.which_modes_to_read():
            if self.skipM0 and modeM == 0: continue
            if modeL not in self.amplitudes: self.amplitudes[modeL] = {}
            if modeM in self.amplitudes[modeL] and not recalculate: continue
            self.amplitudes[modeL][modeM] = self.get_mode_amplitude(
                modeL, modeM)
        return self.amplitudes

    #
    def get_bondi_news_modes(self, recalculate=True):
        """
Compute (l,m) modes of Bondi's News function: N_lm = \dot{h}

** Does not change the S1 state of the object **
        """
        if not recalculate and hasattr(self, "News"): return self.News
        self.News = {}
        # Loop over all modes to be included
        for (modeL, modeM) in self.which_modes_to_read():
            if self.skipM0 and modeM == 0: continue
            if modeL not in self.News: self.News[modeL] = {}
            if modeM in self.News[modeL] and not recalculate: continue
            h_ = self.data.modes[modeL][modeM].data()
            h_re = h_.real()
            h_im = h_.imag()
            # Here, we could use the interpolant created for each mode object
            Nre = TimeSeries(np.gradient(h_re.data, h_re.delta_t),
                             delta_t=h_re.delta_t,
                             dtype=real_same_precision_as(h_re),
                             epoch=h_._epoch)
            Nim = TimeSeries(np.gradient(h_im.data, h_im.delta_t),
                             delta_t=h_im.delta_t,
                             dtype=real_same_precision_as(h_im),
                             epoch=h_._epoch)
            self.News[modeL][modeM] = Nre + Nim * 1.0j
        return self.News

    #
    def get_psi4_modes(self, recalculate=False):
        """
Compute (l,m) modes of Psi4 from: \Psi_4 = - \ddot{h}

** Does not change the S1 state of the object **
        """
        self.get_bondi_news_modes(recalculate=recalculate)
        self.Psi4 = {}
        # Loop over all modes to be included
        for (modeL, modeM) in self.which_modes_to_read():
            if self.skipM0 and modeM == 0: continue
            if modeL not in self.Psi4: self.Psi4[modeL] = {}
            if modeM in self.Psi4[modeL] and not recalculate: continue
            h_ = self.News[modeL][modeM]
            h_re = h_.real()
            h_im = h_.imag()
            # Here, we could use the interpolant created for each mode object
            Pre = TimeSeries(np.gradient(h_re.data, h_re.delta_t),
                             delta_t=h_re.delta_t,
                             dtype=real_same_precision_as(h_re),
                             epoch=h_._epoch)
            Pim = TimeSeries(np.gradient(h_im.data, h_im.delta_t),
                             delta_t=h_im.delta_t,
                             dtype=real_same_precision_as(h_im),
                             epoch=h_._epoch)
            self.Psi4[modeL][modeM] = -1 * (Pre + Pim * 1.0j)
        return self.Psi4

    #
    def dEdt(self, lMax=8, recalculate=True):
        """
Compute dE/dt = \Sum_{l,m} ||h_lm||^2

** Does not change the S1 state of the object **
        """
        if not recalculate and hasattr(aelf, "ValuedEdt"):
            return self.ValuedEdt
        lMax = min(lMax, self.modeLmax)
        _ = self.get_strain_modes_amplitudes()
        momega = self.orbital_frequency()
        dEdt = np.zeros(len(momega))
        # Loop over all modes to be included
        for (modeL, modeM) in self.which_modes_to_read():
            if self.skipM0 and modeM == 0: continue
            if self.verbose > 1:
                print("Adding contribution to dEdt from mode ({},{})".format(
                    modeL, modeM))
            dEdt += modeM * modeM * self.amplitudes[modeL][
                modeM].data * self.amplitudes[modeL][modeM].data
        dEdt *= momega.data * momega.data / 8. / np.pi
        dEdt = TimeSeries(-1 * dEdt,
                          delta_t=momega.delta_t,
                          dtype=real_same_precision_as(momega),
                          epoch=momega._epoch)
        self.ValuedEdt = dEdt
        return dEdt

    #
    def dEdtfunc(self):
        """
Return a interpolant-function that computes dE/dt = \Sum_{l,m} ||h_lm||^2

** Does not change the S1 state of the object **
        """
        dEdt = self.dEdt()
        self.FuncdEdt = InterpolatedUnivariateSpline(dEdt.sample_times.data,
                                                     dEdt.data)
        return self.FuncdEdt

    #
    def E(self, discrete=True, lMax=8, useNews=False):
        """
Compute E = \int_0^T (dE/dt) dt, from start to end, as a function of time.

** Does not change the S1 state of the object **
        """
        lMax = min(lMax, self.modeLmax)
        if discrete:
            dEdt = self.dEdt(lMax=lMax)
            Edisc = cumtrapz(dEdt.data, dEdt.sample_times.data, initial=0)
            Edisc = TimeSeries(Edisc,
                               delta_t=dEdt.delta_t,
                               dtype=real_same_precision_as(dEdt),
                               epoch=dEdt._epoch)
        elif useNews:
            Nlm = self.get_bondi_news_modes()
            Edisc = None
            # Loop over all modes to be included
            for (modeL, modeM) in self.which_modes_to_read():
                if self.skipM0 and modeM == 0: continue
                AbsNlm = abs(Nlm[modeL][modeM])
                if Edisc is None:
                    Edisc = cumtrapz(AbsNlm.data,
                                     AbsNlm.sample_times.data,
                                     initial=0)
                else:
                    Edisc += cumtrapz(AbsNlm.data,
                                      AbsNlm.sample_times.data,
                                      initial=0)
            Edisc = TimeSeries(Edisc,
                               delta_t=AbsNlm.delta_t,
                               dtype=real_same_precision_as(AbsNlm),
                               epoch=dEdt._epoch)
        else:
            raise IOError("supply an integration method..?")
        self.ValueE = Edisc / 16. / np.pi
        return self.ValueE

    #
    def J(self, discrete=False, lMax=8, useNews=True):
        """
Compute J = [FIXME], from start to end, as a function of time.

** Does not change the S1 state of the object **
        """
        lMax = min(lMax, self.modeLmax)
        if discrete:
            raise IOError("discrete option not supported in J")
        elif useNews:
            Nlm = self.get_bondi_news_modes()
            Jdisc = None
            # Loop over all modes to be included
            for (modeL, modeM) in self.which_modes_to_read():
                if self.skipM0 and modeM == 0: continue
                prodJh = self.data.modes[modeL][modeM].data(
                ) * Nlm[modeL][modeM].conj()
                prodJh = prodJh.imag()
                if Jdisc is None:
                    Jdisc = modeM * cumtrapz(
                        prodJh.data, prodJh.sample_times.data, initial=0)
                else:
                    Jdisc += modeM * cumtrapz(
                        prodJh.data, prodJh.sample_times.data, initial=0)
            Jdisc = TimeSeries(Jdisc,
                               delta_t=Nlm[2][2].delta_t,
                               dtype=real_same_precision_as(Nlm[2][2]),
                               epoch=Nlm[2][2]._epoch)
        else:
            raise IOError("supply an integration method..?")
        self.ValueJ = Jdisc / 16. / np.pi
        return self.ValueJ

    #}}}


## Alias "nr_strain" to "nr_wave".
nr_wave = nr_strain


######################################################################
######################################################################
#
#### DEPRECATED
#
######################################################################
######################################################################
class strain():
    #{{{
    def __init__(self, filename=None, filetype='HDF', nogroup=False,\
                  modeLmin=2, modeLmax=8, modeM=2,\
                  sample_rate=8192, time_length=256, ex_order=-1, cce=True,
                  totalmass=None, verbose=True):
        self.filename = filename
        self.filetype = filetype
        self.modeLmin = modeLmin
        self.modeLmax = modeLmax
        self.sample_rate = sample_rate
        self.time_length = time_length
        self.cce = cce
        self.ex_order = ex_order
        self.nogroup = nogroup
        self.totalmass = totalmass
        self.verbose = True
        #
        self.amplitudes = None
        self.momega = None
        self.Ederiv = None
        self.Nlm = None
        return

    #
    def read_strain_modes(self):
        self.rawmodes, self.modes = {}, {}
        for ll in range(self.modeLmin, self.modeLmax + 1):
            self.modes[ll], self.rawmodes[ll] = {}, {}
            for mm in range(-1 * ll, ll + 1):
                if self.verbose:
                    print("Reading mode (%d,%d)" % (ll, mm), file=sys.stdout)
                self.modes[ll][mm] = nr_waveform( filename=self.filename,\
                                      filetype=self.filetype,\
                                      nogroup=self.nogroup, modeL=ll, modeM=mm,\
                                      sample_rate=self.sample_rate,\
                                      time_length=self.time_length,\
                                      ex_order=self.ex_order, cce=self.cce,\
                                      totalmass=self.totalmass,\
                                      rawdelta_t=-1 )
                self.rawmodes[ll][mm] = self.modes[ll][
                    mm].rawhp + self.modes[ll][mm].rawhc * 1j
                self.modes[ll][mm].rescale_to_totalmass(1)
        return

    #
    def good_indices(self):
        self.get_strain_modes_amplitudes()
        ap = self.amplitudes[2][2]
        mask = ap[ap.max_loc()[-1]:].data > 0.1 * ap.max_loc()[0]
        for i in range(len(mask)):
            if not mask[i]: break
        iend = ap.max_loc()[-1] + i
        return 500, iend

    #
    def peak_time(self, ll=2, mm=2):
        self.get_strain_modes_amplitudes()
        ap = self.amplitudes[ll][mm]
        return ap.sample_times[ap.max_loc()[-1]]

    #
    def peak_amplitude(self, ll=2, mm=2):
        self.get_strain_modes_amplitudes()
        ap = self.amplitudes[ll][mm]
        return ap.max_loc()[0]

    #
    #}}}
