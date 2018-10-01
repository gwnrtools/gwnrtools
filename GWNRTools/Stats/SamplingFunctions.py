#!/usr/bin/env python
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
import os, sys, time
import copy as cp
import commands as cmd
import numpy as np
import h5py

######################################################
# Constants
######################################################
verbose  = True
vverbose = True
debug    = True

quantiles_68 = [0.16, 0.5, 0.84] # 1-sigma
quantiles_95 = [0.0228, 0.5, 0.9772] # 2-sigma ~ 95.4%
percentiles_68 = 100*np.array(quantiles_68)
percentiles_95 = 100*np.array(quantiles_95)
perc_int_95 = [2.28, 97.72]
perc_int_99_7 = [0.13, 99.87] # 3 sigma

CILevels=[90.0, 68.26895, 95.44997, 99.73002]

_itime = time.time()
######################################################################
__author__   = "Prayush Kumar <prayush@astro.cornell.edu>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])
verbose      = True



######################################################################
######################################################################
#
#     UTILITIES FOR CREATING 1D / ND DISTRIBUTIONS
#
######################################################################
######################################################################
def uniform_CompactObject_massratio(N, q_min, q_max):
    return np.random.uniform(q_min, q_max, N)

def uniform_CompactObject_mass(N, comp_min, comp_max):
    return np.random.uniform(comp_min, comp_max, N)

def zero_distribution(N):
    return np.zeros(N)

def uniform_spin_magnitude(N, a_min, a_max):
    return np.random.uniform(a_min, a_max, N)

def coalescence_time(N, t_min=1170720018., t_max=1170806417.):
    return np.random.uniform(t_min, t_max, N)

def uniform_in_cos_angle(N,costheta_min=-1, costheta_max=1, offset=0.):
    return np.arccos(np.random.uniform(costheta_min,costheta_max,N)) + offset

def uniform_in_angle(N,theta_min=0., theta_max=2.*np.pi, offset=0.):
    return np.random.uniform(theta_min,theta_max,N) + offset

def CubeToUniformOnS2(u, v):
    """
    Taking in 2 random numbers drawn from U[0,1], and returns 2 that
    are uniformly distributed over the surface of a 2-sphere
    """
    if np.any(u < 0) or np.any(u > 1) or np.any(v < 0) or np.any(v > 1):
        raise IOError("Both inputs should be in [0,1]")
    return [2.*np.pi*u, np.arccos(2.*v - 1.)]





