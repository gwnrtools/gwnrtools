# Copyright (C) 2018 Prayush Kumar
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
"""UTILITIES FOR CREATING 1D / ND DISTRIBUTIONS"""

# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
import numpy as np

uniform_bound = np.random.uniform


def uniform_massratio(N, q_min, q_max):
    return uniform_bound(q_min, q_max, N)


def uniform_mass(N, comp_min, comp_max):
    return uniform_bound(comp_min, comp_max, N)


def uniform_in_totalmass_massratio_masses(N,
                                          mtotal_min, mtotal_max,
                                          q_min, q_max):
    from pycbc.pnutils import mtotal_eta_to_mass1_mass2
    from gwnrtools.waveform.parameters import q_to_eta

    mtotal_samples = uniform_bound(mtotal_min, mtotal_max, N)
    q_samples = uniform_bound(q_min, q_max, N)

    return mtotal_eta_to_mass1_mass2(mtotal_samples, q_to_eta(q_samples))


def zero_distribution(N):
    return np.zeros(N)


def uniform_spin_magnitude(N, a_min, a_max):
    return uniform_bound(a_min, a_max, N)


def uniform_coalescence_time(N, t_min=1170720018., t_max=1170806417.):
    return uniform_bound(t_min, t_max, N)


def uniform_in_cos_angle(N, costheta_min=-1, costheta_max=1, offset=0.):
    return np.arccos(np.random.uniform(costheta_min, costheta_max, N)) + offset


def uniform_in_angle(N, theta_min=0., theta_max=2.*np.pi, offset=0.):
    return uniform_bound(theta_min, theta_max, N) + offset


def cube_to_uniform_on_S2(u, v):
    """
    Taking in 2 random numbers drawn from U[0,1], and returns 2 that
    are uniformly distributed over the surface of a unit 2-sphere
    """
    if np.any(u < 0) or np.any(u > 1) or np.any(v < 0) or np.any(v > 1):
        raise IOError("Both inputs should be in [0,1]")
    return [2.*np.pi*u, np.arccos(2.*v - 1.)]


def uniform_on_S2(N):
    return cube_to_uniform_on_S2(uniform_bound(0, 1, N),
                                 uniform_bound(0, 1, N))


def uniform_in_volume_distance(N, d_min, d_max):
    return d_min + (d_max - d_min) * uniform_bound(0, 1., N)**(1./3.)


####
# **`OneDRandom`**:
# Metaclass holding a dictionary of methods to draw random numbers

class OneDRandom:
    '''
DESCRIPTION: Random number generation meta-class
    '''

    def __init__(self):
        self.draw = {}
        self.draw['uniform'] = np.random.uniform
        self.draw['zero'] = np.zeros
        self.draw['uniform_cos'] = uniform_in_cos_angle
        self.draw['uniform_S2'] = uniform_on_S2

    def available_distributions():
        return list(self.draw.keys())
