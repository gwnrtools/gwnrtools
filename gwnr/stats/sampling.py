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
import logging

logging.getLogger().setLevel(logging.INFO)

uniform_bound = np.random.uniform


def uniform_massratio(N, q_min, q_max):
    return uniform_bound(q_min, q_max, N)


def uniform_mass(N, comp_min, comp_max):
    return uniform_bound(comp_min, comp_max, N)


def uniform_in_totalmass_massratio_masses(N, mtotal_min, mtotal_max, q_min,
                                          q_max):
    from pycbc.pnutils import mtotal_eta_to_mass1_mass2
    from gwnr.waveform.parameters import q_to_eta

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


def uniform_in_angle(N, theta_min=0., theta_max=2. * np.pi, offset=0.):
    return uniform_bound(theta_min, theta_max, N) + offset


def cube_to_uniform_on_S2(u, v):
    """
    Taking in 2 random numbers drawn from U[0,1], and returns 2 that
    are uniformly distributed over the surface of a unit 2-sphere
    """
    if np.any(u < 0) or np.any(u > 1) or np.any(v < 0) or np.any(v > 1):
        raise IOError("Both inputs should be in [0,1]")
    return [2. * np.pi * u, np.arccos(2. * v - 1.)]


def uniform_on_S2(N):
    return cube_to_uniform_on_S2(uniform_bound(0, 1, N),
                                 uniform_bound(0, 1, N))


def uniform_in_volume_distance(N, d_min, d_max):
    return d_min + (d_max - d_min) * uniform_bound(0, 1., N)**(1. / 3.)


def uniform_in_choices(N, choices):
    return np.array(choices)[tuple([np.random.randint(1, len(choices), N)])]


def idempotence(N, x):
    return np.repeat(float(x), np.prod(N)).reshape(N)


####
# **`OneDRandom`**:
# Metaclass holding a dictionary of methods to draw random numbers


class OneDRandom:
    '''
DESCRIPTION: Random number generation meta-class

Input:
------
sampling_vars : pandas.DataFrame. It should have a column:
        - 'dist' that provides a distribution for it.
        - 'range' that provides the allowed numerical range for it.
    '''
    def __init__(self, sampling_vars=None):
        self.draw = {}
        self.draw['uniform'] = np.random.uniform
        self.draw['zero'] = np.zeros
        self.draw['uniform_cos'] = uniform_in_cos_angle
        self.draw['uniform_S2'] = uniform_on_S2
        self.draw['fixed'] = idempotence
        self.draw['choices'] = uniform_in_choices

        self.params = sampling_vars

    def available_parameters(self):
        return list(self.params.columns)

    def available_distributions(self):
        return list(self.draw.keys())

    def sample(self, name, size=1, dist=None):
        assert (self.params is not None),\
            "Provide a DataFrame of sampling parameter info at initialization"
        assert (name in self.params), "Cannot find info on {}".format(name)

        if dist is not None:
            if dist not in self.available_distributions():
                logging.info(
                    "Distribution {} not supported. See `available_distributions`."
                    .format(dist))
                dist = self.params[name].dist
        else:
            dist = self.params[name].dist

        # Final check on requested distribution
        if dist not in self.available_distributions():
            logging.info(
                "Distribution {} not supported. See `available_distributions`."
                .format(dist))
            return None

        sampling_func = self.draw[dist]
        sampling_lims = self.params[name].range

        if dist == 'uniform':
            return sampling_func(*sampling_lims, size=size)
        if dist == 'zero':
            return sampling_func(size)
        if dist == 'uniform_cos':
            return sampling_func(size, *sampling_lims)
        if dist == 'uniform_S2':
            return sampling_func(size)
        if dist == 'fixed':
            assert len(sampling_lims) == 1 or \
                len(set(np.array(sampling_lims).flatten())) == 1,\
                "Range of {}: {} does not support a fixed distribution".format(
                name, sampling_lims)
            return sampling_func(size, np.unique(sampling_lims)[0])
        if dist == 'choices':
            return sampling_func(size, sampling_lims)

        raise IOError("Failed to generate sample for {}, dist = {}.".format(
            name, dist))
