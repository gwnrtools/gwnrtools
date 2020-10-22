# Copyright (C) 2020 Prayush Kumar
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
import numpy as np
import pandas as pd

# BBH
default_bbh_params = pd.DataFrame({})

# Binary source - dynamical parameters
default_bbh_params['mass1'] = ('fixed', [10.0])
default_bbh_params['mass2'] = ('fixed', [10.0])
default_bbh_params['q'] = ('fixed', [1.])
default_bbh_params['total_mass'] = ('fixed', [20.])

default_bbh_params['spin1x'] = ('fixed', [0.])
default_bbh_params['spin1y'] = ('fixed', [0.])
default_bbh_params['spin1z'] = ('fixed', [0.])
default_bbh_params['spin2x'] = ('fixed', [0.])
default_bbh_params['spin2y'] = ('fixed', [0.])
default_bbh_params['spin2z'] = ('fixed', [0.])

# Binary source - kinematic parameters
default_bbh_params['coa_phase'] = ('fixed', [0.])
default_bbh_params['inclination'] = ('fixed', [0.])

default_bbh_params['right_ascension'] = ('fixed', [0.])
default_bbh_params['declination'] = ('fixed', [0.])
default_bbh_params['polarization'] = ('fixed', [0.])

# Set names
default_bbh_params = default_bbh_params.set_index(pd.Index(['dist', 'range']))

# BBH ranges
default_bbh_param_ranges = pd.DataFrame({})

# Binary source - dynamical parameters
default_bbh_param_ranges['mass1'] = ('continuous', [1., 100.])
default_bbh_param_ranges['mass2'] = ('continuous', [1., 100.])
default_bbh_param_ranges['q'] = ('continuous', [1., 4.])
default_bbh_param_ranges['total_mass'] = ('continuous', [2., 100.])

default_bbh_param_ranges['spin1x'] = ('continuous', [-1., 1.])
default_bbh_param_ranges['spin1y'] = ('continuous', [-1., 1.])
default_bbh_param_ranges['spin1z'] = ('continuous', [-1., 1.])
default_bbh_param_ranges['spin2x'] = ('continuous', [-1., 1.])
default_bbh_param_ranges['spin2y'] = ('continuous', [-1., 1.])
default_bbh_param_ranges['spin2z'] = ('continuous', [-1., 1.])

# Binary source - kinematic parameters
default_bbh_param_ranges['coa_phase'] = ('continuous', [0., 2. * np.pi])
default_bbh_param_ranges['inclination'] = ('continuous', [-1. * np.pi, np.pi])

default_bbh_param_ranges['right_ascension'] = ('continuous', [0., 2. * np.pi])
default_bbh_param_ranges['declination'] = ('continuous', [-1. * np.pi, np.pi])
default_bbh_param_ranges['polarization'] = ('continuous', [0, 2. * np.pi])

# Set names
default_bbh_param_ranges = default_bbh_param_ranges.set_index(
    pd.Index(['vartype', 'range']))

# BNS
default_bns_params_ranges = pd.DataFrame({})

# Binary source - dynamical parameters
default_bns_params_ranges['mass1'] = ('continuous', [1., 3.])
default_bns_params_ranges['mass2'] = ('continuous', [1., 3.])
default_bns_params_ranges['q'] = ('continuous', [1., 3.])
default_bns_params_ranges['total_mass'] = ('continuous', [2., 6.])

default_bns_params_ranges['spin1x'] = ('continuous', [-1., 1.])
default_bns_params_ranges['spin1y'] = ('continuous', [-1., 1.])
default_bns_params_ranges['spin1z'] = ('continuous', [-1., 1.])
default_bns_params_ranges['spin2x'] = ('continuous', [-1., 1.])
default_bns_params_ranges['spin2y'] = ('continuous', [-1., 1.])
default_bns_params_ranges['spin2z'] = ('continuous', [-1., 1.])

# Binary source - kinematic parameters
default_bns_params_ranges['coa_phase'] = ('continuous', [0., 2. * np.pi])
default_bns_params_ranges['inclination'] = ('continuous', [-1. * np.pi, np.pi])

default_bns_params_ranges['right_ascension'] = ('continuous', [0., 2. * np.pi])
default_bns_params_ranges['declination'] = ('continuous', [-1. * np.pi, np.pi])
default_bns_params_ranges['polarization'] = ('continuous', [0, 2. * np.pi])

# Set names
default_bns_params_ranges = default_bns_params_ranges.set_index(
    pd.Index(['vartype', 'range']))

# List of the names of possible CBC internal parameters
__all_cbc_parameters__ = [
    'mass1', 'mass2', 'q', 'eta', 'total_mass', 'chirp_mass', 'mchirp',
    'spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z', 'spin1',
    'spin2', 'a1', 'a2', 'distance', 'luminosity_distance', 'coa_phase',
    'inclination', 'iota', 'theta_jn', 'alpha', 'delta', 'ra',
    'right_ascension', 'dec', 'declination', 'psi', 'polarization'
]
