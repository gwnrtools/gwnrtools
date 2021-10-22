# Copyright (C) 2020 Prayush Kumar
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
from __future__ import absolute_import

import os
import glob


def gw_noise_curve_file(filename):
    """ Return path to ASCII file containing noise power spectral
    density estimates for various GW detectors, as a function of
    frequency.
    """
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'gw_noise_curves', filename))


def available_gw_noise_curves():
    """Returns a list of noise curves whose data is available.
    """
    return [
        f.split('/')[-1]
        for f in glob.glob(os.path.dirname(__file__) + '/gw_noise_curves/*')
    ]
