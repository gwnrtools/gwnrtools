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
"""
gwnr is a toolkit for gravitational-wave physics
"""
from __future__ import absolute_import

from . import (analysis, cosmo, data, graph, nr, utils, waveform, workflow)
try:
    from . import stats
except:
    pass
from gwnr.utils import *


def get_version_information():
    import os
    version_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "gwnr/.version")
    try:
        with open(version_file, "r") as f:
            return f.readline().rstrip()
    except EnvironmentError:
        print("No version information file '.version' found")


__version__ = get_version_information()
