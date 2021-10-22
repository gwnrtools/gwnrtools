# Copyright (C) 2019 Prayush Kumar
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
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
from __future__ import print_function

import os
import h5py
import glob
import numpy as np
import re
from gwnr.utils import find_nearest, approx_equal

########################################
# Auxiliary CLASSES
########################################


class ParsePVD():
    """
Elementary class to modify PVD files
    """

    # {{{

    def __init__(self, filename):
        self.pvdfin = open(filename, "r")
        self.pvdlines = self.pvdfin.readlines()
        self.orig_pvdlines = self.pvdlines
        return

    #

    def RemoveBreaks(self):
        # First count the number of breaks
        pvdrev = self.pvdlines[::-1]
        # Second remove all but the last
        revlist = []
        i = cnt = 0
        while i < len(pvdrev):
            pl = pvdrev[i]
            if re.search('</VTKFile>', pl) != None:
                cnt += 1
                if cnt == 1:
                    revlist.append(pvdrev[i])
                    revlist.append(pvdrev[i + 1])
                i += 2
                continue
            else:
                revlist.append(pl)
                i += 1
        self.pvdlines = revlist[::-1]
        return

    #

    def RetrieveUniqueTimeSteps(self):
        """
Returns time stamps of all unique time steps
        """
        self.tsteps = \
            np.array(list(set([float(re.findall('"\d+.\d+"', tl)[0].strip('"'))
                               for tl in self.pvdlines
                               if re.findall('timestep="\d+.\d+"', tl) != []])))
        self.tsteps.sort()
        return self.tsteps

    #

    def DownsampleTimeSteps(self, downsample_factor):
        """
Down-samples the PVD file by a given factor. This
operation is irreversible and cumulative!
        """
        self.RemoveBreaks()
        newpvdlines = []
        cnt = 0
        t_prev = t_curr = -1
        for tl in self.pvdlines:
            if re.findall('timestep="\d+.\d+"', tl) != []:
                t_curr = float(re.findall('"\d+.\d+"', tl)[0].strip('"'))
                # print t_prev, t_curr
                if not approx_equal(t_prev, t_curr, eps=1.e-9):
                    cnt += 1
                # print cnt % downsample_factor
                if cnt % downsample_factor == 0:
                    newpvdlines.append(tl)
                t_prev = t_curr
            else:
                newpvdlines.append(tl)
        self.pvdlines = newpvdlines
        self.RetrieveUniqueTimeSteps()

    def RemoveTimeSteps(self, remove_low_lim, remove_high_lim):
        """
Removes time-steps in a given range. This operation is irreversible
and cumulative!
        """
        newpvdlines = []
        for tl in self.pvdlines:
            if re.findall('timestep="\d+.\d+"', tl) != []:
                t_curr = float(re.findall('"\d+.\d+"', tl)[0].strip('"'))
                if t_curr < remove_low_lim or t_curr > remove_high_lim:
                    newpvdlines.append(tl)
            else:
                newpvdlines.append(tl)
        self.pvdlines = newpvdlines
        self.RetrieveUniqueTimeSteps()

    def WriteFile(self, filename):
        """
Writes out a new PVD file (presumably after processing).
        """
        with open(filename, "w") as fout:
            for pl in self.pvdlines:
                fout.write(pl)

    # }}}
