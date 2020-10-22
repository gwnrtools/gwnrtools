# Copyright (C) 2014 Prayush Kumar, Heather Fong
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

import numpy as np

try:
    pass
except ImportError:
    print("Warning: Dont uniformly sample output")

try:
    pass
except:
    print("Warning: Could not import Psi4->h integration modules")

try:
    from glue.ligolw import ligolw, lsctables

    @lsctables.use_in
    class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
        pass
except:
    print("Warning: Could not import LAL/PyCBC modules")


def extrapolated_outdir_from_cce_outdir(outdir):
    #
    # Accept SKS_d19.8-q1-sA_0_0_-0.8_sB_0_0_-0.8
    # Return BBH_SKS_d19.8_q1_sA_0_0_-0.800_sB_0_0_-0.800
    #
    # {{{
    outdir = outdir.strip('/').split('/')[-1]
    try:
        idtype, dq, s1x, s1y, s1z, _, s2x, s2y, s2z = outdir.split('_')
    except ValueError:
        if outdir[0] == 'd':
            outdir = 'CF_' + outdir
            idtype, dq, s1x, s1y, s1z, _, s2x, s2y, s2z = outdir.split('_')
        else:
            raise ValueError('Cannot translate dir name to extrapolated dir')
    if idtype == 'CF':
        idtype += 'MS'
    d, q, _ = dq.split('-')
    print(q)
    if '.' in q:
        q = 'q%.2f' % np.float64(q[1:])
    if np.float64(d[1:]) == np.round(np.float64(d[1:])):
        d = 'd' + str(int(np.float64(d[1:])))
    print((s1z, s2z))
    if np.float(s1z) == 0.:
        s1z = '0'
    else:
        s1z = '%.3f' % np.float128(s1z)
    if np.float(s2z) == 0.:
        s2z = '0'
    else:
        s2z = '%.3f' % np.float128(s2z)
    retdir = 'BBH_%s_%s_%s_sA_%s_%s_%s_sB_%s_%s_%s' % (idtype, d, q, s1x, s1y,
                                                       s1z, s2x, s2y, s2z)
    return retdir
    # }}}


def initial_frequency_from_metadata(id_string, lev=None, xml_table=None):
    # {{{
    if xml_table == None:
        raise IOError("Please provide the catalog table")
    if lev == None:
        raise IOError("What Lev is the waveform..?")
    for line in xml_table:
        if id_string in line.waveform and lev in line.waveform:
            return line.f_lower
    raise IOError("Waveform not found in the catalog..! Lev missing?")
    # }}}
