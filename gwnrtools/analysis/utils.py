# Copyright (C) 2015 Prayush Kumar
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
import pycbc.pnutils as pnutils
import numpy as np

try:
    pass
except:
    pass

from glue.ligolw import ilwd




def get_uniform_mass_range(m_lower, m_upper, m_sep):
    # {{{
    mlist = [m_lower]
    for m in np.arange(np.ceil(m_lower), np.floor(m_upper), m_sep):
        mlist.append(m)
    mlist.append(m_upper)
    return np.array(mlist)
    # }}}


#############################


def outside_mchirp_window(bank, sim, w):
    # template mchirp
    if hasattr(bank, "mchirp"):
        bmchirp = bank.mchirp
    elif hasattr(bank, "mass1") and hasattr(bank, "mass2"):
        bmchirp, eta =\
            pnutils.mass1_mass2_to_mchirp_eta(bank.mass1, bank.mass2)
    elif hasattr(bank, "mtotal") and hasattr(bank, "eta"):
        bmchirp = bank.mtotal * (bank.eta**0.6)
    # signal / injection / proposal mchirp
    if hasattr(sim, "mchirp"):
        smchirp = sim.mchirp
    elif hasattr(sim, "mass1") and hasattr(sim, "mass2"):
        smchirp, eta =\
            pnutils.mass1_mass2_to_mchirp_eta(sim.mass1, sim.mass2)
    elif hasattr(sim, "mtotal") and hasattr(sim, "eta"):
        smchirp = sim.mtotal * (sim.eta**0.6)
    return abs(smchirp - bmchirp) > (w * bmchirp)


def outside_tau0_window(bank, sim, window, f_lower):
    b_tau0, _ = pnutils.mass1_mass2_to_tau0_tau3(getattr(bank, 'mass1'),
                                                 getattr(bank, 'mass2'),
                                                 f_lower)
    s_tau0, _ = pnutils.mass1_mass2_to_tau0_tau3(getattr(sim, 'mass1'),
                                                 getattr(sim, 'mass2'),
                                                 f_lower)
    return abs(b_tau0 - s_tau0) > window
