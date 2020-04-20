#!/usr/bin/env python
# Copyright (C) 2015 Prayush Kumar
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
from pycbc.pnutils import nearest_larger_binary_number
from pycbc.psd import from_string
from pycbc.filter import match, make_frequency_series
import pycbc.pnutils as pnutils
import pycbc.waveform.generator as pywfg
import pycbc.waveform as pywf
from pycbc.waveform import get_td_waveform, get_fd_waveform
import os
import sys
import time
import numpy as np
from glob import glob
import scipy as sp
import math
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
from scipy.optimize import minimize_scalar

try:
    from pyswarm import pso
except:
    pass

from glue.ligolw import ilwd
from glue import gpstime, git_version
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import process as ligolw_process
import lalsimulation as ls
import lal


@lsctables.use_in
class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
    pass


__author__ = "Prayush Kumar <prayush@astro.cornell.edu>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])
__itime__ = time.time()


######################################################################
######################################################################
#
#     MISCELLANEOUS FUNCTIONS FOR GW DATA ANALYSIS
#
######################################################################
######################################################################

#############################
def get_unique_hex_tag(N=1, num_digits=10):
    import random
    if N == 1:
        return '%0{}x'.format(num_digits) % random.randrange(16**num_digits)
    else:
        return ['%0{}x'.format(num_digits) % random.randrange(16**num_digits)
                for i in range(N)]


def get_sim_hash(N=1, num_digits=10):
    return ilwd.ilwdchar(":{}:0".format(get_unique_hex_tag(N=N, num_digits=num_digits)))

#############################


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
                                                 getattr(bank, 'mass2'), f_lower)
    s_tau0, _ = pnutils.mass1_mass2_to_tau0_tau3(getattr(sim, 'mass1'),
                                                 getattr(sim, 'mass2'), f_lower)
    return abs(b_tau0 - s_tau0) > window

######################################################################
######################################################################
#
#      PARAMETER SPACE FUNCTIONS
#
######################################################################
######################################################################


def get_chip_from_masses_spins(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z):
    """
    Compute the IMRPhenomPv2 chi-precessing "chi_p" parameter, given
    component masses and spins for a binary.

    NOTE: Assumes convention m1 > m2
    """
    m1_2, m2_2 = m1**2, m2**2
    # Magnitude of the spin projections in the orbital plane */
    S1_perp = m1_2*np.sqrt(s1x*s1x + s1y*s1y)
    S2_perp = m2_2*np.sqrt(s2x*s2x + s2y*s2y)
    # /* From this we can compute chip*/
    A1 = 2. + (3.*m2) / (2*m1)
    A2 = 2. + (3.*m1) / (2*m2)
    ASp1 = A1*S1_perp
    ASp2 = A2*S2_perp
    if type(m1) != float and type(m1) != int:
        num = np.zeros(len(m1))
        den = np.zeros(len(m1))
        #
        mask = ASp2 > ASp1
        num[mask] = ASp2[mask]
        mask = ASp2 <= ASp1
        num[mask] = ASp1[mask]
        #
        den = A1*m1_2
    else:
        if ASp2 > ASp1:
            num = ASp2
        else:
            num = ASp1
        if m2 > m1:
            den = A2 * m2_2
        else:
            den = A1 * m1_2
    # chip = max(A1 Sp1, A2 Sp2) / (A_i m_i^2) for i index of larger BH (See Eqn. 32 in technical document)
    chip = num / den
    return chip


def get_q_from_eta(eta):
    import numpy as np
    return ((1. - 2.*eta + np.sqrt(1. - 4.*eta))/(2.*eta))
