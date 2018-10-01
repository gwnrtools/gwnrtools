#!/usr/bin/env python
# Copyright (C) 2015 Prayush Kumar
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
import os, sys
import time
import numpy as np
from pyswarm import pso
from glob import glob
import scipy as sp
import math

from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
from scipy.optimize import minimize_scalar

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

from pycbc.pnutils import nearest_larger_binary_number
from pycbc.waveform import get_td_waveform, get_fd_waveform
import pycbc.waveform as pywf
import pycbc.waveform.generator as pywfg
import pycbc.pnutils as pnutils
from pycbc.filter import match, make_frequency_series
from pycbc.psd import from_string


__author__  = "Prayush Kumar <prayush@astro.cornell.edu>"
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
def get_uniform_mass_range( m_lower, m_upper, m_sep ):
  #{{{
  mlist = [m_lower]
  for m in np.arange( np.ceil(m_lower), np.floor(m_upper), m_sep ):
    mlist.append( m )
  mlist.append( m_upper )
  return np.array( mlist )
  #}}}



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
    #/* From this we can compute chip*/
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
        if ASp2 > ASp1: num = ASp2
        else: num = ASp1
        if m2 > m1: den = A2 * m2_2
        else: den = A1 * m1_2
    chip = num / den; #  chip = max(A1 Sp1, A2 Sp2) / (A_i m_i^2) for i index of larger BH (See Eqn. 32 in technical document)
    return chip
