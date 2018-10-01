#!/usr/bin/env python
# Copyright (C) 2017 Prayush Kumar
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
from matplotlib import use
use('Agg')
import sys, os

import h5py
import glob
import commands as cmd
from numpy import *
import numpy as np
import scipy as sp
import string
import time
import copy as cp

from pyswarm import pso

from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
from scipy.optimize import minimize_scalar

import lal
import lalsimulation as ls

try:
  from glue.ligolw import ligolw, lsctables
  @lsctables.use_in
  class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
    pass
except:
  print "Could not import ligolw in %s, LIGO XML tables wont be read" % __file__

from pycbc.pnutils import nearest_larger_binary_number
from pycbc.detector import *
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.types import TimeSeries, FrequencySeries, complex_same_precision_as, real_same_precision_as
#from pycbc.waveform import *
#from pycbc.types import *

_itime = time.time()
###############################
__author__   = "Prayush Kumar <prayush@astro.cornell.edu>"
PROGRAM_NAME = os.path.abspath(os.path.split(sys.argv[0])[0])
SELF_NAME    = __file__
verbose      = True
###############################


######################################################################
######################################################################
#
#     POSTERIOR UTILITIES
#
######################################################################
######################################################################
def get_param_idx(param_name, header):
    if param_name in header:
        return np.where([param_name == x for x in header])[0][0]
    else:
        raise IOError("%s not found in header" % param_name)

def get_header_data_from_posterior_samples_file(filename, no_of_samples=-1):
    with open(filename, 'r') as fp:
        header = fp.readline()
        header = header.split()
        # Allow for an initial hash = # in case the posterior is saved by
        # numpy.savetxt()
        if header[0] == '#': header = header[1:]
        data = fp.readlines()
        for idx in range(len(data)):
            data[idx] = data[idx].split()
        data = np.array(np.float64(np.array(data)) )
        if no_of_samples > 0 and no_of_samples <= np.shape(data)[0]:
            if verbose:
                print "Keeping only the first %d samples" % no_of_samples
            data = data[:no_of_samples]
    return [header, data]


def get_param_from_names(line, names, header):
    param = None
    for name in names:
        try:
            param = float(line[get_param_idx(name, header)] )
            break
        except IOError: continue
    return param


def get_h_from_posterior_line(line, header, det_tag, approx='IMRPhenomPv2',
                                 delta_f=None, delta_t=None,
                                 filter_n=0, filter_N=0,
                                 return_polarizations=False,
                                 debug=True):
    """
Generate Waveform for LALInference Posterior parameters.
Note: No conditioning is applied.
Note: Frequency domain approximants are NOT converted to Time Domain

Input:
1) [REQUIRED] any line from the posterior_samples.dat file containing all parameters
2) [REQUIRED] header string at the top of posterior_samples.dat that tells which column
   contains what parameter
3) [Default: PhenomP] Approximant!
4) [REQUIRED] delta_f / delta_t: depending on the approximant
5) [REQUIRED] Length of the final vector. Standardizing on length makes filtering easier.
    """
    #{{{
    # Get lower frequency cutoff
    flow = float(line[get_param_idx('flow', header)])

    # Get masses
    m1 = get_param_from_names(line, ['m1_source', 'm1'], header)
    m2 = get_param_from_names(line, ['m2_source', 'm2'], header)

    # Get angles needed to calculate spins
    theta_jn = float(line[get_param_idx('theta_jn', header)])
    phi_jl   = float(line[get_param_idx('phi_jl', header)] )
    phi12    = float(line[get_param_idx('phi12', header)])
    theta1 = get_param_from_names(line, ['theta1', 'tilt1'], header)
    theta2 = get_param_from_names(line, ['theta2', 'tilt2'], header)

    # Get spin magnitudes
    chi1 = float(line[get_param_idx('a1', header)])
    chi2 = float(line[get_param_idx('a2', header)])

    # Get reference-time quantities
    f_ref   = float(line[get_param_idx('f_ref', header)])
    phi_ref = float(line[get_param_idx('phase', header)])

    # Get initial spins in LAL frame
    # DEPENDS ON BRANCH: FIXME
    # See https://github.com/lscsoft/lalsuite/blob/lalinference_o2/lalsimulation/src/LALSimInspiralSpinTaylor.c#L3341
    # Vs https://github.com/lscsoft/lalsuite/blob/master/lalsimulation/src/LALSimInspiral.c#L4000
    try:
        # This won't work on master
        incl, s1x, s1y, s1z, s2x, s2y, s2z = \
            ls.SimInspiralTransformPrecessingNewInitialConditions(\
                theta_jn, phi_jl, theta1, theta2, phi12, chi1, chi2, m1, m2, f_ref)
    except TypeError:
        # This works on master
        # FIXME: Should we use convention from 'cbcBayesPosToSimInspiral.py' instead as
        #phi_ref = float(line[get_param_idx('phi1', header)])
        #phi12   = float(line[get_param_idx('phi2', header)]) - phi_ref
        incl, s1x, s1y, s1z, s2x, s2y, s2z = \
            ls.SimInspiralTransformPrecessingNewInitialConditions(\
                theta_jn, phi_jl, theta1, theta2, phi12, chi1, chi2, m1, m2, f_ref, phi_ref)

    # Get sky location
    ra   = float(line[get_param_idx('ra', header)])
    dec  = float(line[get_param_idx('dec', header)])
    psi  = float(line[get_param_idx('psi', header)])
    dist = get_param_from_names(line, ['distance', 'dist'], header)

    # Get signal end time -- FIXME, do we use detector time or geocentre time?
    #gps_end_time = float(line[get_param_idx(string.lower(det_tag) + '_end_time', header)])
    gps_end_time = float(line[get_param_idx('time', header)])
    gmst         = lal.GreenwichMeanSiderealTime( gps_end_time )

    # Get antenna response functions
    # Assume it does not change over the course of signal
    Fp, Fc = get_detector_response(ra, dec, psi, det_tag, gmst=gmst )

    # Get PhenomP waveform in source frame
    if delta_f is not None:
        hp_1, hc_1 = get_fd_waveform(approximant = approx,
                                mass1 = m1, mass2 = m2,
                                spin1x=s1x, spin1y=s1y, spin1z=s1z,
                                spin2x=s2x, spin2y=s2y, spin2z=s2z,
                                inclination = incl,
                                coa_phase = phi_ref,
                                distance = dist,
                                f_lower = flow,
                                delta_f = delta_f)
        # Convert to detector frame
        ht_1 = Fp * hp_1 + Fc * hc_1
        if filter_n == 0:
            raise IOError("Please provide filter_n with delta_f")
        ht_1 = extend_waveform_FrequencySeries(ht_1, filter_n)
    elif delta_t is not None:
        if 'NRSur7dq2' in approx:
            print "Doubling FLOw"
            flow = flow * 2
        try:
            hp_1, hc_1 = get_td_waveform(approximant = approx,
                                mass1 = m1, mass2 = m2,
                                spin1x=s1x, spin1y=s1y, spin1z=s1z,
                                spin2x=s2x, spin2y=s2y, spin2z=s2z,
                                inclination = incl,
                                coa_phase = phi_ref,
                                distance = dist,
                                f_lower = flow, f_ref = flow,
                                delta_t = delta_t)
            print "Length of raw %s waveform: %d" % (approx, len(hp_1))
        except RuntimeError as re:
            print "WAVEFORM GENERATION FAILED WITH MESSAGE:\n ", re, "\n..."
            if debug:
                print """
            Try this command to reproduce failure:
            get_td_waveform(approximant='%s', mass1=%f, mass2=%f,
                          spin1x=%f, spin1y=%f, spin1z=%f,
                          spin2x=%f, spin2y=%f, spin2z=%f,
                          inclination=%f, distance=%f,
                          f_lower=%f, delta_t=%f)
                """ % (approx, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, incl, dist, flow, delta_t)
            return None
        # Convert to detector frame
        ht_1 = Fp * hp_1 + Fc * hc_1
        if filter_N == 0:
            raise IOError("Please provide filter_N with delta_t")
        ht_1 = extend_waveform_TimeSeries(ht_1, filter_N)
    else:
        raise IOError("Supply either delta_f or delta_t")

    # Set coalescence time to "time"
    ht_1._epoch = gps_end_time

    # Return polarizations too
    if return_polarizations:
        return ht_1, hp_1, hc_1
    # Return h(t)
    return ht_1
    #}}}


def shift_waveform_phase_time(orig_line, phase_shift, time_shift, sample_rate):
    """
Generate waveform for a given point in LI posterior samples, with an additional
phase shift applied, as well as a time shift.

SI Units required: phase shift (radians); time shift(seconds)
    """
    line = cp.deepcopy( orig_line )
    line[get_param_idx("phase", header)] += phase_shift
    # Get SEOBNRv3 h(t) for posterior sample
    ht_1 = get_hoft_from_posterior_line(line, header, det_tag,\
                                        approx='SEOBNRv3',\
                                        delta_t=delta_t, filter_N=filter_N)
    ht_1.roll( int(time_shift * sample_rate) )
    return ht_1

def get_1dslice_posterior(data_iter, header, param='logl'):
    return np.array([float(line[get_param_idx(param, header)] ) for line in data_iter])


def write_posterior_samples_file(filename, header, data):
    header_string = ''
    for param in header: header_string += ('%s\t' % param)
    np.savetxt(filename, data, delimiter='\t', header=header_string)
    return

######################################################################
######################################################################
#
#     BINARY UTILITIES
#
######################################################################
######################################################################
def ParamLatexLabels():
    return {
        'm1':r'$m_1\,(\mathrm{M}_\odot)$',
        'm2':r'$m_2\,(\mathrm{M}_\odot)$',
        'mc':r'$\mathcal{M}\,(\mathrm{M}_\odot)$',
        'eta':r'$\eta$',
        'q':r'$q$',
        'mtotal':r'$M_\mathrm{total}\,(\mathrm{M}_\odot)$',
        'm1_source':r'$m_{1}^\mathrm{source}\,(\mathrm{M}_\odot)$',
        'm2_source':r'$m_{2}^\mathrm{source}\,(\mathrm{M}_\odot)$',
        'mtotal_source':r'$M_\mathrm{total}^\mathrm{source}\,(\mathrm{M}_\odot)$',
        'mc_source':r'$\mathcal{M}^\mathrm{source}\,(\mathrm{M}_\odot)$',
        'redshift':r'$z$',
        'mf':r'$M_\mathrm{final}\,(\mathrm{M}_\odot)$',
        'mf_source':r'$M_\mathrm{final}^\mathrm{source}\,(\mathrm{M}_\odot)$',
        'af':r'$a_\mathrm{final}$',
        'e_rad':r'$E_\mathrm{rad}\,(\mathrm{M}_\odot)$',
        'l_peak':r'$L_\mathrm{peak}\,(10^{56}\,\mathrm{ergs}\,\mathrm{s}^{-1})$',
        'spin1':r'$S_1$',
        'spin2':r'$S_2$',
        'a1':r'$a_1$',
        'a2':r'$a_2$',
        'a1z':r'$a_{1z}$',
        'a2z':r'$a_{2z}$',
        'theta1':r'$\theta_1\,(\mathrm{rad})$',
        'theta2':r'$\theta_2\,(\mathrm{rad})$',
        'phi1':r'$\phi_1\,(\mathrm{rad})$',
        'phi2':r'$\phi_2\,(\mathrm{rad})$',
        'chi_eff':r'$\chi_\mathrm{eff}$',
        'chi_tot':r'$\chi_\mathrm{total}$',
        'chi_p':r'$\chi_\mathrm{P}$',
        'tilt1':r'$t_1\,(\mathrm{rad})$',
        'tilt2':r'$t_2\,(\mathrm{rad})$',
        'costilt1':r'$\mathrm{cos}(t_1)$',
        'costilt2':r'$\mathrm{cos}(t_2)$',
        'iota':r'$\iota\,(\mathrm{rad})$',
        'cosiota':r'$\mathrm{cos}(\iota)$',
        'time':r'$t_\mathrm{c}\,(\mathrm{s})$',
        'time_mean':r'$<t>\,(\mathrm{s})$',
        'dist':r'$d_\mathrm{L}\,(\mathrm{Mpc})$',
        'ra':r'$\alpha$',
        'dec':r'$\delta$',
        'phase':r'$\phi\,(\mathrm{rad})$',
        'psi':r'$\psi\,(\mathrm{rad})$',
        'theta_jn':r'$\theta_\mathrm{JN}\,(\mathrm{rad})$',
        'costheta_jn':r'$\mathrm{cos}(\theta_\mathrm{JN})$',
        'beta':r'$\beta\,(\mathrm{rad})$',
        'cosbeta':r'$\mathrm{cos}(\beta)$',
        'phi_jl':r'$\phi_\mathrm{JL}\,(\mathrm{rad})$',
        'phi12':r'$\phi_\mathrm{12}\,(\mathrm{rad})$',
        'logl':r'$\mathrm{log}(\mathcal{L})$',
        'h1_end_time':r'$t_\mathrm{H}$',
        'l1_end_time':r'$t_\mathrm{L}$',
        'v1_end_time':r'$t_\mathrm{V}$',
        'h1l1_delay':r'$\Delta t_\mathrm{HL}$',
        'h1v1_delay':r'$\Delta t_\mathrm{HV}$',
        'l1v1_delay':r'$\Delta t_\mathrm{LV}$',
        'lambdat' : r'$\tilde{\Lambda}$',
        'dlambdat': r'$\delta \tilde{\Lambda}$',
        'lambda1' : r'$\lambda_1$', 'lambda2': r'$\lambda_2$',
        'lam_tilde' : r'$\tilde{\Lambda}$',
        'dlam_tilde': r'$\delta \tilde{\Lambda}$',
        'calamp_h1' : r'$\delta A_{H1}$', 'calamp_l1' : r'$\delta A_{L1}$',
        'calpha_h1' : r'$\delta \phi_{H1}$', 'calpha_l1' : r'$\delta \phi_{L1}$',
        'polar_eccentricity':r'$\epsilon_{polar}$',
        'polar_angle':r'$\alpha_{polar}$',
        'alpha':r'$\alpha_{polar}$',
        'dchi0':r'$d\chi_0$', 'dchi1':r'$d\chi_1$', 'dchi2':r'$d\chi_2$',
        'dchi3':r'$d\chi_3$', 'dchi4':r'$d\chi_4$', 'dchi5':r'$d\chi_5$',
        'dchi5l':r'$d\chi_{5}^{(l)}$', 'dchi6':r'$d\chi_6$',
        'dchi6l':r'$d\chi_{6}^{(l)}$', 'dchi7':r'$d\chi_7$',
        'dxi1':r'$d\xi_1$', 'dxi2':r'$d\xi_2$', 'dxi3':r'$d\xi_3$',
        'dxi4':r'$d\xi_4$', 'dxi5':r'$d\xi_5$', 'dxi6':r'$d\xi_6$',
        'dalpha1':r'$d\alpha_1$', 'dalpha2':r'$d\alpha_2$', 'dalpha3':r'$d\alpha_3$',
        'dalpha4':r'$d\alpha_4$', 'dalpha5':r'$d\alpha_5$',
        'dbeta1':r'$d\beta_1$', 'dbeta2':r'$d\beta_2$', 'dbeta3':r'$d\beta_3$',
        'dsigma1':r'$d\sigma_1$', 'dsigma2':r'$d\sigma_2$',
        'dsigma3':r'$d\sigma_3$', 'dsigma4':r'$d\sigma_4$',
        }

