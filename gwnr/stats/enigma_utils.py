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
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""Likelihood functions specific to ENIGMA"""

from __future__ import absolute_import

import os
import logging
import traceback
import numpy as np

import pycbc.pnutils as pnu
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.psd import from_string
from pycbc.filter import match

from gwnr.utils import make_padded_frequency_series
from gwnr.stats import OneDRandom
from gwnr.waveform.enigma_utils import FitMOmegaIMRAttachmentNonSpinning

# TAGS for available fits
__available_fits__ = [
    'fit_quadratic_poly', 'fit_cubic_poly', 'fit_ratio_poly_44',
    'fit_ratio_sqrt_poly_44', 'fit_ratio_sqrt_hyb1_poly_44',
    'fit_ratio_poly_43', 'fit_ratio_sqrt_poly_43',
    'fit_ratio_sqrt_hyb1_poly_43', 'fit_ratio_poly_34'
]

# TAGGED list of sampled (free) parameters, in specific order
__order_of_sampled_params__ = {}

# TAGGED dicts of prior ranges for the sampled (free) parameters
# of all available fits
__ranges_of_sampled_params__ = {}

# TAG : fit_quadratic_poly
tmp = 'fit_quadratic_poly'
__order_of_sampled_params__[tmp] = ['PNO', 'a1', 'a2']
__ranges_of_sampled_params__[tmp] = {'PNO': [6, 7, 8, 9, 10, 11, 12]}

# TAG : fit_cubic_poly
tmp = 'fit_cubic_poly'
__order_of_sampled_params__[tmp] = ['PNO', 'a1', 'a2', 'a3']
__ranges_of_sampled_params__[tmp] = {'PNO': [6, 7, 8, 9, 10, 11, 12]}

tmp = 'fit_ratio_poly_44'
__order_of_sampled_params__[tmp] = ['PNO', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3']
__ranges_of_sampled_params__[tmp] = {'PNO': [6, 7, 8, 9, 10, 11, 12]}

tmp = 'fit_ratio_sqrt_poly_44'
__order_of_sampled_params__[tmp] = ['PNO', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3']
__ranges_of_sampled_params__[tmp] = {'PNO': [6, 7, 8, 9, 10, 11, 12]}

tmp = 'fit_ratio_sqrt_hyb1_poly_44'
__order_of_sampled_params__[tmp] = ['PNO', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3']
__ranges_of_sampled_params__[tmp] = {'PNO': [6, 7, 8, 9, 10, 11, 12]}

tmp = 'fit_ratio_poly_43'
__order_of_sampled_params__[tmp] = ['PNO', 'a1', 'a2', 'a3', 'b1', 'b2']
__ranges_of_sampled_params__[tmp] = {'PNO': [6, 7, 8, 9, 10, 11, 12]}

tmp = 'fit_ratio_sqrt_poly_43'
__order_of_sampled_params__[tmp] = ['PNO', 'a1', 'a2', 'a3', 'b1', 'b2']
__ranges_of_sampled_params__[tmp] = {'PNO': [6, 7, 8, 9, 10, 11, 12]}

tmp = 'fit_ratio_sqrt_hyb1_poly_43'
__order_of_sampled_params__[tmp] = ['PNO', 'a1', 'a2', 'a3', 'b1', 'b2']
__ranges_of_sampled_params__[tmp] = {'PNO': [6, 7, 8, 9, 10, 11, 12]}

tmp = 'fit_ratio_poly_34'
__order_of_sampled_params__[tmp] = ['PNO', 'a1', 'a2', 'b1', 'b2', 'b3']
__ranges_of_sampled_params__[tmp] = {'PNO': [6, 7, 8, 9, 10, 11, 12]}

# Set priors on coefficients used in all TAGGED fits
for tag in __available_fits__:
    for p in __order_of_sampled_params__[tag][1:]:  # exclude PNO
        __ranges_of_sampled_params__[tag][p] = [-30., 30.]


def log_prior_enigma(q,
                     total_mass,
                     PNO,
                     coeffs,
                     omega_attach,
                     sp_info,
                     omega_fit_tag,
                     verbose=False):
    '''
Priors:
-------

    '''
    # all_params has ordered sampler_params
    # __ranges_of_sampled_params__ [omega_fit_tag] has ALL ordered sampler params
    sampler_params = sp_info['sampler_params']
    ordered_fit_params = __ranges_of_sampled_params__[omega_fit_tag]

    if np.any(coeffs < -30.) or np.any(coeffs > 30.):
        if verbose:
            logging.info(
                "Rejecting coeffs={} from prior for coeffs".format(coeffs))
        return -np.inf

    PNO = int(np.round(PNO))
    if PNO not in sp_info.PNO.range and str(PNO) not in sp_info.PNO.range:
        if verbose:
            logging.info("Rejecting PNO={} from prior on PNO".format(PNO))
        return -np.inf

    if omega_attach < float(sp_info.omega_attach.range[0]) or \
            omega_attach > float(sp_info.omega_attach.range[-1]):
        if verbose:
            logging.info(
                "Rejecting MOmg={} from prior on omega_attach: {}".format(
                    omega_attach, sp_info.omega_attach.range))
        return -np.inf

    if len(sp_info.q.range) == 1:
        if q != float(sp_info.q.range[0]):
            if verbose:
                logging.info("Rejecting q={} from prior on q: {}".format(
                    q, sp_info.q.range))
            return -np.inf
    elif len(sp_info.q.range) == 2:
        if q < float(sp_info.q.range[0]) or q > float(sp_info.q.range[1]):
            if verbose:
                logging.info("Rejecting q={} from prior on q: {}".format(
                    q, sp_info.q.range))
            return -np.inf
    else:
        raise RuntimeError(
            "Unable to handle q prior with q={}, range={}".format(
                q, sp_info.q.range))

    if len(sp_info.total_mass.range) == 1:
        if total_mass != float(sp_info.total_mass.range[0]):
            if verbose:
                logging.info(
                    "Rejecting M={} from prior on total_mass: {}".format(
                        total_mass, sp_info.total_mass.range))
            return -np.inf
    elif len(sp_info.total_mass.range) == 2:
        if total_mass < float(
                sp_info.total_mass.range[0]) or total_mass > float(
                    sp_info.total_mass.range[1]):
            if verbose:
                logging.info(
                    "Rejecting M={} from prior on total_mass: {}".format(
                        total_mass, sp_info.total_mass.range))
            return -np.inf
    else:
        raise RuntimeError(
            "Unable to handle total_mass prior with M={}, range={}".format(
                total_mass, sp_info.total_mass.range))

    if verbose:
        logging.info("ACCEPTED q={}, M={}, PNO={}, coeffs={}, om={}".format(
            q, total_mass, PNO, coeffs, omega_attach))

    return 0.0


def log_likelihood_enigma(mass1,
                          mass2,
                          omega_attach,
                          PNO,
                          f_lower,
                          sample_rate,
                          psd,
                          dilation_map_match=False):
    '''
This function takes in all parameters, including:
- masses
- omega_attach
- PN order

and computes the inner product between the sampled ENIGMA
waveform and an equivalent EOB waveform m = <h_1|h_2>.

Finally returns L = exp(-0.5 x m x m)
    '''
    # extract MCMC parameters
    PNO = int(np.round(PNO))
    omega_attach = float(omega_attach)

    # Use BASH MAGIC TO PASS MCMC parameters TO ENIGMA
    os.environ['OMEGA_ATTACH'] = '{0:.12f}'.format(omega_attach)
    os.environ['PN_ORDER'] = '{0:d}'.format(PNO)

    dt = 1. / sample_rate
    df = psd.delta_f
    N = int(sample_rate / psd.delta_f)

    # Generate ENIGMA wave
    try:
        h1p, h1c = get_td_waveform(approximant='ENIGMA',
                                   mass1=mass1,
                                   mass2=mass2,
                                   f_lower=f_lower,
                                   delta_t=dt)
    except Exception as e:
        logging.error(traceback.format_exc())
        logging.warn(
            "Could not generate ENIGMA wave..m1={},m2={},omg={},PNO={}".format(
                mass1, mass2, omega_attach, PNO))
        logging.error("\n")
        return -np.inf
    h1p = make_padded_frequency_series(h1p, N, df)
    #h1c = make_padded_frequency_series(h1c, N, df)

    # Generate EOB wave
    try:
        h2p, h2c = get_fd_waveform(approximant='SEOBNRv4_ROM',
                                   mass1=mass1,
                                   mass2=mass2,
                                   f_lower=f_lower,
                                   delta_f=df)
    except:
        logging.info("Could not generate EOB wave..")
        return -np.inf
    h2p = make_padded_frequency_series(h2p, N, df)
    #h2c = make_padded_frequency_series(h2c, N, df)

    # Undo BASH MAGIC TO PASS MCMC parameters TO ENIGMA
    os.environ['OMEGA_ATTACH'] = ''
    os.environ['PN_ORDER'] = ''

    # Compute inner prodcut
    log_like, _ = match(h1p, h2p, psd=psd, low_frequency_cutoff=f_lower)

    if dilation_map_match:

        def obj1(m):
            return np.log(m)

        def obj2(m, exp=30):
            return np.sin(m * np.pi / 2)**exp

        def match_map_for_likelihood(m, exp=30):
            return obj1(m) + obj2(m, exp=30)

        return match_map_for_likelihood(log_like)

    return -(1. - log_like)


# Initialize an object here
fit = FitMOmegaIMRAttachmentNonSpinning()


def log_prob_enigma(theta,
                    inputs,
                    f_lower,
                    sampling_params,
                    psd,
                    dilation_map_match=False,
                    verbose=False,
                    ignore_samples_for=[]):
    '''
    Inputs:
    -------

    theta: (11) mass1, mass2, PNO, a1, a2, a3, a4, b1, b2, b3, b4
           (11)     q, mtotal,PNO, a1, a2, a3, a4, b1, b2, b3, b4
           (11)   eta, mtotal,PNO, a1, a2, a3, a4, b1, b2, b3, b4
    inputs: pandas.core.series.Series
            Attributes should include ('f_lower', 'sample_rate')
    psd: pycbc.FrequencySeries

    '''
    # Ordering is enforced here
    q, total_mass, PNO = theta[:3]
    coeffs = theta[3:]

    eta = q / (1. + q)**2
    mass1, mass2 = pnu.mtotal_eta_to_mass1_mass2(total_mass, eta)

    # Ignore sampling of certain parameters
    # mass1, mass2 = prior.sample(['mass1', 'mass2'])
    # if len(ignore_samples_for) > 0:
    #    for p in ignore_samples_for:
    #        p = prior.sample(p)

    # Evaluate attachment freq from coefficients a1-a4 and b2-b4
    fit = FitMOmegaIMRAttachmentNonSpinning()
    m_omega_attach = fit.fit_ratio_poly_44(eta, coeffs)

    # prior probability
    log_prior = log_prior_enigma(q, total_mass, PNO, coeffs, m_omega_attach,
                                 sampling_params, 'fit_ratio_poly_44')
    if not np.isfinite(log_prior):
        return log_prior

    # posterior = likelihood x prior
    return log_likelihood_enigma(
        mass1,
        mass2,
        m_omega_attach,
        PNO,
        f_lower,
        inputs.sample_rate,
        psd,
        dilation_map_match=dilation_map_match) + log_prior


def log_prob_enigma_fixed_masses(theta,
                                 inputs,
                                 f_lower,
                                 sampling_params,
                                 psd,
                                 omega_fit_tag,
                                 dilation_map_match=False,
                                 verbose=False,
                                 ignore_samples_for=[]):
    '''
    Inputs:
    -------

    theta: (11) PNO, a1, a2, a3, a4, b1, b2, b3, b4
    inputs: pandas.core.series.Series
            Attributes should include ('f_lower', 'sample_rate')
    psd: pycbc.FrequencySeries

    '''
    # Ordering is enforced here
    PNO = theta[0]
    coeffs = theta[1:]

    # Sample masses using information stored in the sampling_params obj
    oned_sampling_obj = OneDRandom(sampling_params)

    def ordered_masses(ss):
        a, b = ss.sample('mass1'), ss.sample('mass2')
        if a > b:
            return (a, b)
        else:
            return (b, a)

    mass1, mass2 = ordered_masses(oned_sampling_obj)
    q = mass1 / mass2
    eta = q / (1. + q)**2
    total_mass = mass1 + mass2

    # Evaluate attachment freq from coefficients a1-a4 and b2-b4
    try:
        m_omega_attach = eval('fit.{}(eta, coeffs)'.format(omega_fit_tag))
    except:
        logging.warn("Fit tag {} not recognized.".format(omega_fit_tag))
        raise

    # prior probability
    log_prior = log_prior_enigma(q,
                                 total_mass,
                                 PNO,
                                 coeffs,
                                 m_omega_attach,
                                 sampling_params,
                                 omega_fit_tag,
                                 verbose=verbose)
    if not np.isfinite(log_prior):
        return log_prior

    # posterior = likelihood x prior
    return log_likelihood_enigma(
        mass1,
        mass2,
        m_omega_attach,
        PNO,
        f_lower,
        inputs.sample_rate,
        psd,
        dilation_map_match=dilation_map_match) + log_prior


def log_prob_enigma_fixed_total_mass_hidden_q(theta,
                                              inputs,
                                              f_lower,
                                              all_params,
                                              psd,
                                              omega_fit_tag,
                                              dilation_map_match=False,
                                              verbose=False):
    '''
    Inputs:
    -------

    theta: (11) PNO, a1, a2, a3, a4, b1, b2, b3, b4
    inputs: pandas.core.series.Series
            Attributes should include ('f_lower', 'sample_rate')
    psd: pycbc.FrequencySeries

    '''
    # all_params has ordered sampler_params
    # __ranges_of_sampled_params__ [omega_fit_tag] has ALL ordered sampler params

    # Ordering is enforced here
    PNO = theta[0]
    coeffs = theta[1:]

    # Sample mass ratio using information stored in the sampling_params obj
    hidden_params = all_params[all_params.columns[[
        all_params[c].vartype == 'hidden' for c in all_params
    ]]]
    oned_sampler = OneDRandom(hidden_params)
    q = oned_sampler.sample('q')

    # Get fixed value of total mass
    total_mass = float(all_params['total_mass'].range[0])

    # Set other mass parameters
    eta = q / (1. + q)**2
    mass1, mass2 = pnu.mtotal_eta_to_mass1_mass2(total_mass, eta)

    # Evaluate attachment freq from coefficients a1-a4 and b2-b4
    try:
        m_omega_attach = eval('fit.{}(eta, coeffs)'.format(omega_fit_tag))
    except:
        logging.warn("Fit tag {} not recognized.".format(omega_fit_tag))
        raise

    # prior probability
    if np.random.uniform() < 0.01:
        debug = True
    else:
        debug = False

    log_prior = log_prior_enigma(q,
                                 total_mass,
                                 PNO,
                                 coeffs,
                                 m_omega_attach,
                                 all_params,
                                 omega_fit_tag,
                                 verbose=debug)

    if not np.isfinite(log_prior):
        return log_prior

    # posterior = likelihood x prior
    return log_likelihood_enigma(
        mass1,
        mass2,
        m_omega_attach,
        PNO,
        f_lower,
        inputs.sample_rate,
        psd,
        dilation_map_match=dilation_map_match) + log_prior


# TAGGED list of log(probability) functions
__log_prob_funcs__ = {}

# TAG : fit_quadratic_poly
tmp = 'fit_quadratic_poly'
__log_prob_funcs__[tmp] = log_prob_enigma_fixed_total_mass_hidden_q

# TAG : fit_cubic_poly
tmp = 'fit_cubic_poly'
__log_prob_funcs__[tmp] = log_prob_enigma_fixed_total_mass_hidden_q

tmp = 'fit_ratio_poly_44'
__log_prob_funcs__[tmp] = log_prob_enigma_fixed_total_mass_hidden_q

tmp = 'fit_ratio_sqrt_poly_44'
__log_prob_funcs__[tmp] = log_prob_enigma_fixed_total_mass_hidden_q

tmp = 'fit_ratio_sqrt_hyb1_poly_44'
__log_prob_funcs__[tmp] = log_prob_enigma_fixed_total_mass_hidden_q

tmp = 'fit_ratio_poly_43'
__log_prob_funcs__[tmp] = log_prob_enigma_fixed_total_mass_hidden_q

tmp = 'fit_ratio_sqrt_poly_43'
__log_prob_funcs__[tmp] = log_prob_enigma_fixed_total_mass_hidden_q

tmp = 'fit_ratio_sqrt_hyb1_poly_43'
__log_prob_funcs__[tmp] = log_prob_enigma_fixed_total_mass_hidden_q

tmp = 'fit_ratio_poly_34'
__log_prob_funcs__[tmp] = log_prob_enigma_fixed_total_mass_hidden_q
