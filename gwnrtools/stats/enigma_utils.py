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
import os
import logging
import numpy as np

import pycbc.pnutils as pnu
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.psd import from_string
from pycbc.filter import match


# Likelihood functions specific to this RUN
def log_prior_enigma(q, total_mass, PNO, coeffs, omega_attach, sp_info):
    '''
Priors:
-------

    '''
    PNO = int(np.round(PNO))

    if np.any(coeffs < -1.) or np.any(coeffs > 1.):
        logging.info(
            "Rejecting coeffs={} from prior for coeffs".format(coeffs))
        return -np.inf

    if PNO not in sp_info.PNO.range:
        logging.info("Rejecting PNO={} from prior on PNO".format(PNO))
        return -np.inf

    if omega_attach < sp_info.omega_attach.range[0] or \
            omega_attach > sp_info.omega_attach.range[-1]:
        logging.info(
            "Rejecting MOmg={} from prior on omega_attach".format(omega_attach))
        return -np.inf

    if q < sp_info.q.range[0] or q > sp_info.q.range[1]:
        logging.info("Rejecting q={} from prior on q".format(q))
        return -np.inf

    if total_mass < sp_info.total_mass.range[0] or total_mass > sp_info.total_mass.range[1]:
        logging.info(
            "Rejecting M={} from prior on total_mass".format(total_mass))
        return -np.inf

    return 0.0


def log_likelihood_enigma(mass1, mass2, omega_attach, PNO, f_lower, sample_rate,
                          psd, dilation_map_match=False):
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
    except:
        print("Could not generate ENIGMA wave..m1={},m2={},omg={},PNO={}".format(
            mass1, mass2, omega_attach, PNO))
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
        print("Could not generate EOB wave..")
        return -np.inf
    h2p = make_padded_frequency_series(h2p, N, df)
    #h2c = make_padded_frequency_series(h2c, N, df)

    # Undo BASH MAGIC TO PASS MCMC parameters TO ENIGMA
    os.environ['OMEGA_ATTACH'] = ''
    os.environ['PN_ORDER'] = ''

    # Compute inner prodcut
    log_like, _ = match(h1p, h2p, psd=psd, low_frequency_cutoff=f_lower)

    if dilation_map_match:
        def obj1(m): return np.log(m)

        def obj2(m, exp=30):
            return np.sin(m*np.pi/2) ** exp

        def match_map_for_likelihood(m, exp=30):
            return obj1(m) + obj2(m, exp=30)
        return match_map_for_likelihood(log_like)

    return -(1. - log_like)



class FitMOmegaIMRAttachmentNonSpinning():
    def __init__(self):
        return

    @staticmethod
    def fit_quadratic_poly(eta, coeffs):
        assert (len(coeffs) == 3), "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3 = coeffs
        return a1 + a2 * eta + a3 * eta * eta

    @staticmethod
    def fit_cubic_poly(eta, coeffs):
        assert (len(coeffs) == 4), "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, a4 = coeffs
        return a1 + a2 * eta + a3 * eta * eta + a4 * eta * eta * eta

    @staticmethod
    def fit_ratio_poly_44(eta, coeffs):
        assert (len(coeffs) == 7), "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, a4, b2, b3, b4 = coeffs
        return (a1 + a2 * eta + a3 * eta * eta + a4 * eta * eta * eta) / (1. + b2 * eta + b3 * eta * eta + b4 * eta * eta * eta)

    @staticmethod
    def fit_ratio_poly_43(eta, coeffs):
        assert (len(coeffs) == 6), "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, a4, b2, b3 = coeffs
        return (a1 + a2 * eta + a3 * eta * eta + a4 * eta * eta * eta) / (1. + b2 * eta + b3 * eta * eta)

    @staticmethod
    def fit_ratio_poly_34(eta, coeffs):
        assert (len(coeffs) == 6), "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b2, b3, b4 = coeffs
        return (a1 + a2 * eta + a3 * eta * eta) / (1. + b2 * eta + b3 * eta * eta + b4 * eta * eta * eta)


__order_of_sampled_params__ = ['q', 'total_mass',
                               'PNO', 'a1', 'a2', 'a3', 'a4', 'b2', 'b3', 'b4']

def log_prob_enigma(theta, inputs, f_lower, sampling_params, psd,
 dilation_map_match=False):
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

    # Evaluate attachment freq from coefficients a1-a4 and b2-b4
    fit = FitMOmegaIMRAttachmentNonSpinning()
    m_omega_attach = fit.fit_ratio_poly_44(eta, coeffs)

    # prior probability
    log_prior = log_prior_enigma(
        q, total_mass, PNO, coeffs, m_omega_attach, sampling_params)
    if not np.isfinite(log_prior):
        return log_prior

    # posterior = likelihood x prior
    return log_likelihood_enigma(mass1, mass2, m_omega_attach, PNO,
                                 f_lower, inputs.sample_rate, psd,
                                 dilation_map_match=dilation_map_match) + log_prior
