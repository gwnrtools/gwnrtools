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

import logging
logging.getLogger().setLevel(logging.INFO)

import emcee
import numpy as np

from gwnrtools.stats import OneDRandom


def get_emcee_ensemble_sampler(log_probability,
                               params_to_sample,
                               myarglist,
                               kwargs=None,
                               nwalkers=32,
                               burn_in=100,
                               backend_hdf=None,
                               pool=None,
                               verbose=False,
                               debug=False):
    """
Initializes and burns-in MCMC sampler

Inputs:
-------

log_probability : function
                  This method must take as input the parameters being sampled
                  (individually), and a list of other fixed parameters that
                  are needed to compute the probability density we want the
                  emcee sampler to sample. It should return the natural log
                  of that probability
params_to_sample: pandas.DataFrame
                  DF with a unique column for each parameter being sampled,
                  its first row indicating the type of variable it is, and
                  the second row indicating the allowed range etc
myarglist       : list
                  Non-variable parameters that are passed as-is to
                  `log_probability`
kwargs          : dict
                  Keyword arguments to be passed to `log_probability`
nwalkers        : int
                  Number of ensemble sampling walkers
pool            : mulitprocessing.pool object

Outputs:
--------
sampler         : emcee.EnsembleSampler object
state           : current state of sampler
p0              : 
    """
    # Setup hyper-parameters for the sampler
    ndim = params_to_sample.shape[-1]

    # Arguments for ensembleSampler
    kws = {'pool': pool, 'args': myarglist, 'kwargs': kwargs}

    # HDF5 backend to save progress
    if int(emcee.__version__.split('.')[0]) >= 3 and backend_hdf != None:
        logging.info("Initializing backend: {}".format(backend_hdf))
        backend = emcee.backends.HDFBackend(backend_hdf)
        backend.reset(nwalkers, ndim)
        kws['backend'] = backend
    else:
        logging.info("Ignoring backend because emcee major version: {} and backend provided: {}".format(
            int(emcee.__version__.split('.')[0]), backend_hdf))

    # Initialize emsemble sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, **kws)

    # Run the sampler for a few steps to burn-in,
    # ie erase memory of the starting locations
    if burn_in > 0:
        dist_sampler = OneDRandom(params_to_sample)

        initial_param_values = []
        for param in params_to_sample.columns:
            param_values = dist_sampler.sample(param, size=(nwalkers, 1))
            initial_param_values.append(param_values)
        p0 = np.hstack(initial_param_values)
        if debug:
            logging.info("DEBUG: will burn-in for {}".format(burn_in))
            logging.info("DEBUG: initial point shape: {}".format(p0.shape))
            logging.info("DEBUG: Initial point p0: {}".format(p0))
        state = sampler.run_mcmc(p0, burn_in)
        sampler.reset()
    return sampler, state, p0


# Single-point entry to above methods
get_sampler = {}
get_sampler['emcee_ensemble'] = get_emcee_ensemble_sampler


def emcee_samples_to_dict(sampler, params_to_sample):
    """
Receives a sampler object and retrieves samples for all
parameters from it. It returns a dictionary with all samples
for each parameter.

Inputs:
-------
sampler          : emcee.EnsembleSampler object
params_to_sample : pandas.DataFrame
                   DF with a unique column for each parameter being sampled,
                   its first row indicating the type of variable it is, and
                   the second row indicating the allowed range etc

Outputs:
--------
all_samples       : dict
                    Dictionary containing samples for all parameters,
                    with param names as keys.
    """
    try:
        _log_prob = sampler.get_log_prob().flatten()
    except AttributeError:
        _log_prob = sampler.lnprobability.flatten()
    mask_not_failed = np.isfinite(_log_prob)
    all_samples = {str(c): sampler.chain[..., idx].T.flatten()[
        mask_not_failed] for idx, c in enumerate(params_to_sample.columns)}
    all_samples['log_prob'] = _log_prob[mask_not_failed]
    return all_samples


def write_output_from_emcee_sampler(output_file_name, sampler, params_to_sample):
    """
Function to write output of an emcee ensemble sampler to ASCII (text) file

Inputs:
-------
output_file_name : str. Complete file path for output to disk.
sampler          : emcee.EnsembleSampler object
params_to_sample : pandas.DataFrame
                   DF with a unique column for each parameter being sampled,
                   its first row indicating the type of variable it is, and
                   the second row indicating the allowed range etc

**TODO**: thin samples by autocorrelation-length here.
    """
    # Simplify samples from all chains to a named dictionary
    all_samples = emcee_samples_to_dict(sampler, params_to_sample)
    # Prepare header and samples
    out_header = ''
    out_array = []
    for idx, p in enumerate(list(params_to_sample.columns) + ['log_prob']):
        out_header = out_header + '[{0}] {1}\n'.format(idx, p)
        out_array.append(all_samples[p])
    out_array = np.array(out_array)
    # write samples
    np.savetxt(output_file_name, out_array, delimiter='\t', header=out_header)
