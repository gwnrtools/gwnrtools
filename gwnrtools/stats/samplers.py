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
import emcee
import numpy as np


def get_emcee_ensemble_sampler(log_probability,
                               params_to_sample,
                               myarglist,
                               nwalkers=32,
                               pool=None):
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
nwalkers        : int
                  Number of ensemble sampling walkers
pool            : mulitprocessing.pool object

Outputs:
--------
sampler         : emcee.EnsembleSampler object
state           : current state of sampler
    """
    # Setup hyper-parameters for the sampler
    ndim = params_to_sample.shape[-1]

    # Initialize emsemble sampler
    sampler = emcee.EnsembleSampler(nwalkers,
                                    ndim,
                                    log_probability,
                                    pool=pool,
                                    args=myarglist)

    # Run the sampler for a few steps to burn-in,
    # ie erase memory of the starting locations
    initial_param_values = []
    for param in params_to_sample.columns:
        if params_to_sample[param]['vartype'] == 'continuous':
            param_values = np.random.uniform(params_to_sample[param]['range'][0],
                                             params_to_sample[param]['range'][-1],
                                             (nwalkers, 1))
        elif params_to_sample[param]['vartype'] == 'discrete':
            param_values = np.random.choice(params_to_sample[param]['range'],
                                            (nwalkers, 1))
        initial_param_values.append(param_values)
    p0 = np.hstack(initial_param_values)

    state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    return sampler, state


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
    _log_prob = sampler.get_log_prob().flatten()
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
    for idx, p in enumerate(params_to_sample.columns + ['log_prob']):
        out_header = out_header + '[{0}] {1}\n'.format(idx, p)
        out_array.append(all_samples[p])
    out_array = np.array(out_array)
    # write samples
    np.savetxt(output_file_name, out_array, delimiter='\t', header=out_header)
