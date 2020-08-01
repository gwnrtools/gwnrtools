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
