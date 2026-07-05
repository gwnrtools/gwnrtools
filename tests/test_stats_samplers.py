# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnr/gwnr/blob/master/LICENSE>
"""Tests for gwnr.stats.samplers (emcee helpers)"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("emcee")

from gwnr.stats.samplers import emcee_samples_to_dict, get_emcee_ensemble_sampler


def _log_gaussian(theta, mu, sigma):
    return -0.5 * np.sum(((theta - mu) / sigma) ** 2)


@pytest.mark.slow
def test_emcee_sampler_recovers_gaussian_mean():
    params = pd.DataFrame(
        {
            "x": pd.Series({"dist": "uniform", "range": (-10.0, 10.0)}),
            "y": pd.Series({"dist": "uniform", "range": (-10.0, 10.0)}),
        }
    )
    sampler, state, p0 = get_emcee_ensemble_sampler(
        _log_gaussian,
        params,
        [np.array([1.0, -2.0]), np.array([0.5, 0.5])],
        nwalkers=16,
        burn_in=200,
    )
    assert p0.shape == (16, 2)

    sampler.run_mcmc(state, 800)
    samples = emcee_samples_to_dict(sampler, params, burnin=200, thin=5)
    assert set(samples.keys()) >= {"x", "y"}
    assert np.isclose(np.mean(samples["x"]), 1.0, atol=0.2)
    assert np.isclose(np.mean(samples["y"]), -2.0, atol=0.2)
