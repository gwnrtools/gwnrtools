---
title: gwnr.stats
layout: default
parent: API Reference
nav_order: 4
permalink: /docs/api/stats/
---

# `gwnr.stats` — statistics & inference
{: .no_toc }

Bayesian-inference configuration, sampling utilities, Fisher matrices and distribution
containers. Source:
[`gwnr/stats/`](https://github.com/gwnrtools/gwnrtools/tree/master/gwnr/stats).

1. TOC
{:toc}

---

## `config_utils.py` — configuration infrastructure

- **`class ConfigWriter`** — writes stored config-file templates to disk, filling in
  formatting blanks (e.g. event-specific data options) via `write(name, **formatting_kwargs)`.
- **`class ConfigBase`** — stores config-file samples by category as a dict of dicts, and hands
  out `ConfigWriter`s (`available_configs()`, `get_config_writer(name)`, `get`/`set`).

These are the base classes for the PyCBC Inference and Bilby config generators below.

## `pycbc_inference_utils.py` — PyCBC Inference configs

**`class InferenceConfigs(ConfigBase)`** — stores and writes configuration files for
`pycbc_inference` runs: `add_data_configs(event_name)` (data/PSD options per GW event),
`add_sampler_configs(n_cpus=10, n_live=2000, n_maxmcmc=8000, d_logz=0.1, n_walkers=1000, n_temperatures=20, ...)`
(nested-sampling / ensemble-MCMC settings), and `add_inference_configs()` (model, priors,
variable/static parameters). Driven by the `gwnr_write_pycbc_inference_configs` CLI tool.

## `bilby_utils.py` — Bilby configs and scripts

- **`class InferenceConfigs(ConfigBase)`** — Bilby counterpart: default BBH priors, prior files,
  event and injection configs.
- **`class BilbyScriptWriterBase`** and subclasses **`BilbyScriptWriterInjection`** /
  **`BilbyScriptWriterEvent`** — programmatically compose complete Bilby run scripts section by
  section (imports, data/injection setup, template, priors, likelihood, sampler,
  post-processing) via `script_lines()`, `write_script()`, `write_prior_file()`.

## `lal_inference_utils.py` — LALInference posteriors

Utilities for `posterior_samples.dat` files produced by LALInference:

- `get_header_data_from_posterior_samples_file(filename, ...)`, `get_param_idx`,
  `get_param_from_names`, `get_1dslice_posterior`, `write_posterior_samples_file`
- **`get_h_from_posterior_line(line, header, det_tag, approx='IMRPhenomPv2', ...)`** — regenerate
  the waveform corresponding to any posterior sample (no conditioning applied; frequency-domain
  approximants are not converted to the time domain)
- `shift_waveform_phase_time(orig_line, phase_shift, time_shift, sample_rate)` — same, with
  additional phase/time shifts (SI units)

---

## `samplers.py` — emcee helpers

- **`get_emcee_ensemble_sampler(log_probability, params_to_sample, myarglist, ..., nwalkers=32, burn_in=100, backend_hdf=None, pool=None)`** —
  initialize and burn in an `emcee` ensemble sampler, optionally with an HDF5 checkpoint backend.
- **`emcee_samples_to_dict(sampler, params_to_sample, burnin=1000, thin=10)`** — pull thinned
  samples into a parameter-keyed dictionary.
- **`emcee_samples_from_checkpoint(checkpoint_file, ...)`** — same, from a checkpoint file.
- **`write_output_from_emcee_sampler(output_file_name, sampler, params_to_sample, ...)`** — dump
  samples to ASCII.

## `sampling.py` — drawing from standard distributions

Convenience samplers used to build injection sets and priors: `uniform_massratio`,
`uniform_mass`, `uniform_in_totalmass_massratio_masses`, `uniform_spin_magnitude`,
`uniform_coalescence_time`, `uniform_in_angle` / `uniform_in_cos_angle`, `uniform_on_S2`
(isotropic sky/spin directions via `cube_to_uniform_on_S2`), `uniform_in_volume_distance`,
`uniform_in_choices`, `zero_distribution`, `idempotence`.

**`class OneDRandom`** — meta-class mapping named parameters to their distributions (driven by a
`pandas.DataFrame` with a `dist` column); `sample(name, size=1)`.

## `distribution.py` — posterior containers

- **`class OneDDistribution`** — one-dimensional sample sets: `mean`, `median`, `percentile`,
  `kde`, `normalization`, range-restricted variants (`mean_in_range`, `pdf_over_range`).
- **`class MultiDDistribution`** — n-dimensional sample sets: named columns
  (`index_of_name`, `sliced`), `mean`/`median`/`percentile`/`credible_interval` per parameter,
  `kde`, `corner_plot`, `plot_twod_slice`.
- **`class MultipleOneDDistributions`** — combine 1-D marginal distributions across many events
  (e.g. stacking H₀ posteriors): `process_oned_slices`, `combine_oned_slices`,
  `plot_combined_oned_slices`.

---

## `fisher_information.py` — Fisher matrices

- **`get_waveform_derivatives_wrt_params(approximant='SEOBNRv2', ..., deriv_params=['mass1','mass2','spin1z','spin2z'], delta_m1=1e-3, ...)`** —
  numerical derivatives ∂h/∂θ of the waveform with respect to binary parameters, with adjustable
  finite-difference steps and tolerance.
- **`get_correlation_fisher_matrices(..., psd='aLIGOZeroDetHighPower', return_derivs=False, ...)`** —
  assembles the Fisher information matrix Γ_ij = (∂h/∂θ_i | ∂h/∂θ_j) and the corresponding
  correlation (inverse-Fisher) matrix for parameter-uncertainty estimates.

## `enigma_utils.py` — ENIGMA calibration likelihoods

Log-prior/likelihood/probability functions for calibrating attachment parameters of the
**ENIGMA/ESIGMA** eccentric waveform model against reference waveforms:
`log_prior_esigma`, `log_likelihood_esigma(mass1, mass2, omega_attach, PNO, f_lower, sample_rate, psd, ...)`
(match-based likelihood), `log_prob_esigma`, `log_prob_esigma_fixed_masses`,
`log_prob_esigma_fixed_total_mass_hidden_q`. Used by the `gwnr_enigma_*` CLI tools with the
`emcee` helpers above.

{: .note }
The ESIGMA waveform-generation code itself has been moved out of this repository; these
calibration utilities remain here.

## `priors.py`

Placeholder module for prior definitions (currently empty).
