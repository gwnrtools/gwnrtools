---
title: Command-line tools
layout: default
nav_order: 6
permalink: /docs/cli/
---

# Command-line tools
{: .no_toc }

Installing the package puts the scripts below on your `PATH` (see the `scripts` list in
[`setup.py`](https://github.com/gwnr/gwnr/blob/master/setup.py)). All are Python
scripts; run any of them with `--help` for the full option list. Additional unpackaged
scripts live under [`bin/`](https://github.com/gwnr/gwnr/tree/master/bin)
(NR run management under `bin/nr/`, plotting under `bin/graph/`, etc.).

Sample configuration files for several workflows are provided in
[`bin/configs/`](https://github.com/gwnr/gwnr/tree/master/bin/configs)
(`banksim.ini`, `faithsim.ini`, `pe_events.ini`).

1. TOC
{:toc}

---

## Template banks and banksims

| Tool | Purpose |
|:-----|:--------|
| `gwnr_create_bank_workflow` | Set up an HTCondor workflow that builds a stochastic template bank (iterative propose/reject placement) |
| `gwnr_create_banksim_workflow` | Set up a workflow measuring a bank's **effectualness**: match every injection against the bank |
| `gwnr_banksim` | The banksim worker executed by workflow nodes: computes matches between one injection set and one bank split |
| `choose_testpoints.py` | Propose new random test points for the stochastic bank iteration |
| `choose_best_testpoints.py` | Select the proposed points that survive the minimal-match rejection test |
| `remove_eliminated_testpoints.py` | Drop covered/rejected points from the proposal set |
| `banksim_generic.py` | Generic standalone banksim over arbitrary approximants |
| `split_table_geometrically.py` | Split a LIGO_LW table into geometrically sized chunks for parallel jobs |

A typical bank-construction cycle alternates `choose_testpoints.py` → banksim over the current
bank → `choose_best_testpoints.py` / `remove_eliminated_testpoints.py`, orchestrated by the DAG
that `gwnr_create_bank_workflow` writes.

## Faithfulness studies

| Tool | Purpose |
|:-----|:--------|
| `gwnr_create_faithsim_workflow` | Set up an HTCondor workflow computing faithfulness between two waveform models over a parameter-space sample |
| `gwnr_faithsim` | The faithsim worker: computes matches for one parameter split (see [`calculate_faithfulness`]({{ site.baseurl }}/docs/api/analysis/)) |
| `gwnr_sample_parameter_space` | Draw samples over binary parameter space (using [`gwnr.stats.sampling`]({{ site.baseurl }}/docs/api/stats/)) to feed the above workflows |

## Parameter estimation — PyCBC Inference

| Tool | Purpose |
|:-----|:--------|
| `gwnr_write_pycbc_inference_configs` | Write `pycbc_inference` configuration files (data, sampler, model/prior sections) from the templates in [`gwnr.stats.pycbc_inference_utils`]({{ site.baseurl }}/docs/api/stats/) |
| `gwnr_create_injections_pycbc_inference_workflow` | Set up a batch of inference runs on synthetic injections |
| `gwnr_create_public_events_pycbc_inference_workflow` | Set up a batch of inference runs on public GW events, fetching open strain data and PSDs automatically |

## Parameter estimation — Bilby

| Tool | Purpose |
|:-----|:--------|
| `gwnr_write_bilby_configs` | Write Bilby configuration/prior files from the templates in [`gwnr.stats.bilby_utils`]({{ site.baseurl }}/docs/api/stats/) |
| `gwnr_create_injections_bilby_workflow` | Batch Bilby runs on injections (writes per-run priors and run scripts) |
| `gwnr_create_public_events_bilby_workflow` | Batch Bilby runs on public GW events |

## ENIGMA model calibration

| Tool | Purpose |
|:-----|:--------|
| `gwnr_enigma_plan_calib_grid_and_make_dag` | Plan a calibration grid for the ENIGMA eccentric model and emit the HTCondor DAG |
| `gwnr_enigma_sample_calib_parameters` | MCMC-sample ENIGMA attachment parameters using the likelihoods in [`gwnr.stats.enigma_utils`]({{ site.baseurl }}/docs/api/stats/) |

## Analysis and post-processing

| Tool | Purpose |
|:-----|:--------|
| `ComputeOptimalSNRForGWSignals.py` | Compute optimal SNRs for a set of GW signals against chosen PSDs |
| `ComputeInferredParametersFromLIPosterior.py` | Derive additional physical parameters from LALInference posterior sample files |
| `JoinDatainHDF` | Join NR data split across multiple HDF5 files |

## Utilities

| Tool | Purpose |
|:-----|:--------|
| `gwnr_force_success_from_condor_sub` | Mark HTCondor jobs as succeeded from their submit files (for resuming DAGs) |
| `toggle_lsctable_type` | Toggle a LIGO_LW XML table between `sim_inspiral` and `sngl_inspiral` types |
| `ConvertHTMLToIpynb` | Convert an HTML-rendered notebook back to `.ipynb` |
| `makepdf` | Assemble figures into a PDF |

---

## Unpackaged script collections (`bin/`)

Not installed by `setup.py`, but included in the repository:

- **`bin/nr/SetupCCERuns/`** — set up, submit, resubmit and check Cauchy-characteristic
  extraction (CCE) runs for SpEC simulations.
- **`bin/nr/SimulationAnnex/`** — build and maintain catalogs of the SXS SimulationAnnex:
  gather run parameters, populate waveform locations, create injection sets, sync data.
- **`bin/nr/SpEC/`** — SpEC run management: continuation on specific clusters
  (Guillimin, Niagara), output-size reduction, extraction/truncation/model mismatch computations.
- **`bin/nr/spectre/`** — build SpECTRE with dependencies; combine element-wise volume data.
- **`bin/waveform/`** — waveform alignment experiments and EOB data extraction.
- **`bin/graph/`** — 2-D posterior density estimation and plotting (bounded 2-D KDEs).
- **`bin/stats/`** — LALInference file checks (NaN scan, nested-sample counting).
