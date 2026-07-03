---
title: gwnr.analysis
layout: default
parent: API Reference
nav_order: 2
permalink: /docs/api/analysis/
---

# `gwnr.analysis` — data-analysis tools
{: .no_toc }

Matched-filter comparisons between waveform models, PSD utilities, GW transient catalog access,
and template-bank construction. Source:
[`gwnr/analysis/`](https://github.com/gwnrtools/gwnrtools/tree/master/gwnr/analysis).

1. TOC
{:toc}

---

## `filter.py` — matches, faithfulness and fitting factors

### `calculate_faithfulness(m1, m2, s1x=0, ..., signal_approx='IMRPhenomD', signal_file='', signal_h=None, tmplt_approx='IMRPhenomC', tmplt_file='', tmplt_h=None, aligned_spin_tmplt_only=True, non_spin_tmplt_only=False, f_lower=15.0, sample_rate=4096, signal_duration=32, psd_string='aLIGOZeroDetHighPower', verbose=True, debug=False)`

Computes the **match** (overlap maximized over time and phase) between a signal and a template
with the *same* physical parameters, as modeled by two different approximants. Signals/templates
may be specified by approximant name, by file, or as pre-generated waveforms (`signal_h` /
`tmplt_h`). Template spin components transverse to the orbital angular momentum can be zeroed
(`aligned_spin_tmplt_only`) or all spins dropped (`non_spin_tmplt_only`).

### `calculate_fitting_factor(m1, m2, tmplt_approx, ..., vary_masses_only=True, vary_masses_and_aligned_spin_only=False, chirp_mass_window=0.2, effective_spin_window=0.75, f_lower=15.0, sample_rate=4096, signal_duration=16, psd_string='aLIGOZeroDetHighPower', ff_max=0.99999, pso_swarm_size=100, pso_omega=0.5, pso_phip=0.5, pso_phig=0.25, pso_minfunc=0.001, pso_n_processes=1, num_retries=5, verbose=True, debug=False)`

Computes the **fitting factor**: the match additionally maximized over the template's physical
parameters, using particle-swarm optimization (via `pyswarm`). The search space is either
component masses only, or masses plus aligned spins, restricted to windows around the signal's
chirp mass and effective spin. PSO hyperparameters (swarm size, inertia `omega`, cognitive/social
weights `phip`/`phig`) are tunable, and the optimization is retried up to `num_retries` times.

### Other functions

- **`overlap_between_waveforms(wav1, wav2, psd, f_lower=15.0)`** — plain overlap between two
  already-generated waveforms.
- **`compute_snr_vs_time(wave, psd, time_step=0.01, f_lower=15.0)`** — accumulated SNR as a
  function of time.

---

## `psd.py` — power spectral densities

### `resample_and_extrapolate_psd(freq_vals, psd_vals, delta_f, f_max, precision=None, interpolation_func=scipy.interpolate.interp1d)`

Resamples measured PSD data onto a uniform frequency grid with spacing `delta_f` up to `f_max`,
and extrapolates toward f = 0 when the measurement doesn't extend below the physically measurable
band. Returns a PSD usable with PyCBC filtering.

---

## `gw_transient_catalog.py` — GW event catalogs

Extends `pycbc.catalog` with data-fetching conveniences.

### `class Merger(pycbc.catalog.Merger)`

Information about a specific compact-binary merger, with methods to locate and download the
public data around it:

- `operating_ifos(ignore_ifos=['G1'])`, `gpstime()`
- `frame_data_url(ifo, duration=32, sample_rate=4096)` / `frame_data_name(...)`
- `fetch_data(ifo, duration=32, sample_rate=4096, save_dir='')` — download strain frames
- `channel_name(ifo, sample_rate)`
- `psd_file_name(ifo)` / `fetch_psds(duration=32, sample_rate=4096, save_dir='')` — download the
  event PSDs

### `class Catalog(pycbc.catalog.Catalog)`

Catalog counterpart of the above. `get_psd_url(source, name)` resolves PSD download URLs per
catalog release.

---

## `utils.py` — bank/sim helpers

- **`get_uniform_mass_range(m_lower, m_upper, m_sep)`** — uniformly spaced mass values.
- **`outside_mchirp_window(bank, sim, w)`** — is a bank point outside a fractional chirp-mass
  window of an injection? (Used to prune match computations.)
- **`outside_tau0_window(bank, sim, window, f_lower)`** — same, in Newtonian chirp time τ₀.

---

## `template_banks/` — template-bank construction

Tools for building and manipulating template banks stored as LIGO_LW XML tables.

- **`ConvertTableType.py`** — convert between `sngl_inspiral` and `sim_inspiral` tables
  (`invert_tabletype`, `new_row`).
- **`MapTableToNRData.py`** — map template-bank rows to NR catalog simulations by matching
  physical parameters (`does_this_map(p, c, param='eta', ...)`).
- **`TurnCatalogIntoInjections.py`** — turn an NR catalog into an injection set, resolving each
  simulation's waveform data location (CCE, extrapolated, or finite-radius files) from run
  metadata.

### `template_banks/stochastic_bank/` — stochastic placement pipeline

A self-contained pipeline for **stochastic template-bank placement**: propose random points in
parameter space, reject those whose match with the existing bank exceeds the minimal-match
threshold, and iterate to convergence. Key scripts:

| Script | Role |
|:-------|:-----|
| `script1.py` … `script4.py` | Core proposal/rejection iterations (`get_new_sample_point`, `reject_new_sample_point`, waveform generation per point) |
| `ChooseTestPoints.py` | Draw random test points (masses, spins, angles) with acceptance regions for nonspinning / aligned-spin banks |
| `banksim.py` | Bank-simulation engine: pad waveforms to power-of-2 lengths, generate detector strain, compute matches |
| `cut_bank.py`, `split_table.py` | Restrict a bank to a parameter region; split tables for parallel jobs |
| `push_eta_bank.py`, `push_chi_bank.py`, `move_eta_bank.py` | Push bank boundaries in symmetric mass ratio / spin |
| `take_uncovered_add_to_bank.py`, `match_combine.py`, `checkundone.py` | Combine match results, add uncovered points, track job completion |
| `make_dag.py` | Generate the HTCondor DAG for the whole procedure |
| `plotConvergance.py`, `plot_injection_match_mult.py`, `recovered_hist.py` | Convergence and coverage diagnostics |
| `EtasToQs.py`, `QsToEtas.py` | Convert bank coordinates between η and q |

These are batch scripts rather than an importable API; the installed
[command-line tools]({{ site.baseurl }}/docs/cli/) (`gwnr_create_bank_workflow`,
`choose_testpoints.py`, …) drive them.
