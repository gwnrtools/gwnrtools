---
title: gwnr.nr
layout: default
parent: API Reference
nav_order: 3
permalink: /docs/api/nr/
---

# `gwnr.nr` — numerical relativity
{: .no_toc }

Reading, characterizing and using numerical-relativity waveforms and simulation output, with
support for SXS/SpEC catalogs and the SpECTRE code. Source:
[`gwnr/nr/`](https://github.com/gwnr/gwnr/tree/master/gwnr/nr).

1. TOC
{:toc}

---

## `nr.types` — waveform containers

### `class nr_mode` (`types/single_mode.py`)

Container for a single spherical-harmonic mode h_lm(t):

- `data()`, `amplitude()`, `phase()`, `angular_velocity()`, `frequency()` — the mode and its
  derived time series (radians or cycles per time unit)
- `resample(delta_t)` / `resample_to_Hz(delta_t, total_mass, distance=1e6)` — resample in
  dimensionless or physical units
- `data_start_time()`, `data_end_time()`, `data_duration()` — extent bookkeeping

### `class nr_strain` (`types/strain.py`)

High-level container for a full mode set. It tracks an internal unit state ("dimensionless" vs
physical) and provides:

- **Mode access**: `get_mode_amplitude/frequency/phase(modeL, modeM, ..., dimensionless=...)`,
  `amplitude()`, `phase()`, `frequency()`, `orbital_frequency()`
- **Rescaling**: `make_modes_dimensionless()`, `rescale_modes(delta_t, M, distance)`,
  `rescale_to_totalmass(M)`, `rescale_to_distance(d)`, `rotate(inclination, phi)`
- **Polarizations**: `get_polarizations(delta_t, M, distance, inclination, phi)`
- **Frequency/time queries**: `get_t_frequency(f)`, `get_frequency_t(t)`,
  `get_lowest_binary_mass(f_lower, t_start)` — the lowest total mass the waveform can be scaled
  to while starting at `f_lower`
- **Conditioning**: `taper_filter_waveform(...)` — Planck-taper window plus high-pass filtering
- **Radiative quantities**: `get_bondi_news_modes()` (N_lm = ḣ_lm), `get_psi4_modes()`
  (Ψ₄ = −ḧ), `dEdt()`, `E()`, `J()` — energy/angular-momentum fluxes summed over modes

`class strain` is a lighter reader with `read_strain_modes()`, `peak_time()`, `peak_amplitude()`.

### `class nr_data` (`types/data_sxs.py`)

Low-level reader for NR data on disk — ASCII, HDF5 datasets, or SXS-format HDF5 groups
(`read_nr_data()`, `read_nr_data_hdf5()`, `get_nr_data_hdf5_wavetype()`, …).

---

## `nr_waveform_sxs.py` — SXS waveforms via the PyCBC interface

Same interface as [`gwnr.waveform.nr_waveform_sxs`]({{ site.baseurl }}/docs/api/waveform/#nr_waveform_sxspy--sxs-nr-waveforms-as-templates):
`get_nr_data_location`, `get_hplus_hcross_from_sxs`, and
`get_hplus_hcross_from_get_td_waveform`, which lets NR waveforms be requested through
`pycbc.waveform.get_td_waveform`-style calls.

---

## `utils.py` — PN/orbital helpers

- **`JOfOmegaNonSpinning(m, eta, om)`** — PN orbital angular momentum J(ω) for non-spinning
  binaries (Blanchet 2013 Living Review, Eq. 234).
- **`InnerProductVectors(v1, v2)`**, **`PhaseFromPlanarOrbit(rVec)`** — trajectory utilities.

---

## `nr.analysis` — NR waveforms in data analysis

### `analysis/filter.py`

- **`overlaps_vs_totalmass(wav1, wav2, psd=None, m_lower=-1, m_upper=100, m_delta=5, ...)`** —
  rescale two NR waveform objects across a range of total masses and compute their overlaps at
  each mass; the standard tool for quantifying NR truncation/extraction errors in a
  detection-relevant way.
- **`calculate_mismatch_between_levs_hdf5(...)`** — mismatches between resolutions (Levs) of the
  same simulation, written to HDF5.

### `analysis/types.py`

Classes abstracting HDF5 stores of overlap results:

- **`Overlaps`** — iterate over the directory/dataset structure of an overlaps file; retrieve
  `overlaps_vs_totalmass`, `overlaps(itr)`, `mismatches(itr)`.
- **`SimulationErrors`** — aggregates all error sources for one simulation (extraction method
  CCE vs extrapolation, extraction radii, resolution Levs); `get_max_cce_mismatch()` estimates
  the total error budget of the highest-resolution waveform.
- **`EffectualnessAndBias`** — reads effectualness/parameter-bias results
  (`read_data_from_combined_file`, `effectualness_vs_totalmass`, `best_match_parameters`,
  `parameterbiases_vs_parameters`) for template-model studies against NR signals.

### `analysis/support.py`

`extrapolated_outdir_from_cce_outdir(outdir)` and
`initial_frequency_from_metadata(id_string, ...)` — bookkeeping helpers for simulation output
layouts and metadata.

### `analysis/UseNRinDA_V1_08162018.py`

Frozen (dated) version of the original `nr_wave` class — mode rescaling to physical units,
polarizations, amplitude/frequency/phase extraction, peak finding, tapering — retained for
reproducibility of older analyses. Prefer `gwnr.nr.types.nr_strain` in new code.

---

## `nr.spec` — SpEC simulation output

`spec/utils.py` parses output of the **SpEC** (Spectral Einstein Code) NR code, whose runs are
split across segments and adaptive subdomains:

- **`ReadSpECTabularOutputFromASCII(DIR, LEV, FILE, ...)`** — combine `.dat`/`.txt` output across
  all segments of a run.
- **`ReadSpECTabularOutputWithColsFromASCII(...)`** / **`ReadSpECGlobalOutputWithColsFromASCII(...)`** —
  read subdomain-by-subdomain or global quantities into dictionaries keyed by the file header,
  handling subdomains that appear/disappear as the grid changes.
- **`ReadSpECTabularOutputFromH5(...)`**, **`ReadSpECTabularOutputWithColsFromH5(...)`**,
  **`ReadH5Dir(...)`** — the HDF5 equivalents, preserving file structure recursively.
- **`GetSpECRDStartTime(DIR, LEV)`**, **`GetSpECAhCAppearanceTime(DIR, LEV)`** — locate ringdown
  start / common-horizon appearance times.
- **`GetSegmentDirectories(...)`**, **`ParseHeaderForSpECTabularOutputASCII(...)`**,
  **`GetOpOfQuantityOverDomain(...)`** — segment discovery, header parsing, and reduction of
  per-subdomain quantities to a single time series.

---

## `nr.spectre` — SpECTRE evolutions

Batch management of test evolutions with the next-generation
[SpECTRE](https://spectre-code.org) code (`spectre/evolutions/`):

- **`class BatchEvolutions`** — plan, set up, submit and check a suite of runs:
  `available_tests()`, `check_exes()`, `setup_run(test, cluster, compiler, ...)`,
  `submit_to_cluster(test, cluster)`, `run(test, ...)`, `check_output(test)`,
  `read_output_file(test)`, `convert_volume_output_to_xdmf(test)`.
- **`configurations.py`** — input-file and cluster submission-script templates as Python
  dictionaries (`cluster_submission_file(cluster, **args)`).
- **`class HandleSpectreReductionDatum` / `HandleSpectreReductionData`** — read and plot reduction
  (error-norm) observables.
- **`class HandleSpectreVolumeDatum` / `HandleSpectreVolumeData`** — read volume data, extract
  fields and coordinates over time, convert to XDMF, and `make_movie(field_name, ...)` render
  field evolution movies.
