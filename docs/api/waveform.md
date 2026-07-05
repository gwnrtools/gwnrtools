---
title: gwnr.waveform
layout: default
parent: API Reference
nav_order: 1
permalink: /docs/api/waveform/
---

# `gwnr.waveform` — waveform tools
{: .no_toc }

Generation, conditioning, alignment, hybridization and characterization of gravitational
waveforms. Source: [`gwnr/waveform/`](https://github.com/gwnr/gwnr/tree/master/gwnr/waveform).

1. TOC
{:toc}

---

## `waveform.py` — waveform generation

General-purpose entry points wrapping PyCBC/LAL generators.

### `get_waveform(approximant, phase_order, amplitude_order, spin_order, template_params, start_frequency, sample_rate, length, datafile=None, verbose=False)`

Generates a waveform for the point described by `template_params` (an object carrying masses,
spins, orientation angles — e.g. a `SimInspiral` row). Dispatches to the appropriate PyCBC
time-domain or frequency-domain generator based on `approximant`, and supports NR data files via
`datafile`.

### `get_polarizations_from_multipoles(waveform_multipoles, inclination, coa_phase, verbose=False)`

Assembles the plus/cross polarizations from a dictionary of complex spherical-harmonic multipoles
indexed by `(l, m)`, evaluating spin-weighted spherical harmonics at the given inclination and
coalescence phase.

### `project_polarizations_onto_detector(ifo_name, hp, hc, ra, dec, pol, tc, taper_mode='TAPER_NONE', amp_scaler=1.0)`

Projects polarizations onto a named detector (`H1`, `L1`, `V1`, `G1`, …) using its antenna
pattern for sky location `(ra, dec)`, polarization angle `pol` and coalescence time `tc`.
Optionally tapers the input via LAL taper modes.

---

## `align.py` — waveform alignment

Functions to time/phase-shift one waveform to agree with another. All operate on PyCBC
`TimeSeries` pairs `(hp, hc)` interpreted as `h = A(t) exp(-i Φ(t))`.

| Function | Alignment criterion |
|:---------|:--------------------|
| `shift_waveform_phase_time(hp, hc, t_shift, ph_shift, ...)` | Apply a given time and phase shift |
| `shift_waveform_phase(hp, hc, ph_shift, ...)` | Apply a given phase shift only |
| `shift_waveform_time(hp, hc, t_shift, ...)` | Apply a given time shift only |
| `align_waveforms_amplitude_peak(hp1, hc1, hp2, hc2, ...)` | Align the two waveforms at their amplitude peaks |
| `align_waveforms_at_frequency(hp1, hc1, hp2, hc2, falign, ...)` | Align where the instantaneous GW frequency crosses `falign` |
| `align_waveforms_optimally(hp1, hc1, hp2, hc2, psd='aLIGOZeroDetHighPower', ...)` | Iteratively determine and apply the time/phase shifts that maximize the noise-weighted inner product |
| `align_waveforms_suboptimally(...)` | Cheaper variant of the above |
| `align_curves(x1, y1, x2, y2, ...)` | Generic 1-D alignment: minimizes the integral of \|y2(x+Δx) − y1(x)\| over the offset Δx |

Most functions accept `shift_epochs_only` (shift epochs rather than rolling data),
`trim_leading` / `trim_trailing` (drop zeros introduced by shifting), and `verbose` flags.

---

## `condition.py` — conditioning and windowing

- **`smooth(x, window_len=11, window='flat')`** — moving-window smoothing by convolution with a
  chosen window (`flat`, `hanning`, …), with reflected-copy edge handling.
- **`moving_window_average(x, i, window_len=10)`** — average of `x` in a window centered on index `i`.
- **`planck_window(N, eps, one_sided=True, winstart=0)`** — Planck-taper window, the standard
  smooth turn-on used to avoid Gibbs artifacts when Fourier-transforming NR waveforms.
- **`windowing_tanh(waveform_array, bin_to_center_window, sharpness)`** — hyperbolic-tangent window.
- **`blend(hin, mm, sample, time, t_opt, WinID=-1)`** — blend (window) an NR waveform with a set
  of Planck-taper windows; `blendTimeSeries` is the deprecated `TimeSeries` variant.

---

## `hybridize.py` — inspiral ↔ merger-ringdown hybridization

Machinery to hybridize complex time series (waveform modes), aligning and blending an inspiral
model with merger–ringdown (typically NR) data over a frequency-selected window.

### `hybridize_modes(inspiral_modes, merger_ringdown_modes, inspiral_orbital_frequency, frq_attach, frq_width=10.0, delta_t=1/4096, no_sp=8, modes_to_hybridize=[(2,2),(3,3),(4,4)], mode_to_align_by=(2,2), hybridize_using_avg_orbital_frequency=True, hybridize_aligning_merger_to_inspiral=True, include_conjugate_modes=True, verbose=False)`

Hybridizes each requested `(l, m)` mode: locates the attachment window from the (optionally
averaged) orbital frequency around `frq_attach` with width `frq_width`, phase-aligns the
merger–ringdown against the inspiral using `mode_to_align_by`, and blends the two across the
window. Conjugate `(l, −m)` modes can be filled in automatically.

Supporting functions: `find_first_value_location_in_series` /
`find_last_value_location_in_series` (locate frequency crossings), `mismatch_discrete`,
`align_in_phase`, `blend_series`, and `compute_amplitude` / `compute_phase` /
`compute_frequency` for decomposing complex mode data.

---

## `eccentric.py` — eccentric binaries

- **`get_periastron_frequencies(hp, hc)` / `get_apastron_frequencies(hp, hc)`** — locate the
  periastron/apastron passages from oscillations in the instantaneous GW frequency of the
  polarizations, returning their times and frequencies (`get_peak_freqs` is the underlying
  peak-finder).
- **`eccentricity_at_extremum_frequency(mass1, mass2, spin1z, spin2z, e0, l0, f_lower, sample_rate, f_extremum, extremum='periastron', ...)`** —
  evolve an eccentric system from initial eccentricity `e0` and mean anomaly `l0` and measure the
  eccentricity when the chosen extremum sweeps through `f_extremum`.
- **`eccentricity_at_reference_frequency(..., f_reference, ...)`** — orbital eccentricity at a
  reference orbit-averaged frequency, given initial conditions at `f_lower`.
- **`get_eccentric_waveform_and_dynamics(...)`** — runs an external eccentric-IMR executable to
  produce both the coordinate trajectory and GW polarizations over a grid of masses and
  eccentricities.
- **`optimize_eccentricity(x1, y1, q, ...)`** — fits `(e, mean anomaly)` at `f_lower` so the
  model's radial evolution best matches a given (e.g. NR) trajectory.

---

## `parameters.py` — parameter conversions

Small, pure conversion helpers:

| Function | Meaning |
|:---------|:--------|
| `spins_to_PNeffective_spin(m1, m2, chi1, chi2)` | Leading-order PN effective spin |
| `spins_to_2PNeffective_spin(m1, m2, chi1, chi2)` | 2PN effective spin combination |
| `spins_to_massweighted_spin(m1, m2, chi1, chi2)` | Mass-weighted spin (χ_eff) |
| `spins_to_damoureffective_spin(m1, m2, chi1, chi2)` | Damour effective spin |
| `chip_from_masses_spins(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z)` | IMRPhenomPv2 precession parameter χ_p (assumes m1 > m2) |
| `q_to_eta(q)` / `eta_to_q(eta)` | Mass ratio ↔ symmetric mass ratio |

---

## `utils.py` — waveform utilities

- **`get_detector_response(ra, dec, psi, detector_tag, gmst=0)`** — antenna-pattern response
  factors for a detector.
- **`generate_detector_strain(template_params, h_plus, h_cross)`** — combine polarizations into
  detector strain using the source's sky/polarization angles.
- **`get_ncycles_to_merger(hp, hc)`** — number of GW cycles before merger.
- **`get_time_at_frequency_from_polarizations(hp, hc, fvalue)`, `get_time_at_frequency(fr, fvalue)`, `get_time_at_y(fr, fvalue)`** —
  locate when a frequency (or generic y-value) is attained in a `TimeSeries`.
- **`get_freq_crossings(freq, f0, df_threshold=0.4)`** — all crossing times of frequency `f0`
  (useful for eccentric signals where the frequency oscillates).
- **`get_isco_x`, `get_isco_frequency`, `f_ISCO_spin(mass1, mass2, spin1z, spin2z)`** — Kerr
  ISCO frequency fitting formulas for aligned-spin binaries.

---

## `tidal.py` — tidal corrections for NSBH/BNS

### `class tidalWavs`

Applies tidal amplitude and phase corrections on top of a point-particle frequency-domain model:
`tidalCorrectionAmplitude(mf, eta, sBH, tidalLambda)`, `tidalPNPhase`,
`tidalPNPhaseDeriv`, `tidalCorrectionPhase`, and
`getWaveform(M, eta, sBH, Lambda, distance=1e6*lal.PC_SI, f_lower=15.0, ...)` which returns the
tidally-corrected waveform.

### `random_match(...)`

Monte-Carlo study helper: draws random NSBH parameters and computes matches between tidal and
point-particle waveforms, writing results to `match.dat`.

---

## `nr_waveform_sxs.py` — SXS NR waveforms as templates

- **`get_nr_data_location(p, ...)`** — resolves the location of NR data corresponding to
  parameters `p`, via the `numrel_data` field, environment variables
  (`NR_CATALOG_PATH`, `NR_CATALOG_FILE`), or a template-bank-to-NR mapping.
- **`get_hplus_hcross_from_sxs(hdf5_file_name, template_params, delta_t, modeLmin=2, modeLmax=8, modeMmin=2, modeMmax=None, junk_duration=600, taper=True, ...)`** —
  reads SXS-format HDF5 modes, sums them at the requested orientation, rescales to physical
  mass/distance, removes junk radiation and tapers, returning `(hp, hc)`.
- **`get_hplus_hcross_from_get_td_waveform(**p)`** — adapter so NR waveforms can be generated
  through the standard `pycbc.waveform.get_td_waveform` interface.

(A parallel copy of this module lives at `gwnr.nr.nr_waveform_sxs`.)

---

## `prepare_waveforms.py` — SXS waveform preparation pipeline

### `class PrepareSXSWaveform`

Prepares raw SpEC/SXS simulation output for analysis, for a given resolution (`Lev`) and
eccentricity subdirectory:

1. `join_waveform_h5_files()` — join per-segment waveform HDF5 files,
2. `extrapolate(ch_mass=1.0, ...)` — extrapolate finite-radius waveforms to null infinity,
3. `join_horizons()` — join apparent-horizon data,
4. `transform_to_com_frame(...)` — correct for center-of-mass drift,
5. `prepare_waveform(...)` — run the full pipeline (optionally uploading results).

Properties expose the directory layout (`sim_dir`, `out_dir`, `joined_outfile_dir`,
`extrap_out_dir`, …). Requires SXS post-processing tools to be available.
