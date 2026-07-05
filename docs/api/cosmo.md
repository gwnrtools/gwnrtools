---
title: gwnr.cosmo
layout: default
parent: API Reference
nav_order: 6
permalink: /docs/api/cosmo/
---

# `gwnr.cosmo` — cosmology
{: .no_toc }

Cosmological conversions used in GW population and rates work. Source:
[`gwnr/cosmo/utils.py`](https://github.com/gwnr/gwnr/blob/master/gwnr/cosmo/utils.py).

1. TOC
{:toc}

## Redshift and frame conversions

### `calculate_redshift(distance, h=0.679, om=0.3065, ol=0.6935, w0=-1.0)`

Redshift from luminosity distance using LAL's cosmology calculator. Defaults are the Planck 2015
`TT+lowP+lensing+ext` parameters (arXiv:1502.01589, Table 4): Ω_M = 0.3065, Ω_Λ = 0.6935,
H₀ = 67.9 km/s/Mpc. Accepts arrays.

### `source_to_detector_frame(m, z)` / `detector_to_source_frame(m, z)`

Convert masses between source frame and detector frame: m_det = m_src (1 + z).

## Redshift distributions for populations

Tools to draw merger redshifts assuming a rate density uniform in comoving volume:

- **`make_z_cosmo_inverseCDF(z_max, R0, H0, Omega_m, Omega_Lambda, Omega_k, w0, w1)`** — build
  the inverse CDF of the redshift distribution up to `z_max` for local rate `R0`.
- **`z_samples_from_iCDF(iCDF, N)`** — draw N redshift samples through the inverse CDF.
- **`probability_density_Uniform_comoving_volume(z, ...)`** — the corresponding pdf.
- **`dR_dz(z, ...)`, `dV_dz(z, ...)`** — differential merger rate and comoving-volume element.

## Background cosmology

Low-level functions parameterized by (H₀, Ω_m, Ω_Λ, Ω_k, w₀, w₁), supporting an evolving
dark-energy equation of state:

`H(z, ...)`, `OneOverH(z, ...)`, `E(z, w0, w1)`, `Hubble_integral(z_prime, ...)`,
`DL(z, ...)` and its vectorized form `DL_vector(z_arr, ...)` — Hubble rate, dimensionless
dark-energy factor, comoving-distance integral, and luminosity distance.
