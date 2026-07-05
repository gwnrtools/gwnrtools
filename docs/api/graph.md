---
title: gwnr.graph
layout: default
parent: API Reference
nav_order: 5
permalink: /docs/api/graph/
---

# `gwnr.graph` — visualization
{: .no_toc }

Plotting utilities for posteriors, model-comparison studies and simulation data. Source:
[`gwnr/graph/`](https://github.com/gwnr/gwnr/tree/master/gwnr/graph).

1. TOC
{:toc}

---

## `corner.py` — corner plots

### `class CornerPlot(MultiDDistribution)`

Corner-plot generator built on the
[`MultiDDistribution`]({{ site.baseurl }}/docs/api/stats/#distributionpy--posterior-containers)
container: pass ≥2-dimensional data (columns = parameters) and call `draw(...)`.

`draw(params_plot, ...)` is highly configurable; the main option groups are:

- **Panel type**: `plot_type='scatter'` or `'contour'`; 2-D panels can show data points or
  percentile contours (`contour_levels=[68.27, 90.0, 95.45]`, line styles, inline labels,
  optionally return enclosed areas via `return_areas_in_contours`)
- **Truth values and priors**: `true_params_vals`, `params_oned_priors` overlay injected values
  and 1-D priors
- **Point coloring**: `param_color` colors scatter points by a third parameter, with
  `cmap`, `color_min`/`color_max`, and a colorbar label
- **1-D marginals**: histogram type/bins, median line (`show_oned_median`), percentile band
  (`show_oned_percentiles=90.0`), labeling and placement
- **Layout**: existing `fig`/`axes_array` reuse, axis limits (`plim_low`/`plim_high`),
  fonts, legends, grid, figure title

### `cornerplot_dataframe(df, cols=None)`

One-liner corner plot from a `pandas.DataFrame` (defaults to all columns).

---

## `misc.py` — general plotting helpers

- **`set_matplotlib_params()`** — house style for matplotlib.
- **`make_filled_contour_plot(x, y, z, ...)`** — filled contours from scattered (x, y, z) data,
  with a choice of interpolators (`Rbf`, `griddata`, `SmoothBivariateSpline`) and per-interpolator
  defaults.
- **`make_2Dplot_errorbars(Xs, Ys, Xerrs, Yerrs, ...)`** — multi-series 2-D plots with error bars.
- **`make_scatter_plot`, `make_scatter_plot3D`, `make_scatter_plot3D_mult`, `make_scatter_plot3D_multrow`** —
  color-mapped scatter plots in 2-D/3-D, singly or in multi-panel grids (used heavily for
  effectualness-vs-parameters figures).
- **`make_contour_plot_multrow`, `make_contourf_mult`, `make_parameters_plot`** — multi-panel
  contour figures.

## `cbc.py`

**`ParamLatexLabels()`** — dictionary of LaTeX axis labels for standard compact-binary parameters
(masses, spins, distance, angles, …), shared across plotting code.

---

## `analysis_products.py` — analysis-result figures

Classes that read analysis outputs (HDF5 match/mismatch files) and produce publication figures:

- **`class plot_mismatches_sim`** — per-simulation NR error plots: mismatches between CCE
  extraction radii, resolutions (Levs), and extrapolation orders
  (`plot_cce_mismatches_all`, `plot_cce_extrapolation_mismatches`, `plot_cce_max_mismatch`, …).
- **`class plot_mismatches_sims`** — population-level versions across a catalog, to correlate NR
  errors with binary parameters (`hist_cce_mismatch`, …).
- **`class plot_effectualness_vs_totalmass`** — the full suite of effectualness / fitting-factor
  and parameter-bias figures versus total mass and intrinsic parameters, including multi-approximant
  panel grids, contour versions, and recovered-parameter plots
  (`plot_effectualness_vs_totalmass`, `plot_parameterbiases_vs_parameters_multrow`,
  `plot_effectualness_contours_vs_parameters`, …). Reads data written by the NR
  [`EffectualnessAndBias`]({{ site.baseurl }}/docs/api/nr/#analysistypespy) store.

---

## `paraview.py` — ParaView helpers

**`class ParsePVD`** — edit ParaView `.pvd` collection files: `RetrieveUniqueTimeSteps()`,
`DownsampleTimeSteps(factor)`, `RemoveTimeSteps(low, high)`, `WriteFile(filename)`. Useful for
trimming heavy 3-D visualization datasets from NR runs.

## `visualization.py` — notebook embedding

`play_movie(m)` and `embed_video(fname, mimetype)` — display movies (e.g. SpECTRE field
evolutions) inline in Jupyter notebooks.
