---
title: Getting started
layout: default
nav_order: 3
permalink: /docs/getting-started/
---

# Getting started
{: .no_toc }

A tour of the most commonly used functionality. All examples assume the package is
[installed]({{ site.baseurl }}/docs/installation/) along with PyCBC and LALSuite.

1. TOC
{:toc}

## Faithfulness between two waveform models

*Faithfulness* is the noise-weighted overlap between two waveforms with **identical physical
parameters**, maximized only over time and phase. It quantifies how well one model reproduces
another (values near 1 are good). This is the core of
[`gwnr.analysis.calculate_faithfulness`]({{ site.baseurl }}/docs/api/analysis/):

```python
from gwnr.analysis import calculate_faithfulness

match = calculate_faithfulness(
    m1=36.0, m2=29.0,
    s1z=0.3, s2z=-0.2,
    signal_approx="SEOBNRv4",
    tmplt_approx="IMRPhenomD",
    aligned_spin_tmplt_only=True,
    f_lower=20.0,
    sample_rate=4096,
    signal_duration=32,
    psd_string="aLIGOZeroDetHighPower",
)
```

See the [ComputeFaithfulness tutorial]({{ site.baseurl }}/tutorials/ComputeFaithfulness.html) for a
worked notebook.

## Fitting factor (effectualness) of a template family

The *fitting factor* additionally maximizes the match over the **template's physical parameters**,
measuring the best a template family can do at recovering a given signal. `gwnr` uses
particle-swarm optimization (PSO) for this maximization:

```python
from gwnr.analysis import calculate_fitting_factor

result = calculate_fitting_factor(
    m1=36.0, m2=29.0,
    tmplt_approx="IMRPhenomD",
    signal_approx="SEOBNRv4",
    vary_masses_and_aligned_spin_only=True,
    chirp_mass_window=0.2,       # fractional search window around signal chirp mass
    effective_spin_window=0.75,  # search window on effective spin
    f_lower=20.0,
    sample_rate=4096,
    pso_swarm_size=100,
    psd_string="aLIGOZeroDetHighPower",
)
```

The [ComputeEffectualness tutorial]({{ site.baseurl }}/tutorials/ComputeEffectualness.html) shows
this end-to-end, including template banks.

## Aligning and comparing waveforms

[`gwnr.waveform.align`]({{ site.baseurl }}/docs/api/waveform/) provides several strategies for
aligning two sets of GW polarizations:

```python
from gwnr.waveform.align import (
    align_waveforms_amplitude_peak,   # align at the amplitude peak
    align_waveforms_at_frequency,     # align where the GW frequency crosses f_align
    align_waveforms_optimally,        # maximize the noise-weighted inner product
)

hp1, hc1, hp2, hc2 = align_waveforms_optimally(
    hp1, hc1, hp2, hc2,
    psd="aLIGOZeroDetHighPower",
    low_frequency_cutoff=20.0,
)
```

## Hybridizing inspiral and merger–ringdown modes

[`gwnr.waveform.hybridize`]({{ site.baseurl }}/docs/api/waveform/) stitches post-Newtonian /
inspiral-only mode data onto merger–ringdown (e.g. NR) modes at a chosen attachment frequency:

```python
from gwnr.waveform.hybridize import hybridize_modes

hybrid_modes, retval = hybridize_modes(
    inspiral_modes,            # dict indexed by (l, m)
    merger_ringdown_modes,     # dict indexed by (l, m)
    inspiral_orbital_frequency,
    frq_attach=0.025,          # attachment frequency
    frq_width=10.0,
    modes_to_hybridize=[(2, 2), (3, 3), (4, 4)],
)
```

## Working with NR (SXS) waveforms

[`gwnr.nr`]({{ site.baseurl }}/docs/api/nr/) reads SXS-format HDF5 waveforms and rescales them to
physical mass/distance, tapers them, and produces detector-frame polarizations:

```python
from gwnr.waveform.nr_waveform_sxs import get_hplus_hcross_from_sxs

hp, hc = get_hplus_hcross_from_sxs(
    "rhOverM_Asymptotic_GeometricUnits_CoM.h5",
    template_params,   # object carrying masses, spins, inclination, etc.
    delta_t=1.0 / 4096,
    modeLmin=2, modeLmax=8,
    taper=True,
)
```

The `gwnr.nr.types.nr_strain` class provides a higher-level container with mode amplitudes,
phases and frequencies, mass/distance rescaling, energy and angular-momentum fluxes, Bondi news
and Psi4 modes.

## Bayesian inference campaigns

[`gwnr.stats`]({{ site.baseurl }}/docs/api/stats/) and
[`gwnr.workflow`]({{ site.baseurl }}/docs/api/workflow/) automate setting up parameter-estimation
runs with **PyCBC Inference** or **Bilby**, on both real events and injections. In practice you
drive these through the command-line tools:

```bash
# Write PyCBC Inference configuration files for GW150914
gwnr_write_pycbc_inference_configs --write-data-config-for-event GW150914 ...

# Set up a batch of inference runs on public events
gwnr_create_public_events_pycbc_inference_workflow ...

# Or the Bilby equivalents
gwnr_write_bilby_configs ...
gwnr_create_injections_bilby_workflow ...
```

See the tutorials on
[inference for GW events]({{ site.baseurl }}/tutorials/BayesianInferenceOnGWEvents.html) and
[inference on injections]({{ site.baseurl }}/tutorials/BayesianInferenceOnGWInjections.html).

## Corner plots of posteriors

[`gwnr.graph`]({{ site.baseurl }}/docs/api/graph/) has a flexible corner-plot class supporting
scatter panels, percentile contours, per-point coloring, and prior overlays:

```python
import pandas as pd
from gwnr.graph import CornerPlot

samples = pd.read_csv("posterior_samples.dat", delim_whitespace=True)
cp = CornerPlot(samples)
fig, axes = cp.draw(
    params_plot=["mass1", "mass2", "chi_eff", "distance"],
    plot_type="contour",
    contour_levels=[68.27, 90.0, 95.45],
)
```

The [MakingUsefulCornerPlots tutorial]({{ site.baseurl }}/tutorials/MakingUsefulCornerPlots.html)
walks through the options.

## Detector noise curves

The package ships ASCII noise curves for various detectors under `gwnr/data/gw_noise_curves/`:

```python
from gwnr.data import available_gw_noise_curves, gw_noise_curve_file

print(available_gw_noise_curves())
psd_path = gw_noise_curve_file("<curve-name>.txt")
```

`gwnr.analysis.psd.resample_and_extrapolate_psd` regularizes measured PSD data onto a uniform
frequency grid (with extrapolation toward f = 0) so it can be used in PyCBC filtering.

## Where to go next

- [Tutorials]({{ site.baseurl }}/docs/tutorials/) — executable notebooks rendered as HTML
- [API reference]({{ site.baseurl }}/docs/api/) — subpackage-by-subpackage documentation
- [Command-line tools]({{ site.baseurl }}/docs/cli/) — installed scripts and workflow generators
