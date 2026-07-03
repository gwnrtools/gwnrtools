---
title: Tutorials
layout: default
nav_order: 4
permalink: /docs/tutorials/
---

# Tutorials

Executable Jupyter notebooks live in
[`notebooks/`](https://github.com/gwnrtools/gwnrtools/tree/master/notebooks); HTML renderings of
the main ones are published with this site.

## Waveform comparisons

- **[Faithfulness between GW template models]({{ site.baseurl }}/tutorials/ComputeFaithfulness.html)** —
  compute noise-weighted matches between two approximants at fixed physical parameters with
  `gwnr.analysis.calculate_faithfulness`.
- **[Effectualness of GW search template banks]({{ site.baseurl }}/tutorials/ComputeEffectualness.html)** —
  fitting factors via particle-swarm optimization with `calculate_fitting_factor`, and bank
  effectualness studies.

## Bayesian inference

- **[Inference on GW events]({{ site.baseurl }}/tutorials/BayesianInferenceOnGWEvents.html)** —
  set up and run parameter estimation on real (public) events.
- **[Inference on injections]({{ site.baseurl }}/tutorials/BayesianInferenceOnGWInjections.html)** —
  the same for synthetic injections, for model validation and studies of parameter biases.
- **[Making useful corner plots]({{ site.baseurl }}/tutorials/MakingUsefulCornerPlots.html)** —
  visualize posteriors with `gwnr.graph.CornerPlot`: contours, credible levels, colored scatter,
  priors and truth overlays.

## Data access and signal processing

- **[Accessing the GW Open Science catalog]({{ site.baseurl }}/tutorials/AccessGWOpenScienceCatalog.html)** —
  fetch strain data and PSDs for cataloged events with `gwnr.analysis.gw_transient_catalog`.
- **[Q-transforms]({{ site.baseurl }}/tutorials/QTransforms.html)** — time–frequency
  visualization of GW data.

## Waveform models

- **[ESIGMA waveform generation]({{ site.baseurl }}/tutorials/ESIGMA_generation.html)** —
  generating waveforms with the ESIGMA eccentric inspiral-merger-ringdown model.
  (Note: the ESIGMA generation code now lives outside this repository; the calibration
  utilities remain in [`gwnr.stats.enigma_utils`]({{ site.baseurl }}/docs/api/stats/).)

## Additional notebooks

The repository also carries research notebooks that are not rendered here, including ENIGMA
parameter-optimization studies (`OptimizeENIGMAParameters.ipynb`,
`OptimizeENIGMARingdownAttachmentFrequencyFITParameters.ipynb`), GW skymap visualization
(`VisualizeGWSkymaps.ipynb`), and geodesic-kinematics explorations. Browse them in
[`notebooks/`](https://github.com/gwnrtools/gwnrtools/tree/master/notebooks).
