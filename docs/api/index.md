---
title: API Reference
layout: default
nav_order: 5
has_children: true
permalink: /docs/api/
---

# API Reference

The Python package installed by this repository is named **`gwnr`**. It is organized into nine
subpackages:

| Subpackage | Purpose |
|:-----------|:--------|
| [`gwnr.waveform`]({{ site.baseurl }}/docs/api/waveform/) | Waveform generation, conditioning, alignment, hybridization, eccentricity, tidal corrections, parameter conversions |
| [`gwnr.analysis`]({{ site.baseurl }}/docs/api/analysis/) | Matched-filter comparisons (faithfulness, fitting factors), PSDs, GW catalogs, template banks |
| [`gwnr.nr`]({{ site.baseurl }}/docs/api/nr/) | Numerical-relativity waveforms and simulation output (SXS, SpEC, SpECTRE) |
| [`gwnr.stats`]({{ site.baseurl }}/docs/api/stats/) | Bayesian inference setup, samplers, Fisher matrices, distributions |
| [`gwnr.graph`]({{ site.baseurl }}/docs/api/graph/) | Plotting: corner plots, contour/scatter panels, ParaView, notebook video embedding |
| [`gwnr.cosmo`]({{ site.baseurl }}/docs/api/cosmo/) | Cosmological conversions and redshift sampling |
| [`gwnr.workflow`]({{ site.baseurl }}/docs/api/workflow/) | HTCondor DAG / batch-analysis generation |
| [`gwnr.data`]({{ site.baseurl }}/docs/api/data/) | Bundled detector noise curves |
| [`gwnr.utils`]({{ site.baseurl }}/docs/api/utils/) | General-purpose helpers and type conversions |

{: .note }
These pages document the public API as it exists in the source tree; docstring text is summarized
from the code. For exact call signatures and full parameter lists, consult the linked source files
or use `help()` / `?` in an interactive session.

Top-level package contents:

- **`gwnr.get_version_information()`** — returns the version string stamped into `gwnr/.version`
  at build time.
