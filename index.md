---
title: Home
layout: default
nav_order: 1
permalink: /
---

# gwnr

**A collection of tools for academic research in gravitational-wave astronomy, astrophysics, and numerical relativity.**
{: .fs-6 .fw-300 }

[Get started]({{ site.baseurl }}/docs/getting-started/){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/gwnrtools/gwnr){: .btn .fs-5 .mb-4 .mb-md-0 }

---

`gwnr` (imported as the Python package **`gwnr`**) is a research toolkit built on top of
[PyCBC](https://pycbc.org/), [LALSuite](https://git.ligo.org/lscsoft/lalsuite),
[Bilby](https://lscsoft.docs.ligo.org/bilby/) and the scientific Python stack. It collects the
utilities, analysis pipelines and plotting machinery developed over years of research on
gravitational-wave (GW) source modeling, template-bank construction, matched-filter searches,
Bayesian parameter estimation, and numerical relativity (NR).

## What's inside

| Area | Package | Highlights |
|:-----|:--------|:-----------|
| Waveform tools | [`gwnr.waveform`]({{ site.baseurl }}/docs/api/waveform/) | Waveform generation and conditioning, alignment, hybridization of inspiral and merger–ringdown modes, eccentricity measurement, tidal corrections, parameter conversions |
| Data analysis | [`gwnr.analysis`]({{ site.baseurl }}/docs/api/analysis/) | Faithfulness (match) and fitting-factor calculations, PSD handling, GW transient catalog access, stochastic template-bank construction |
| Numerical relativity | [`gwnr.nr`]({{ site.baseurl }}/docs/api/nr/) | SXS/SpEC waveform handling, strain-mode containers, SpEC and SpECTRE simulation output parsing, NR data in matched-filtering analyses |
| Statistics & inference | [`gwnr.stats`]({{ site.baseurl }}/docs/api/stats/) | Bayesian inference configuration writers (PyCBC Inference, Bilby, LALInference), Fisher-matrix computations, distribution utilities, MCMC sampler helpers |
| Visualization | [`gwnr.graph`]({{ site.baseurl }}/docs/api/graph/) | Corner plots, contour/scatter plotting for effectualness and bias studies, ParaView helpers, movie embedding |
| Cosmology | [`gwnr.cosmo`]({{ site.baseurl }}/docs/api/cosmo/) | Redshift–distance conversions, source/detector-frame mass conversions, merger-rate-weighted redshift sampling |
| Workflow automation | [`gwnr.workflow`]({{ site.baseurl }}/docs/api/workflow/) | HTCondor DAG generation for banksims, faithsims, and batch parameter-estimation campaigns |
| Bundled data | [`gwnr.data`]({{ site.baseurl }}/docs/api/data/) | Detector noise curves (PSDs/ASDs) shipped with the package |
| General utilities | [`gwnr.utils`]({{ site.baseurl }}/docs/api/utils/) | LAL/PyCBC type conversions, array helpers, memory profiling, function timeouts |

In addition, [~30 command-line tools]({{ site.baseurl }}/docs/cli/) are installed for building
template banks, running banksims/faithsims on HTCondor clusters, and orchestrating
parameter-estimation campaigns on GW events and injections.

## Quick example

Compute the faithfulness (noise-weighted match) between two waveform models:

```python
from gwnr.analysis import calculate_faithfulness

match = calculate_faithfulness(
    m1=36.0, m2=29.0,            # component masses (solar masses)
    s1z=0.3, s2z=-0.2,           # aligned spin components
    signal_approx="SEOBNRv4",    # "signal" model
    tmplt_approx="IMRPhenomD",   # "template" model
    f_lower=20.0,
    sample_rate=4096,
    signal_duration=32,
    psd_string="aLIGOZeroDetHighPower",
)
```

## Documentation

- [Installation]({{ site.baseurl }}/docs/installation/)
- [Getting started]({{ site.baseurl }}/docs/getting-started/)
- [Tutorials]({{ site.baseurl }}/docs/tutorials/)
- [API reference]({{ site.baseurl }}/docs/api/)
- [Command-line tools]({{ site.baseurl }}/docs/cli/)
- [Development]({{ site.baseurl }}/docs/development/)

## Citation & license

`gwnr` is developed by [Prayush Kumar](https://github.com/prayush) and collaborators, and is
distributed under the GNU General Public License. If you use it in published work, please cite the
repository: <https://github.com/gwnr/gwnr>.
