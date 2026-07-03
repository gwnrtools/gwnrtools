---
title: Installation
layout: default
nav_order: 2
permalink: /docs/installation/
---

# Installation
{: .no_toc }

1. TOC
{:toc}

## Requirements

`gwnrtools` targets **Python 3** and builds on the LIGO/Virgo software stack. The heavyweight
dependencies are:

- [PyCBC](https://pycbc.org/) — waveform generation, matched filtering, types (`TimeSeries`, `FrequencySeries`)
- [LALSuite](https://git.ligo.org/lscsoft/lalsuite) — LAL waveform approximants and constants
- [Bilby](https://lscsoft.docs.ligo.org/bilby/) — Bayesian inference (used by the `stats` and `workflow` subpackages)
- `lscsoft-glue` — LIGO_LW XML tables and HTCondor DAG utilities

plus the standard scientific stack: `numpy`, `scipy`, `matplotlib`, `pandas`, `h5py`, `astropy`,
`scikit-learn`, `seaborn`, `statsmodels`, `romspline`, `numexpr`, and `pyswarm` (particle-swarm
optimization, used by the fitting-factor machinery).

The full list is in
[`requirements.txt`](https://github.com/gwnrtools/gwnrtools/blob/master/requirements.txt) and
[`setup.py`](https://github.com/gwnrtools/gwnrtools/blob/master/setup.py).

{: .note }
Some optional features shell out to external software that must be installed separately:
HTCondor (workflow DAGs), ParaView (`graph.paraview`), the SpECTRE code
(`nr.spectre`), and SXS/SpEC post-processing tools (`waveform.prepare_waveforms`).

## Installing from source

```bash
git clone https://github.com/gwnrtools/gwnrtools.git
cd gwnrtools
pip install -r requirements.txt
python setup.py install        # or: pip install .
```

For development, use an editable install instead:

```bash
pip install -e .
```

The installed Python package is named **`gwnr`** (the repository and project are called
*gwnrtools*):

```python
import gwnr
print(gwnr.get_version_information())
```

## Conda environments

Because LALSuite and PyCBC ship compiled extensions, the most reliable route is a conda
environment with dependencies from conda-forge:

```bash
conda create -n gwnr python=3.10
conda activate gwnr
conda install -c conda-forge lalsuite pycbc bilby astropy h5py \
    matplotlib pandas scikit-learn scipy seaborn statsmodels
pip install lscsoft-glue romspline numexpr "pyswarm @ git+https://github.com/tisimst/pyswarm@master"
git clone https://github.com/gwnrtools/gwnrtools.git && cd gwnrtools && pip install .
```

## Verifying the installation

```python
import gwnr
import gwnr.waveform as gwf
import gwnr.analysis as gan

# List the detector noise curves shipped with the package
from gwnr.data import available_gw_noise_curves
print(available_gw_noise_curves())
```

Installed command-line tools (e.g. `gwnr_banksim`, `gwnr_faithsim`,
`gwnr_create_bank_workflow`) should be on your `PATH`; see the
[CLI reference]({{ site.baseurl }}/docs/cli/).

## Version information

The package version is stamped at build time from git metadata into `gwnr/.version` (see
`write_version_file()` in `setup.py`); at run time
`gwnr.get_version_information()` reads that file back. Versions follow a
calendar scheme (e.g. `v2021.09.20`).
