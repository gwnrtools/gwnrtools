---
title: gwnr.data
layout: default
parent: API Reference
nav_order: 7
permalink: /docs/api/data/
---

# `gwnr.data` — bundled data

Detector noise curves shipped with the package under `gwnr/data/gw_noise_curves/` (ASCII `.txt`
and `.dat` files of PSD/ASD estimates versus frequency for various GW detectors). Source:
[`gwnr/data/data.py`](https://github.com/gwnr/gwnr/blob/master/gwnr/data/data.py).

### `available_gw_noise_curves()`

Returns the list of noise-curve names whose data files are available in the installed package.

### `gw_noise_curve_file(filename)`

Returns the absolute path to the named noise-curve file, suitable for passing to PSD readers
(e.g. `pycbc.psd.from_txt`, or
[`gwnr.analysis.psd.resample_and_extrapolate_psd`]({{ site.baseurl }}/docs/api/analysis/#psdpy--power-spectral-densities)).

```python
from gwnr.data import available_gw_noise_curves, gw_noise_curve_file

curves = available_gw_noise_curves()
path = gw_noise_curve_file(curves[0])
```
