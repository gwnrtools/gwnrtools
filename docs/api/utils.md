---
title: gwnr.utils
layout: default
parent: API Reference
nav_order: 8
permalink: /docs/api/utils/
---

# `gwnr.utils` — general utilities
{: .no_toc }

Cross-cutting helpers used throughout the package. Source:
[`gwnr/utils/`](https://github.com/gwnr/gwnr/tree/master/gwnr/utils).

1. TOC
{:toc}

## Package-level

- **`get_unique_hex_tag(N=1, num_digits=10)`** / **`get_sim_hash(...)`** — unique hex identifiers
  for runs/simulations.

## `support.py` — general helpers

- **Strings/formatting**: `add_strings`, `join_list_of_strings`, `format_string` (template
  filling), `get_sec` / `get_time` (time parsing)
- **Arrays**: `find_nearest(a, a0)`, `approx_equal(A, B, eps=1e-4)`, `nextpow2(n)`, `nCr(n, r)`,
  `insert_min_max_into_array(arr, low, high)`, `trim_leading_zeros` / `trim_trailing_zeros`,
  `zero_pad_beginning(h, steps=1)`
- **Geometry**: `area_inside_contour(vs)` — enclosed area via Green's theorem
- **Filesystem**: `mkdir`, `rmdir`
- **Progress/robustness**: `update_progress(progress)` — simple progress bar;
  `call_with_timeout(myfunc, args, kwargs, timeout=5)` — run a function in a separate
  `multiprocessing.Process`, raising if it exceeds `timeout` seconds (used to guard against
  hanging waveform generators)
- `get_uniform_mass_range(m_lower, m_upper, m_sep)`

## `types.py` — LAL/PyCBC type conversions

- **`convert_TimeSeries_to_lalREAL8TimeSeries(h, name=None)`** /
  **`convert_lalREAL8TimeSeries_to_TimeSeries(h)`** — round-trip between PyCBC and LAL time-series
  types.
- **`convert_numpy_to_pycbc_type(arr, out_type, sample_rate, time_length)`** — numpy →
  `TimeSeries`/`FrequencySeries` with length made consistent with `time_length`.
- **`extend_waveform_TimeSeries(wav, filter_N)`** / **`extend_waveform_FrequencySeries(wav, filter_n, force_fit=False)`** —
  zero-extend series to a target filter length.
- **`make_padded_frequency_series(vec, filter_N=None, delta_f=None)`** — convert a time- or
  frequency-series to a `FrequencySeries` at a target `delta_f`, padding time series to avoid
  wraparound (unless `delta_f` forces a shorter duration).
- **`write_series(series, filename)`** — write a series to disk.

## `memory.py` — memory profiling

- **`MemoryUsage(o)`** — recursive memory footprint of a Python object graph (nested dicts,
  lists, tuples, sets).
- **`ShowMemoryUsage(objs=[], prefac=1e-6, prefac_name='Mb')`** — print total sizes for a list of
  objects.
