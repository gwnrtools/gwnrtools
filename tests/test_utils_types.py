# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnrtools/gwnrtools/blob/master/LICENSE>
"""Tests for gwnr.utils.types"""

import numpy as np
import pytest

pycbc_types = pytest.importorskip("pycbc.types")
from pycbc.types import FrequencySeries, TimeSeries

from gwnr.utils.types import (
    convert_lalREAL8TimeSeries_to_TimeSeries,
    convert_TimeSeries_to_lalREAL8TimeSeries,
    extend_waveform_TimeSeries,
    make_padded_frequency_series,
)


@pytest.fixture
def short_timeseries():
    dt = 1.0 / 64
    data = np.sin(2 * np.pi * 4.0 * np.arange(0, 1, dt))
    return TimeSeries(data, delta_t=dt)


def test_extend_waveform_TimeSeries(short_timeseries):
    N = 256
    out = extend_waveform_TimeSeries(short_timeseries, N)
    assert len(out) == N
    assert out.delta_t == short_timeseries.delta_t
    n0 = len(short_timeseries)
    assert np.allclose(out.data[:n0], short_timeseries.data)
    assert np.all(out.data[n0:] == 0)


def test_extend_waveform_TimeSeries_noop_when_same_length(short_timeseries):
    out = extend_waveform_TimeSeries(short_timeseries, len(short_timeseries))
    assert len(out) == len(short_timeseries)
    assert np.allclose(out.data, short_timeseries.data)


def test_lal_timeseries_roundtrip(short_timeseries):
    lal_ts = convert_TimeSeries_to_lalREAL8TimeSeries(short_timeseries)
    back = convert_lalREAL8TimeSeries_to_TimeSeries(lal_ts)
    assert len(back) == len(short_timeseries)
    assert np.isclose(back.delta_t, short_timeseries.delta_t)
    assert np.allclose(back.data, short_timeseries.data)


def test_make_padded_frequency_series_from_timeseries(short_timeseries):
    filter_N = 256
    out = make_padded_frequency_series(short_timeseries.copy(), filter_N=filter_N)
    assert isinstance(out, FrequencySeries)
    assert len(out) == filter_N // 2 + 1
    expected_delta_f = 1.0 / filter_N / short_timeseries.delta_t
    assert np.isclose(out.delta_f, expected_delta_f)


def test_make_padded_frequency_series_from_frequencyseries():
    fs = FrequencySeries(np.ones(33, dtype=np.complex128), delta_f=1.0)
    out = make_padded_frequency_series(fs, filter_N=128)
    assert isinstance(out, FrequencySeries)
    assert len(out) == 128 // 2 + 1
    assert np.allclose(out.data[:33], 1.0)
    assert np.all(out.data[33:] == 0)


def test_make_padded_frequency_series_rejects_arrays():
    with pytest.raises(IOError):
        make_padded_frequency_series(np.ones(16), filter_N=32)
