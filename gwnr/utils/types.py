# Copyright (C) 2014 Prayush Kumar
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
"""
Functions related to LAL and PyCBC datatypes
"""

# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
from __future__ import print_function

from numpy import *
import numpy as np

from pycbc.types import TimeSeries, FrequencySeries, complex_same_precision_as
from pycbc.pnutils import nearest_larger_binary_number


def convert_TimeSeries_to_lalREAL8TimeSeries(h, name=None):
    tmp = lal.CreateREAL8Sequence(len(h))
    tmp.data = np.array(h.data)
    hnew = lal.REAL8TimeSeries()
    hnew.data = tmp
    hnew.deltaT = h.delta_t
    hnew.epoch = h._epoch
    if name is not None:
        hnew.name = name
    return hnew


def convert_lalREAL8TimeSeries_to_TimeSeries(h):
    return TimeSeries(h.data.data,
                      delta_t=h.deltaT,
                      copy=True,
                      epoch=h.epoch,
                      dtype=h.data.data.dtype)


def extend_waveform_TimeSeries(wav, filter_N):
    # {{{
    if len(wav) != filter_N:
        try:
            _wav = TimeSeries(np.zeros(filter_N),
                              delta_t=wav.delta_t,
                              dtype=real_same_precision_as(wav),
                              epoch=wav._epoch)
        except MemoryError as merr:
            print(("Do you really want to allocate %d doubles?" % filter_N))
            raise MemoryError(merr)
        _wav[:len(wav)] = wav
    else:
        _wav = wav
    return _wav
    # }}}


def extend_waveform_FrequencySeries(wav, filter_n, force_fit=False):
    # {{{
    if len(wav) < filter_n:
        _wav = FrequencySeries(np.zeros(filter_n),
                               delta_f=wav.delta_f,
                               dtype=complex_same_precision_as(wav),
                               epoch=wav._epoch)
        _wav[:len(wav)] = wav
    elif len(wav) == filter_n:
        _wav = wav
    elif force_fit:
        print(("WARNING: Ignoring high-frequency content above %.3f Hz" %
               (wav.delta_f * filter_n)))
        _wav = FrequencySeries(np.zeros(filter_n),
                               delta_f=wav.delta_f,
                               dtype=complex_same_precision_as(wav),
                               epoch=wav._epoch)
        _wav[:len(_wav)] = wav[:len(_wav)]
    else:
        raise IOError(
            "Passed FrequencySeries has length %d, cannot shrink to %d" %
            (len(wav), filter_n))
    return _wav
    # }}}


def convert_numpy_to_pycbc_type(arr,
                                out_type,
                                sample_rate=None,
                                time_length=None):
    """
Convert numpy.array to pycbc.types, TimeSeries and FrequencySeries.
Output array is extended to length consistent with time_length passed

ALL ARGUMENTS ARE NECESSARY.
    """
    delta_t = 1. / sample_rate
    delta_f = 1. / time_length
    N = sample_rate * time_length
    n = N / 2 + 1
    if out_type == TimeSeries:
        out_arr = TimeSeries(arr, delta_t=delta_t)
        out_arr = extend_waveform_TimeSeries(out_arr, N)
    elif out_type == FrequencySeries:
        out_arr = FrequencySeries(arr, delta_f=delta_f)
        out_arr = extend_waveform_FrequencySeries(out_arr, n)
    return out_arr


def make_padded_frequency_series(vec, filter_N=None, delta_f=None):
    """
Convert vec (TimeSeries or FrequencySeries) to a FrequencySeries. For
a TimeSeries input, first it is resized to filter_N and  is  If
filter_N and/or delta_f are given, the output will take those values. If
    not told otherwise the code will attempt to pad a timeseries first such that
    the waveform will not wraparound. However, if delta_f is specified to be
    shorter than the waveform length then wraparound *will* be allowed.
    """
    # {{{
    if filter_N is None:
        filter_N = nearest_larger_binary_number(len(vec))
    filter_n = filter_N / 2 + 1

    if isinstance(vec, FrequencySeries):
        vectilde = FrequencySeries(zeros(filter_n),
                                   delta_f=vec.get_delta_f(),
                                   dtype=complex_same_precision_as(vec))
        cplen = min(len(vec), len(vectilde))
        vectilde[:cplen] = vec[:cplen]
        if delta_f is not None:
            delta_f_ratio = max(1, int(ceil(vectilde.get_delta_f() / delta_f)))
            vectilde = vectilde[:len(vectilde):delta_f_ratio]
    elif isinstance(vec, TimeSeries):
        delta_f_from_filter_N = 1. / filter_N / vec.get_delta_t()
        vec.resize(filter_N)
        v_tilde = vec.to_frequencyseries()
        vectilde = FrequencySeries(v_tilde[:],
                                   delta_f=delta_f_from_filter_N,
                                   dtype=complex_same_precision_as(vec),
                                   copy=True)
    else:
        vectilde = None
        raise IOError("Input is neither a TimeSeries nor a FrequencySeries")

    return vectilde
    # }}}
