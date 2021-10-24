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
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""General Utilities"""

from __future__ import (absolute_import, print_function)

import sys
from numpy import *
import numpy as np
import math


def add_strings(strlist):
    """Concatenate a list of strings"""
    return "".join(strlist)


def join_list_of_strings(lt):
    """Concatenate a list of strings with space in between"""
    return " ".join(lt)


def find_nearest(a, a0):
    """Element in nd array `a` closest to the scalar value `a0`"""
    idx = np.abs(np.array(a) - a0).argmin()
    return idx, np.array(a).flat[idx]


def approx_equal(A, B, eps=1.e-4):
    return (np.abs(A - B) / (np.abs(A) + np.abs(B)) < eps)


def update_progress(progress):
    print(('\r\r[{0}] {1:.2%}'.format(
        '#' * (int(progress * 100) / 2) + ' ' * (50 - int(progress * 100) / 2),
        progress)))
    if progress == 100:
        print("Done")
    sys.stdout.flush()


def nextpow2(n):
    return 2**int(ceil(log2(n)))


def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n - r)


def area_inside_contour(vs):
    '''Use Green's theorem to compute the area
    enclosed by a given contour.'''
    x = vs[:, 0]
    y = vs[:, 1]
    a = 0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y))
    return np.abs(a)


def zero_pad_beginning(h, steps=1):
    h.data = np.roll(h.data, steps)
    return h


def get_sec(s):
    l = s.split(':')
    return int(l[0]) * 3600 + int(l[1]) * 60 + int(l[2])


def get_time(date, time):
    date_contrib = int(date.split('/')[-1]) * 24 * 60 * 60
    time_contrib = get_sec(time)
    return date_contrib + time_contrib


def trim_trailing_zeros(hp):
    for i in np.arange(len(hp) - 1, 0, -1):
        if hp[i] != 0:
            break
    return hp[:i + 1]


def trim_leading_zeros(hp):
    for i in np.arange(len(hp)):
        if hp[i] != 0:
            break
    return hp[i:]


def format_string(string_template, **string_kwargs):
    import string

    class FormatDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    formatter = string.Formatter()
    mapping = FormatDict(**string_kwargs)
    return formatter.vformat(string_template, (), mapping)


def mkdir(dir_name):
    import subprocess
    try:
        subprocess.call(["mkdir", "-p", dir_name])
    except OSError:
        pass


def rmdir(dir_name):
    import subprocess
    try:
        subprocess.call(["rm", "-rf", dir_name])
    except OSError:
        pass


def insert_min_max_into_array(arr, low, high):
    '''
    Assume an ordered array is passed. Insert min and max and force that
    '''
    if low > arr.max() or high < arr.min():
        return np.array([low, high])
    new_arr = arr
    mask = new_arr > low
    new_arr = np.append(low, new_arr[mask])
    mask = new_arr < high
    new_arr = np.append(new_arr[mask], high)
    return new_arr


def get_uniform_mass_range(m_lower, m_upper, m_sep):
    mlist = [m_lower]
    for m in np.arange(np.ceil(m_lower), np.floor(m_upper), m_sep):
        mlist.append(m)
    mlist.append(m_upper)
    return np.array(mlist)
