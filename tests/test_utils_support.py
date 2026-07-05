# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnr/gwnr/blob/master/LICENSE>
"""Tests for gwnr.utils.support"""

import numpy as np
import pytest

from gwnr.utils.support import (
    add_strings,
    approx_equal,
    area_inside_contour,
    call_with_timeout,
    find_nearest,
    format_string,
    get_sec,
    get_uniform_mass_range,
    insert_min_max_into_array,
    join_list_of_strings,
    nCr,
    nextpow2,
    trim_leading_zeros,
    trim_trailing_zeros,
)


def test_add_strings():
    assert add_strings(["a", "b", "c"]) == "abc"
    assert add_strings([]) == ""


def test_join_list_of_strings():
    assert join_list_of_strings(["a", "b", "c"]) == "a b c"


def test_find_nearest():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    idx, val = find_nearest(a, 2.2)
    assert idx == 2
    assert val == 2.0


def test_find_nearest_accepts_lists():
    idx, val = find_nearest([10, 20, 30], 19)
    assert idx == 1
    assert val == 20


def test_approx_equal():
    assert approx_equal(1.0, 1.0 + 1e-9)
    assert not approx_equal(1.0, 2.0)
    assert approx_equal(1.0, 1.01, eps=0.1)


def test_nextpow2():
    assert nextpow2(5) == 8
    assert nextpow2(8) == 8
    assert nextpow2(9) == 16
    assert nextpow2(1) == 1


def test_ncr():
    assert nCr(5, 2) == 10
    assert nCr(4, 0) == 1
    assert nCr(6, 6) == 1


def test_area_inside_contour_unit_square():
    vs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    assert np.isclose(area_inside_contour(vs), 1.0)


def test_area_inside_contour_circle():
    theta = np.linspace(0, 2 * np.pi, 1000)
    vs = np.column_stack([np.cos(theta), np.sin(theta)])
    assert np.isclose(area_inside_contour(vs), np.pi, rtol=1e-3)


def test_get_sec():
    assert get_sec("01:02:03") == 3723
    assert get_sec("00:00:00") == 0


def test_format_string_partial_substitution():
    out = format_string("{a}/{b}", a="x")
    assert out == "x/{b}"
    out = format_string("{a}-{b}", a="1", b="2")
    assert out == "1-2"


def test_insert_min_max_into_array():
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    out = insert_min_max_into_array(arr, 1.5, 3.5)
    assert out[0] == 1.5
    assert out[-1] == 3.5
    assert np.all(np.diff(out) > 0)
    # Disjoint range collapses to its own bounds
    out = insert_min_max_into_array(arr, 10.0, 20.0)
    assert np.allclose(out, [10.0, 20.0])


def test_get_uniform_mass_range():
    out = get_uniform_mass_range(3.3, 10.7, 1.0)
    assert out[0] == 3.3
    assert out[-1] == 10.7
    assert np.all(np.diff(out) > 0)


def test_trim_zeros():
    arr = np.array([0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 0.0])
    lead = trim_leading_zeros(arr)
    assert lead[0] == 1.0
    trail = trim_trailing_zeros(arr)
    assert trail[-1] == 3.0


def _quick(x):
    return x + 1


def test_call_with_timeout_returns_result():
    assert call_with_timeout(_quick, args=(41,), timeout=10) == 42


def _slow():
    import time

    time.sleep(30)
    return "never"


def test_call_with_timeout_raises_on_timeout():
    with pytest.raises(Exception, match="Timeout"):
        call_with_timeout(_slow, timeout=1)
