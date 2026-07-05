# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnr/gwnr/blob/master/LICENSE>
"""Tests for gwnr.utils package-level helpers and gwnr.utils.memory"""

from gwnr.utils import get_sim_hash, get_unique_hex_tag
from gwnr.utils.memory import MemoryUsage


def test_get_unique_hex_tag_single():
    tag = get_unique_hex_tag()
    assert isinstance(tag, str)
    assert len(tag) == 10
    int(tag, 16)  # must be valid hex


def test_get_unique_hex_tag_multiple():
    tags = get_unique_hex_tag(N=5, num_digits=8)
    assert len(tags) == 5
    for t in tags:
        assert len(t) == 8
        int(t, 16)


def test_get_sim_hash():
    tag = get_sim_hash()
    assert isinstance(tag, str)


def test_memory_usage_positive_and_monotonic():
    small = MemoryUsage({"a": [1, 2, 3]})
    large = MemoryUsage({"a": list(range(10000))})
    assert small > 0
    assert large > small
