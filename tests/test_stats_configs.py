# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnrtools/gwnrtools/blob/master/LICENSE>
"""Tests for gwnr.stats config-writing machinery (pycbc_inference / bilby)"""

import os

import pytest


def test_pycbc_inference_configs(tmp_path):
    pytest.importorskip("pycbc")
    from gwnr.stats.pycbc_inference_utils import InferenceConfigs

    run_dir = str(tmp_path)
    configs = InferenceConfigs(run_dir)
    available = configs.available_configs()
    assert len(available) > 0

    # Every advertised config should hand back a usable writer
    name = available[0]
    writer = configs.get_config_writer(name)
    assert writer is not None


def test_bilby_inference_configs(tmp_path):
    pytest.importorskip("bilby")
    from gwnr.stats.bilby_utils import InferenceConfigs

    run_dir = str(tmp_path)
    configs = InferenceConfigs(run_dir)
    assert len(configs.available_configs()) > 0
