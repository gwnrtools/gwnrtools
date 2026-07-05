# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnrtools/gwnrtools/blob/master/LICENSE>
"""Smoke tests: every eagerly-imported gwnr subpackage must be importable.

These guard the compatibility shims for the modern GW software stack
(igwn_ligolw instead of glue.ligolw, scipy>=1.14, numpy>=1.24).
"""

import importlib

import pytest

SUBMODULES = [
    "gwnr",
    "gwnr.analysis",
    "gwnr.analysis.filter",
    "gwnr.analysis.psd",
    "gwnr.analysis.utils",
    "gwnr.cosmo",
    "gwnr.data",
    "gwnr.graph",
    "gwnr.graph.cbc",
    "gwnr.graph.corner",
    "gwnr.graph.misc",
    "gwnr.graph.paraview",
    "gwnr.nr",
    "gwnr.nr.types",
    "gwnr.nr.utils",
    "gwnr.nr.analysis",
    "gwnr.nr.spec.utils",
    "gwnr.stats",
    "gwnr.stats.distribution",
    "gwnr.stats.sampling",
    "gwnr.stats.samplers",
    "gwnr.utils",
    "gwnr.utils.support",
    "gwnr.utils.types",
    "gwnr.waveform",
    "gwnr.waveform.align",
    "gwnr.waveform.condition",
    "gwnr.waveform.hybridize",
    "gwnr.waveform.parameters",
    "gwnr.waveform.utils",
    "gwnr.waveform.waveform",
    "gwnr.workflow",
]


@pytest.mark.parametrize("module", SUBMODULES)
def test_module_imports(module):
    importlib.import_module(module)


def test_version_attribute():
    import gwnr

    assert hasattr(gwnr, "__version__")
    assert hasattr(gwnr, "get_version_information")
