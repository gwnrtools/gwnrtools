# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnrtools/gwnrtools/blob/master/LICENSE>
"""Tests for gwnr.graph"""

import numpy as np
import pytest

from gwnr.graph.cbc import ParamLatexLabels
from gwnr.graph.paraview import ParsePVD

PVD_CONTENT = """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1">
<Collection>
<DataSet timestep="0.5" file="step0.vtu"/>
<DataSet timestep="1.5" file="step1.vtu"/>
<DataSet timestep="2.5" file="step2.vtu"/>
<DataSet timestep="2.5" file="step2b.vtu"/>
</Collection>
</VTKFile>
"""


def test_param_latex_labels():
    labels = ParamLatexLabels()
    assert isinstance(labels, dict)
    for key in ["mass1", "mass2", "mchirp"]:
        assert key in labels
        assert labels[key].startswith("$")


def test_parsepvd_unique_timesteps(tmp_path):
    pvd_file = tmp_path / "test.pvd"
    pvd_file.write_text(PVD_CONTENT)
    pvd = ParsePVD(str(pvd_file))
    tsteps = pvd.RetrieveUniqueTimeSteps()
    assert np.allclose(tsteps, [0.5, 1.5, 2.5])


def test_cornerplot_draw():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("statsmodels")
    from gwnr.graph import CornerPlot

    rng = np.random.RandomState(3)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 1, 500),
            "y": rng.normal(5, 2, 500),
            "z": rng.uniform(0, 1, 500),
        }
    )
    cp = CornerPlot(df, var_type="ccc", var_names=["x", "y", "z"])
    out = cp.draw(params_plot=["x", "y", "z"])
    assert out is not None
