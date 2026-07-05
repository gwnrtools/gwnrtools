# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnrtools/gwnrtools/blob/master/LICENSE>
"""Integration tests for gwnr.analysis.filter (waveform generation + matches)"""

import numpy as np
import pytest

pytest.importorskip("pycbc")
pytest.importorskip("lalsimulation")

from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.waveform import get_td_waveform

from gwnr.analysis.filter import calculate_faithfulness, overlap_between_waveforms

F_LOWER = 30.0
SAMPLE_RATE = 2048
DURATION = 16


@pytest.mark.slow
def test_overlap_of_waveform_with_itself_is_one():
    hp, _ = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=40.0,
        mass2=40.0,
        f_lower=F_LOWER,
        delta_t=1.0 / SAMPLE_RATE,
    )
    hp.resize(DURATION * SAMPLE_RATE)
    n = len(hp) // 2 + 1
    psd = aLIGOZeroDetHighPower(n, 1.0 / DURATION, F_LOWER)
    olap = overlap_between_waveforms(hp, hp.copy(), psd, f_lower=F_LOWER)
    assert np.isclose(olap, 1.0, atol=1e-6)


@pytest.mark.slow
def test_faithfulness_same_model_is_one():
    match, _idx = calculate_faithfulness(
        40.0,
        40.0,
        s1z=0.2,
        s2z=0.1,
        signal_approx="IMRPhenomD",
        tmplt_approx="IMRPhenomD",
        aligned_spin_tmplt_only=True,
        f_lower=F_LOWER,
        sample_rate=SAMPLE_RATE,
        signal_duration=DURATION,
        psd_string="aLIGOZeroDetHighPower",
        verbose=False,
    )
    assert np.isclose(float(match), 1.0, atol=1e-4)


@pytest.mark.slow
def test_faithfulness_different_models_below_one():
    match, _idx = calculate_faithfulness(
        40.0,
        40.0,
        signal_approx="IMRPhenomD",
        tmplt_approx="SEOBNRv4_opt",
        aligned_spin_tmplt_only=True,
        f_lower=F_LOWER,
        sample_rate=SAMPLE_RATE,
        signal_duration=DURATION,
        psd_string="aLIGOZeroDetHighPower",
        verbose=False,
    )
    # Different models of the same signal: high but imperfect agreement
    assert 0.9 < float(match) < 1.0
