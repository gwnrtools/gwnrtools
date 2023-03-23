# Copyright (C) 2023 Kartikey Sharma
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
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""Master function to hybridise any complex timeseries using the 'frequency' as a user input, 
specifically to be used for gravitational waveform hybridisation, fine-tuned for a single mode.
"""

import numpy as np
import scipy.optimize
from scipy.integrate import cumulative_trapezoid


def find_last_value_location_in_series(frq_timeseries, frq_desired):

    if frq_desired < np.min(frq_timeseries):
        raise Exception(
            'Desired frequency out of bounds, lower than min frequency')

    if frq_desired > np.max(frq_timeseries):
        raise Exception(
            'Desired frequency out of bounds, higher than max frequency')
    ''' 
        We reverse the array and traverse it to find the location where the i_th value is more than
        the desired value while the i+1_th value is less, hence locating the desired value somewhere
        between those two points. We then choose the value closer to the value desired (among i and i+1) 
        and call it the location of the desired value. 
    '''

    reversed_freq_timeseries = frq_timeseries[::-1]
    final_idx = len(reversed_freq_timeseries) - 1

    for idx, f_value in enumerate(reversed_freq_timeseries):
        if idx != len(reversed_freq_timeseries) - 1:
            if reversed_freq_timeseries[
                    idx] >= frq_desired and reversed_freq_timeseries[
                        idx + 1] <= frq_desired:
                fr1 = reversed_freq_timeseries[idx]
                fr2 = reversed_freq_timeseries[idx + 1]

                if abs(frq_desired - fr1) <= abs(frq_desired - fr2):
                    final_idx = idx
                else:
                    final_idx = idx + 1
                break
    return len(frq_timeseries) - 1 - final_idx


def mismatch_discrete(w1, w2, sample_indices_insp, sample_indices_mr):
    w1_d = w1[sample_indices_insp]
    w2_d = w2[sample_indices_mr]  # can't give the same comb to w2
    w2sq = np.square(np.abs(w2_d))
    # w1sq = np.square(np.abs(w2_d)) # another normalising factor can be (w1sq + w2sq) / 2
    diff = np.abs(w1_d - w2_d)
    diffsq = np.square(diff)
    mm = 0.5 * (np.sum(diffsq) / np.sum(w2sq))
    return mm


def align_in_phase(inspiral, merger_ringdown, sample_indices_insp,
                   sample_indices_mr, t1_index_insp, t2_index_insp,
                   t1_index_mr, t2_index_mr):
    # Function alignes the two waveforms using the phase, optimised over the attachment region

    def optfn_ph(phaseshift_correction):
        phase_corrected_insp = inspiral * np.exp(1j * phaseshift_correction)
        m = mismatch_discrete(
            phase_corrected_insp[t1_index_insp:t2_index_insp + 1],
            merger_ringdown[t1_index_mr:t2_index_mr + 1], sample_indices_insp,
            sample_indices_mr)
        return m

    phase_optimizer = scipy.optimize.minimize(optfn_ph, 0)
    phaseshift_required_for_alignment = phase_optimizer.x

    inspiral_aligned = inspiral * np.exp(
        1j * phaseshift_required_for_alignment)

    return inspiral_aligned, phaseshift_required_for_alignment


def blend_series(x1, x2, t1_index_insp, t2_index_insp, t1_index_mr,
                 t2_index_mr):

    assert t1_index_mr - t2_index_mr == t1_index_insp - t2_index_insp, "Inconsistent indices passed to blending function"

    # blending fn is an array
    blfn_var = np.arange(t1_index_insp, t2_index_insp)
    tau = np.square((np.sin((np.pi / 2) * (blfn_var - t1_index_insp) /
                            (t2_index_insp - t1_index_insp))))

    x_hyb = (1 - tau) * x1[t1_index_insp:t2_index_insp] + tau * x2[
        t1_index_mr:t2_index_mr]
    return x_hyb


''' Computing waveform attributes: amplitude, phase, frequency (no interpolation) '''


def compute_amplitude(waveform):
    amplitude = np.abs(waveform)
    return amplitude


def compute_phase(waveform):
    phase = np.unwrap(-np.angle(waveform))
    return phase


def compute_frequency(phase, dt):
    frequency = np.gradient(phase, dt) / (2 * np.pi)
    return frequency


def perform_hybridisation(inspiral,
                          merger_ringdown,
                          dt,
                          frq_attach,
                          frq_width,
                          no_sp=4):

    phase_insp = compute_phase(inspiral)
    frq_insp = compute_frequency(phase_insp, dt)

    phase_mr = compute_phase(merger_ringdown)
    frq_mr = compute_frequency(phase_mr, dt)
    amp_mr = compute_amplitude(merger_ringdown)
    ''' first we need to find the attachment region, based on the frequency '''

    t1_index_mr = find_last_value_location_in_series(
        frq_mr, frq_attach - frq_width / 2)

    t2_index_mr = find_last_value_location_in_series(
        frq_mr, frq_attach + frq_width / 2)
    ''' 
    For eccentric inspiral, there will be multiple instances of the 
    same frequency. Pick the one having the highest index value (i.e. 
    the one at the rightmost occurance in time) 

    '''
    t2_index_insp = find_last_value_location_in_series(
        frq_insp, frq_attach + frq_width / 2)

    # another way to define t2_index_mr is through number of points in the inspiral window
    t1_index_insp = t2_index_insp - (t2_index_mr - t1_index_mr)
    ''' 
        Theoretically, we NEED a timeshift to align the waveforms in frequency. 
        Instead of shifting one of the two waveforms for alignment, we are defining
        the time such that the frequencies are pre-aligned to the best of the 
        discrete interval errors. That is: 
            deltaT (timeshift) = t1_index_insp - t1_index_mr
        The mathematical way is to optimise the difference in frequencies over the matching 
        region and using that to determine deltaT, hence arriving at t1_index_mr. 
    '''

    sample_indices_insp = np.linspace(t1_index_insp, t2_index_insp,
                                      no_sp).astype(int) - t1_index_insp
    sample_indices_mr = sample_indices_insp  # since the attachment region in both has the same length
    ''' alignment using corrective phase addition '''

    inspiral_aligned, phasecorr = align_in_phase(inspiral, merger_ringdown,
                                                 sample_indices_insp,
                                                 sample_indices_mr,
                                                 t1_index_insp, t2_index_insp,
                                                 t1_index_mr, t2_index_mr)

    amp_insp_aligned = compute_amplitude(inspiral_aligned)
    phase_insp_aligned = compute_phase(inspiral_aligned)
    '''
        It would be same as frq_mr as the corrected phase factor will be canceled in the derivative, 
        defining frq_insp_aligned just for consistency 
    '''
    frq_insp_aligned = frq_insp
    ''' Performing attachment using the blending function '''

    amp_hyb_window = blend_series(amp_insp_aligned, amp_mr, t1_index_insp,
                                  t2_index_insp, t1_index_mr, t2_index_mr)
    frq_hyb_window = blend_series(frq_insp_aligned, frq_mr, t1_index_insp,
                                  t2_index_insp, t1_index_mr, t2_index_mr)
    ''' Integrating frq_hyb to obtain phase_hyb and removing discontinuities, 
        compiling amp_hyb and phase_hyb to obtain the hybrid waveform.     '''

    phase_hyb_window = (2 * np.pi) * cumulative_trapezoid(
        frq_hyb_window, dx=dt, initial=0
    )  # Length of this will be one point shorter than frq_hyb_window
    ''' Right now the phase is integrated only inside the hybrid window, 
    need to add constants to preserve phase continuity and compile full IMR phase '''
    def remove_phase_discontinuity(phase_insp, phase_hyb_window,
                                   phase_mr_aligned):
        delta1 = phase_insp_aligned[t1_index_insp] - phase_hyb_window[0]
        phase_hyb_1 = np.append(phase_insp_aligned[:t1_index_insp],
                                phase_hyb_window + delta1)
        delta2 = phase_hyb_1[t2_index_insp - 1] - phase_mr[t2_index_mr - 1]
        phase_hyb_2 = np.append(phase_hyb_1[:t2_index_insp - 1],
                                phase_mr[t2_index_mr - 1:] + delta2)
        return phase_hyb_2

    phase_hyb_full = remove_phase_discontinuity(phase_insp_aligned,
                                                phase_hyb_window, phase_mr)

    amp_hyb_full = np.append(
        np.concatenate([amp_insp_aligned[:t1_index_insp],
                        amp_hyb_window])[:t2_index_insp - 1],
        amp_mr[t2_index_mr - 1:])

    waveform_hyb = amp_hyb_full * np.exp(-1j * phase_hyb_full)

    return (waveform_hyb, t1_index_insp, t1_index_mr, t2_index_insp,
            t2_index_mr, frq_insp, frq_mr, frq_mr_aligned, inspiral_aligned,
            sample_indices_insp, sample_indices_mr, amp_insp, amp_insp_aligned,
            amp_hyb_window, amp_hyb_full, phase_insp, phase_hyb_window,
            phase_hyb_full)


''' END OF CODE '''
