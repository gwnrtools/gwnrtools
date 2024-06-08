# Copyright (C) 2020 Prayush Kumar
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
"""Fitting functions specific to ESIGMA"""

from __future__ import absolute_import

import logging
import numpy as np
import time
import gwnr
import lal
import lalsimulation as ls
import pycbc.types as pt
from .utils import f_ISCO_spin

ECCENTRICITY_LEVEL_ISCO_WARNING = 0.02
ECCENTRICITY_LEVEL_ISCO_ERROR = 0.1


def find_x_for_y(x, y, y0):
    return x[abs(y - y0).argmin()]


def eccentricity_at_extremum_frequency(
    mass1,
    mass2,
    spin1z,
    spin2z,
    e0,
    l0,
    f_lower,
    sample_rate,
    f_extremum,
    extremum="periastron",
    show_figures=False,
    verbose=False,
):
    """ """
    if extremum.lower() not in ["periastron", "apastron"]:
        raise IOError("Allowed values for extremum: periastron, apastron")

    itime = time.perf_counter()
    retval = ls.SimInspiralENIGMADynamics(
        mass1, mass2, spin1z, spin2z, e0, f_lower, l0, 1e-12, sample_rate, False
    )
    t, x, e, l, phi, phidot, r, rdot = retval[:8]
    t.data.data *= lal.MTSUN_SI

    if verbose:
        print(f"Orbital evolution took: {time.perf_counter() - itime} seconds")

    omega = pt.TimeSeries(phidot.data.data, delta_t=t.data.data[1] - t.data.data[0])
    if extremum == "periastron":
        extremum_frequencies_times, extremum_frequencies = gwnr.waveform.get_peak_freqs(
            omega
        )
    else:
        extremum_frequencies_times, extremum_frequencies = gwnr.waveform.get_peak_freqs(
            -1 * omega
        )
        extremum_frequencies *= -1

    piMf_extremum = f_extremum * lal.PI * (mass1 + mass2) * lal.MTSUN_SI
    time_at_sensitive_freq = find_x_for_y(
        extremum_frequencies_times, extremum_frequencies, piMf_extremum
    )
    idx_e0 = abs(t.data.data - time_at_sensitive_freq).argmin()
    e0 = e.data.data[idx_e0]

    if show_figures:
        import matplotlib.pyplot as plt

        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
        ax2 = ax1
        ax1.plot(
            extremum_frequencies_times,
            extremum_frequencies,
            "o",
            markersize=1,
            label="extrema",
        )
        ax1.plot(t.data.data, x.data.data**1.5, "--", label="x ** {3/2}")
        ax1.plot(t.data.data, phidot.data.data, lw=0.5, label="omega")

        ax1.axhline(piMf_extremum, c="c", lw=1)
        ax1.axvline(time_at_sensitive_freq, c="r", lw=1)

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Orbital angular velocity")

        ax = plt.twinx(ax2)
        ax.plot(t.data.data, e.data.data, label="eccentricity", lw=1)
        ax.axhline(e0, c="k", lw=1)
        ax.set_xlim(ax1.get_xlim())

        ax1.legend(loc="center left")
        ax.legend(loc="upper center")

        return e0, fig

    return e0


def eccentricity_at_reference_frequency(
    mass1,
    mass2,
    spin1z,
    spin2z,
    e0,
    l0,
    f_lower,
    sample_rate,
    f_reference,
    show_figures=False,
    verbose=False,
):
    """ """
    itime = time.perf_counter()
    retval = ls.SimInspiralENIGMADynamics(
        mass1, mass2, spin1z, spin2z, e0, f_lower, l0, 1e-12, sample_rate, False
    )
    t, x, e, l, phi, phidot, r, rdot = retval[:8]
    t.data.data *= lal.MTSUN_SI

    if verbose:
        print(f"Orbital evolution took: {time.perf_counter() - itime} seconds")

    x_reference = (np.pi * (mass1 + mass2) * lal.MTSUN_SI * f_reference) ** (2.0 / 3.0)

    idx_e0 = abs(x.data.data - x_reference).argmin()
    e0 = e.data.data[idx_e0]

    if show_figures:
        import matplotlib.pyplot as plt

        fig, (ax) = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
        ax.plot(t.data.data, x.data.data**1.5, "--", label="x ** {3/2}")
        ax.plot(t.data.data, phidot.data.data, lw=0.5, label="omega")
        ax.axhline(x_reference**1.5, color="c", lw=1)
        ax.axvline(t.data.data[idx_e0], color="r", lw=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Orbital angular velocity")

        ax2 = plt.twinx(ax)
        ax2.plot(t.data.data, e.data.data, label="eccentricity", lw=1)
        ax2.axhline(e0, c="k", lw=1)
        ax2.set_xlim(ax.get_xlim())

        ax.legend(loc="center left")
        ax2.legend(loc="upper center")
        return e0, fig

    return e0


def get_inspiral_esigma_modes(
    mass1,
    mass2,
    f_lower,
    delta_t,
    spin1z=0.0,
    spin2z=0.0,
    eccentricity=0.0,
    mean_anomaly=0.0,
    distance=1.0,
    f_ref=None,
    modes_to_use=[(2, 2), (3, 3), (4, 4)],
    include_conjugate_modes=True,
    return_orbital_params=False,
    return_pycbc_timeseries=True,
    verbose=False,
):
    """
    Returns inspiral ESIGMA GW modes

    Parameters:
    -----------
        mass1, mass2            -- Binary's component masses (in solar masses)
        f_lower                 -- Starting frequency of the waveform (in Hz)
        f_ref                   -- Reference frequency at which to define the waveform
                                   parameters.
                                   We require f_ref <= f_lower. f_ref = f_lower by default.
        delta_t                 -- Waveform's time grid-spacing (in s)
        spin1z, spin2z          -- z-components of component dimensionless spins (lies in [0,1))
        eccentricity            -- Initial eccentricity
        mean_anomaly            -- Mean anomaly of the periastron (in rad)
        distance                -- Luminosity distance to the binary (in Mpc)
        modes_to_use            -- GW modes to use. List of tuples (l, |m|)
        include_conjugate_modes -- If True, (l, -|m|) modes are included as well
        return_orbital_params   -- If True, returns the orbital evolution of all the orbital elements.
                                   Can also be a list of orbital variable names to return only those
                                   specific variables. Available orbital variables are:
                                   ['x', 'e', 'l', 'phi', 'phidot', 'r', 'rdot']
        return_pycbc_timeseries -- If True, returns data in the form of PyCBC timeseries.
                                   True by default.
        verbose                 -- Verbosity flag

    Returns:
    --------
        t                 -- Time grid (in seconds).
                             Returned only if return_pycbc_timeseries=False
        orbital_var_dict  -- Dictionary of evolution of orbital elements.
                             Returned only if "return_orbital_params" is specified
        modes             -- Dictionary of GW modes
    """

    if return_orbital_params:
        orbital_var_names = ["x", "e", "l", "phi", "phidot", "r", "rdot"]
        if return_orbital_params != True:
            for name in return_orbital_params:
                if name not in orbital_var_names:
                    raise Exception(
                        f"{name} is not a valid orbital variable name. Available orbital variable names are: {orbital_var_names}."
                    )

    if f_ref is None:
        f_ref = f_lower
    elif f_ref > f_lower:
        raise NotImplementedError("We do not support f_ref > f_lower yet.")

    distance *= 1.0e6 * lal.PC_SI  # Mpc to SI conversion

    # Calculating the orbital variables
    itime = time.perf_counter()
    retval = ls.SimInspiralENIGMADynamics(
        mass1,
        mass2,
        spin1z,
        spin2z,
        eccentricity,
        f_ref,
        mean_anomaly,
        1e-12,
        1 / delta_t,
        False,
    )

    if f_ref < f_lower:
        ref_idx = np.searchsorted(
            (retval[1].data.data ** 1.5) / ((mass1 + mass2) * lal.MTSUN_SI * np.pi),
            f_lower,
        )
        new_len = len(retval[0].data.data) - ref_idx
        for ii in range(8):
            lal.ResizeREAL8TimeSeries(retval[ii], ref_idx, new_len)

    t, x, e, l, phi, phidot, r, rdot = retval[:8]
    t.data.data *= (
        mass1 + mass2
    ) * lal.MTSUN_SI  # Time from geometrized units to seconds

    if verbose:
        print(f"Orbital evolution took: {time.perf_counter() - itime} seconds")

    # Include conjugate modes in the mode list
    if include_conjugate_modes:
        for el, em in modes_to_use:
            if (el, -em) not in modes_to_use:
                modes_to_use.append((el, -em))

    itime = time.perf_counter()
    modes = {}
    for el, em in modes_to_use:
        modes[(el, em)] = ls.SimInspiralENIGMAModeFromDynamics(
            el,
            em,
            t.data,
            x.data,
            phi.data,
            phidot.data,
            r.data,
            rdot.data,
            mass1,
            mass2,
            spin1z,
            spin2z,
            distance,
        )

    if return_pycbc_timeseries:
        modes = {
            k: pt.TimeSeries(modes[k].data.data, delta_t=delta_t, epoch=0)
            for k in modes
        }
    else:
        modes = {k: np.asarray(modes[k].data.data) for k in modes}

    if verbose:
        print(f"Modes generation took: {time.perf_counter() - itime} seconds")

    if return_orbital_params:
        orbital_var_dict = {}
        if return_orbital_params == True:
            return_orbital_params = orbital_var_names

        if return_pycbc_timeseries:
            for name in return_orbital_params:
                exec(
                    f"orbital_var_dict['{name}'] = pt.TimeSeries({name}.data.data, delta_t=delta_t, epoch=0)"
                )
            return orbital_var_dict, modes

        for name in return_orbital_params:
            exec(f"orbital_var_dict['{name}'] = {name}.data.data")
        return (t.data.data - t.data.data[-1]), orbital_var_dict, modes

    if return_pycbc_timeseries:
        return modes
    return (t.data.data - t.data.data[-1]), modes


def get_inspiral_esigma_waveform(
    mass1,
    mass2,
    f_lower,
    delta_t,
    spin1z=0.0,
    spin2z=0.0,
    eccentricity=0.0,
    mean_anomaly=0.0,
    inclination=0.0,
    coa_phase=0.0,
    distance=1.0,
    f_ref=None,
    modes_to_use=[(2, 2), (3, 3), (4, 4)],
    return_orbital_params=False,
    return_pycbc_timeseries=True,
    verbose=False,
    **kwargs,
):
    """
    Returns inspiral ESIGMA GW polarizations

    Parameters:
    -----------
        mass1, mass2            -- Binary's component masses (in solar masses)
        f_lower                 -- Starting frequency of the waveform (in Hz)
        f_ref                   -- Reference frequency at which to define the waveform
                                   parameters.
                                   We require f_ref <= f_lower. f_ref = f_lower by default.
        delta_t                 -- Waveform's time grid-spacing (in s)
        spin1z, spin2z          -- z-components of component dimensionless spins (lies in [0,1))
        eccentricity            -- Initial eccentricity
        mean_anomaly            -- Mean anomaly of the periastron (in rad)
        inclination             -- Inclination (in rad), defined as the angle between the orbital
                                   angular momentum L and the line-of-sight
        coa_phase               -- Coalesence phase of the binary (in rad)
        distance                -- Luminosity distance to the binary (in Mpc)
        modes_to_use            -- GW modes to use. List of tuples (l, |m|)
        return_orbital_params   -- If True, returns the orbital evolution of all the orbital elements (in
                                   geometrized units). Can also be a list of orbital variable names to return
                                   only those specific variables. Available orbital variables names are:
                                   ['x', 'e', 'l', 'phi', 'phidot', 'r', 'rdot']
        return_pycbc_timeseries -- If True, returns data in the form of PyCBC timeseries.
                                   True by default
        verbose                 -- Verbosity level. Available values are: 0, 1, 2

    Returns:
    --------
        t                 -- Time grid (in seconds).
                             Returned only if return_pycbc_timeseries=False
        orbital_var_dict  -- Dictionary of evolution of orbital elements.
                             Returned only if "return_orbital_params" is specified
        hp, hc            -- Plus and cross GW polarizations
    """

    retval = get_inspiral_esigma_modes(
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        eccentricity=eccentricity,
        mean_anomaly=mean_anomaly,
        distance=distance,
        f_ref=f_ref,
        f_lower=f_lower,
        delta_t=delta_t,
        modes_to_use=modes_to_use,
        include_conjugate_modes=True,  # Always include conjugate modes while generating polarizations
        return_orbital_params=return_orbital_params,
        verbose=verbose,
        return_pycbc_timeseries=False,
    )

    if return_orbital_params:
        t, orbital_var_dict, modes_imr = retval
    else:
        t, modes_imr = retval

    hp_ihc = modes_imr[(2, 2)] * 0  # Initialize with zeros
    if verbose:
        print("Shape of hp_ihc: {}".format(np.shape(hp_ihc)), flush=True)

    for el, em in modes_imr:
        ylm = lal.SpinWeightedSphericalHarmonic(inclination, coa_phase, -2, el, em)
        hp_ihc = hp_ihc + modes_imr[(el, em)] * ylm
        if verbose == 2:
            print(f"Adding mode {el}, {em} with ylm = {ylm}", flush=True)
            print(
                "... adding {}, {}".format(
                    modes_imr[(el, em)], modes_imr[(el, em)] * ylm
                ),
                flush=True,
            )
            print(f"hp after adding: {hp_ihc}", flush=True)

    if return_pycbc_timeseries:
        hp = pt.TimeSeries(hp_ihc.real, delta_t=t[1] - t[0], epoch=0)
        hc = pt.TimeSeries(-1 * hp_ihc.imag, delta_t=t[1] - t[0], epoch=0)
    else:
        hp = hp_ihc.real
        hc = -1 * hp_ihc.imag

    if return_orbital_params:
        if return_pycbc_timeseries:
            for name in orbital_var_dict:
                exec(
                    f"orbital_var_dict['{name}'] = pt.TimeSeries(orbital_var_dict['{name}'], delta_t=t[1]-t[0], epoch=0)"
                )
            return orbital_var_dict, hp, hc
        return t, orbital_var_dict, hp, hc

    if return_pycbc_timeseries:
        return hp, hc
    return t, hp, hc


def _get_window_start(freq_, delta_t_, delta_phi_, direction="forward"):
    """Integrates frequency backward/forward from the start/end until
    `delta_phi_` radians of orbital phase has elapsed.

    Args:
        freq_ (numpy.array):  Frequency time series, uniformly sampled
        delta_t_ (float64):   Step size in time
        delta_phi_ (float64): Phase shift in radians to integrate across
        direction (str, optional): Direction of integration in time.
                                   Defaults to "forward".

    Returns:
        int: Index in frequency array where `delta_phi` is reached
    """
    from scipy import integrate

    if direction == "backward":
        for idx in range(len(freq_) - 2, 0, -1):
            # this could be optimized to not repeat integration
            if abs(integrate.trapezoid(freq_[idx:], dx=delta_t_)) >= abs(delta_phi_):
                return idx
    elif direction == "forward":
        for idx in range(1, len(freq_)):
            # this could be optimized to not repeat integration
            if abs(integrate.trapezoid(freq_[: idx + 1], dx=delta_t_)) >= abs(
                delta_phi_
            ):
                return idx


def _get_transition_frequency_window(
    orbital_phase,
    orbital_freq,
    delta_t,
    f_mr_transition,  # orbital frequency
    num_hyb_orbits,
    keep_f_mr_transition_at_center,
    hybridize_using_avg_orbital_frequency,
    failsafe,
    verbose=False,
):
    """
    This function first finds where the `orbital_freq` crosses the transition
    frequency `f_mr_transition`, and finds the point in time that is
    `num_hyb_orbits` orbits before that time. This marks the start of the
    hybridization window, and the orbital frequency at that time is returned

    Parameters:
    -----------
        orbital_phase             -- Orbital phase evolution
        orbital_freq              -- Orbital frequency evolution
        delta_t                   -- Waveform's time grid-spacing (s)
        f_mr_transition           -- Inspiral to merger transition frequency
                                     (Hz). This should be the same (mode/orbital)
                                     frequency as passed for `orbital_freq`.
        num_hyb_orbits            -- number of orbital cycles to hybridize over.
        keep_f_mr_transition_at_center    -- If True, `f_mr_transition` is kept
                                     at the center of the hybridization window.
                                     Otherwise, it's kept at the end of the
                                     window (default).
        hybridize_using_avg_orbital_frequency -- If True, the orbit averaged
                                     frequency during the inspiral is used to
                                     hybridize modes, instead of the modes'
                                     frequency.
        failsafe                  -- If True, we make reasonable choices for the
                                     user, if the inputs to this method lead
                                     into exceptions.
        verbose                   -- Verbosity flag

    Returns:
    --------
        f_window_mr_transition    -- Hybridization frequency window (in Hz).
    """
    transition_idx = (
        len(orbital_freq) - 1 - np.argmax(orbital_freq[::-1] < f_mr_transition)
    )  # index at which orbital_freq becomes just smaller than transition orbital
    # frequency, towards the end of waveform

    if verbose > 4:
        print(
            f"""Transition frequency found at index {transition_idx}
            in the orbital frequency array of length {len(orbital_freq)}"""
        )

    if hybridize_using_avg_orbital_frequency:
        # We integrate `orbital_freq` to obtain `orbital_phase`
        from scipy import integrate

        orbital_phase = integrate.cumulative_trapezoid(
            orbital_freq, dx=delta_t, initial=0
        )
        if len(orbital_phase) != len(orbital_freq):
            raise RuntimeError(
                f"""Something went wrong while integrating orbital frequency.
They have different lengths: {len(orbital_freq)} and {len(orbital_phase)}."""
            )

    if keep_f_mr_transition_at_center:
        # Orbital cycle based hybridization that keeps f_mr_transition at
        # the hybridization frequency window's midpoint
        if hybridize_using_avg_orbital_frequency:
            window_start_idx = _get_window_start(
                orbital_freq[: transition_idx + 1],
                delta_t,
                num_hyb_orbits * np.pi,
                direction="backward",
            )
            window_end_idx = transition_idx + _get_window_start(
                orbital_freq[transition_idx:],
                delta_t,
                num_hyb_orbits * np.pi,
                direction="forward",
            )
            # FAILSAFE mode behavior
            if window_start_idx is None:
                if failsafe:
                    window_start_idx = 0
                else:
                    raise RuntimeError(
                        f"""Requested number of orbits to hybridize over not available
in the waveform before the transition frequency. Either decrease the number of
orbits to hybridize over (currently {num_hyb_orbits}) or increase the
inspiral-to-merger transition frequency. `window_start_idx` is None."""
                    )
            if window_end_idx is None:
                if failsafe:
                    window_end_idx = len(orbital_freq) - 1
                else:
                    raise RuntimeError(
                        f"""Requested number of orbits to hybridize over not available
in the waveform after the transition frequency. Either decrease the number of
orbits to hybridize over (currently {num_hyb_orbits}) or decrease the
inspiral-to-merger transition frequency. `window_end_idx` is None."""
                    )
        else:
            # phase is always a monotonically increasing function,
            #    hence using the more efficient np.searchsorted method
            window_start_idx = -1 + np.searchsorted(
                orbital_phase, orbital_phase[transition_idx] - num_hyb_orbits * np.pi
            )
            window_end_idx = np.searchsorted(
                orbital_phase, orbital_phase[transition_idx] + num_hyb_orbits * np.pi
            )
        f_window_mr_transition = 2 * np.min(
            [
                orbital_freq[window_end_idx] - f_mr_transition,
                f_mr_transition - orbital_freq[window_start_idx],
            ]
        )
    elif not keep_f_mr_transition_at_center:
        # Orbital cycle based hybridization that keeps f_mr_transition at
        # the hybridization frequency window's right end
        if hybridize_using_avg_orbital_frequency:
            window_start_idx = _get_window_start(
                orbital_freq[: transition_idx + 1],
                delta_t,
                2 * num_hyb_orbits * np.pi,
                direction="backward",
            )
            if window_start_idx is None:
                if failsafe:
                    window_start_idx = 0
                else:
                    raise RuntimeError(
                        f"""Requested number of orbits to hybridize over not available
in the waveform before the transition frequency. Either decrease the number of
orbits to hybridize over (currently {num_hyb_orbits}) or increase the
inspiral-to-merger transition frequency. `window_start_idx` is None."""
                    )
        else:
            # Orbital cycle based hybridization that keeps f_mr_transition
            # at the hybridization frequency window's end
            window_start_idx = (
                np.searchsorted(
                    orbital_phase,
                    orbital_phase[transition_idx] - 2 * np.pi * num_hyb_orbits,
                )
                - 1
            )
        f_window_mr_transition = (
            orbital_freq[transition_idx] - orbital_freq[window_start_idx]
        )

        if verbose > 4:
            print(
                f"""The transition is to happen in a window from
                frequencies: [{orbital_freq[window_start_idx]}, {orbital_freq[transition_idx]}]Hz,
                between indices: [{window_start_idx}, {transition_idx}]"""
            )
    else:
        raise RuntimeError(
            """We should never reach here. Did you set a non-bool value for the
            flag `keep_f_mr_transition_at_center`?"""
        )

    if verbose > 4:
        print(f"""f_window_mr_transition = {f_window_mr_transition}""")

    return f_window_mr_transition


def get_imr_esigma_modes(
    mass1,
    mass2,
    f_lower,
    delta_t,
    spin1z=0.0,
    spin2z=0.0,
    eccentricity=0.0,
    mean_anomaly=None,
    coa_phase=None,
    distance=1.0,
    f_ref=None,
    modes_to_use=[(2, 2), (3, 3), (4, 4)],
    mode_to_align_by=(2, 2),
    include_conjugate_modes=True,
    f_mr_transition=None,
    f_window_mr_transition=None,
    num_hyb_orbits=0.25,
    hybridize_using_avg_orbital_frequency=False,
    hybridize_aligning_merger_to_inspiral=False,
    keep_f_mr_transition_at_center=False,
    merger_ringdown_approximant="NRSur7dq4",
    return_hybridization_info=False,
    return_orbital_params=False,
    failsafe=True,
    verbose=False,
):
    """
    Returns IMR GW modes constructed using ESIGMA for inspiral and
    NRSur7dq4/SEOBNRv4PHM for merger-ringdown

    Parameters:
    -----------
        mass1, mass2              -- Binary's component masses (in solar masses)
        f_lower                   -- Starting frequency of the waveform (in Hz)
        f_ref                     -- Reference frequency at which to define the
                                     waveform parameters.
                                     We require f_ref <= f_lower.
                                     f_ref = f_lower by default.
        delta_t                   -- Waveform's time grid-spacing (in s)
        spin1z, spin2z            -- z-components of component dimensionless
                                     spins (lies in [0,1))
        eccentricity              -- Initial eccentricity
        mean_anomaly              -- Mean anomaly of the periastron (radians)
        coa_phase                 -- Coalesence phase of the binary (in rad)
        distance                  -- Luminosity distance to the binary (in Mpc)
        modes_to_use              -- GW modes to use. List of tuples (l, |m|)
        mode_to_align_by          -- GW mode to use to align inspiral and merger
                                     in phase and time
        include_conjugate_modes   -- If True, (l, -|m|) modes are included as
                                     well
        f_mr_transition           -- Inspiral to merger transition GW frequency
                                     (Hz). Should correspond to the mode
                                     specified by `mode_to_align_by`.
                                     Defaults to the minimum of the Kerr and
                                     Schwarzschild ISCO frequency equivalent
                                     for the mode `mode_to_align_by`.
        f_window_mr_transition    -- Hybridization frequency window (in Hz).
                                     Should correspond to the mode specified by
                                     `mode_to_align_by`.
                                     Disabled by the default value (None). In
                                     such a case, the hybridization proceeds
                                     over a window of `num_hyb_orbits` orbital
                                     cycles (1 orbital cycle ~ 2 GW cycles)
                                     that ends at the frequency value given by
                                     `f_mr_transition`.
                                     Also see `keep_f_mr_transition_at_center`
                                     to choose the position of `f_mr_transition`
                                     within this window.
        num_hyb_orbits            -- number of orbital cycles to hybridize over.
                                     Only used if f_window_mr_transition is not
                                     specified.
        hybridize_using_avg_orbital_frequency -- If True, the orbit averaged
                                     frequency during the inspiral is used to
                                     hybridize modes, instead of the modes'
                                     frequency.
        hybridize_aligning_merger_to_inspiral -- (default: False) If True, the
                                     merger-ringdown mode would be phase aligned
                                     to the inspiral
                                     If False, the inspiral is phase aligned
                                     Note: specify the desired
        keep_f_mr_transition_at_center -- If True, `f_mr_transition` is kept at
                                     the center of the hybridization window.
                                     Otherwise, it's kept at the end of the
                                     window (default).
        merger_ringdown_approximant    -- Choose merger-ringdown model. Tested
                                     choices: [NRSur7dq4, SEOBNRv4PHM]
        return_hybridization_info -- If True, returns hybridization related data
        return_orbital_params     -- If True, returns the orbital evolution of
                                     all the orbital elements (in
                                     geometrized units). Can also be a list of
                                     orbital variable names to return
                                     only those specific variables. Available
                                     orbital variables names are:
                                  ['x', 'e', 'l', 'phi', 'phidot', 'r', 'rdot'].
                                     Note that these are available only for the
                                     inspiral portion of the waveform!
        failsafe                  -- If True, we make reasonable choices for the
                                     user, if the inputs to this method lead
                                     into exceptions.
        verbose                   -- Verbosity level. Available values are: 0, 1, 2

    Returns:
    --------
        modes_imr         -- Dictionary of IMR GW modes (PyCBC TimeSeries)
        orbital_var_dict  -- Dictionary of evolution of orbital elements.
                             Returned only if the flag `return_orbital_params`
                             is set
        retval            -- Hybridization related data. Returned only if the
                             flag `return_hybridization_info` is set
    """
    if not hasattr(ls, merger_ringdown_approximant):
        raise IOError(
            """We cannot generate individual modes for {merger_ringdown_approximant}.
                      Try one of: [NRSur7dq4, SEOBNRv4PHM]"""
        )
    if (mean_anomaly is None) and (coa_phase is None):
        raise IOError(
            f"""Please specify one of the phase angles, either of
                      `mean_anomaly` or `coa_phase`."""
        )
    if hybridize_aligning_merger_to_inspiral and (coa_phase is None):
        raise IOError(
            f"""If you want to attach ESIGMA inspiral to merger, by
                      phase shifting merger to inspiral, please specify the
                      phase angle `coa_phase`"""
        )
    if (not hybridize_aligning_merger_to_inspiral) and (mean_anomaly is None):
        raise IOError(
            f"""If you want to attach ESIGMA inspiral to merger, by
                      phase shifting inspiral to merger, please specify the
                      phase angle `mean_anomaly`"""
        )
    if mean_anomaly is None:
        mean_anomaly = 0
    if coa_phase is None:
        coa_phase = 0

    available_inspiral_orbital_params = ["x", "e", "l", "phi", "phidot", "r", "rdot"]
    if return_orbital_params == True:
        return_orbital_params = available_inspiral_orbital_params
        return_orbital_params_user = set(return_orbital_params)
    elif (
        isinstance(return_orbital_params, list)
        or isinstance(return_orbital_params, set)
        or isinstance(return_orbital_params, tuple)
    ):
        return_orbital_params_user = return_orbital_params.copy()
        return_orbital_params_user = set(return_orbital_params_user).intersection(
            set(available_inspiral_orbital_params)
        )
        if return_orbital_params_user != set(return_orbital_params):
            print(
                f"""Warning: You requested the following list of orbital
parameters to be returned: {return_orbital_params}, but we reduce it to
{return_orbital_params_user} as we only have the evolution of the following 
parameters available with us: {available_inspiral_orbital_params}.
                  """
            )
    elif not return_orbital_params:
        return_orbital_params = []
        return_orbital_params_user = False

    return_orbital_params = set(return_orbital_params)
    return_orbital_params = return_orbital_params.union(
        set(["e"])
    )  # "e" needed necessarily for hybridization error printing

    if failsafe or (verbose > 1):
        return_orbital_params = return_orbital_params.union(set(["phidot"]))

    # If the user does not provide the width of hybridization window (in terms
    # of orbital frequency) over which the inspiral should transition to
    # merger-ringdown, we switch schemes and hybridize over `num_hyb_orbits`
    # orbits instead.
    if f_window_mr_transition is None:
        # These will be used for figuring out the hybridization window
        return_orbital_params = return_orbital_params.union(set(["phi", "phidot"]))
        if hybridize_using_avg_orbital_frequency:
            return_orbital_params = return_orbital_params.union(set(["x"]))

    _, mode_to_align_by_em = mode_to_align_by

    # If the user does not provide the orbital frequency at which the inspiral
    # should transition to merger-ringdown, we use sensible defaults here.
    if f_mr_transition is None:
        # Kerr ISCO frequency
        f_Kerr = f_ISCO_spin(mass1, mass2, spin1z, spin2z)
        # Schwarzschild ISCO frequency
        f_Schwarz = 6.0**-1.5 / (mass1 + mass2) / lal.MTSUN_SI / lal.PI
        f_mr_transition = min(f_Kerr, f_Schwarz) * (mode_to_align_by_em / 2)

    retval = get_inspiral_esigma_modes(
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        eccentricity=eccentricity,
        mean_anomaly=mean_anomaly,
        distance=distance,
        f_lower=f_lower,
        f_ref=f_ref,
        delta_t=delta_t,
        modes_to_use=modes_to_use,
        include_conjugate_modes=include_conjugate_modes,
        return_orbital_params=list(return_orbital_params),
        return_pycbc_timeseries=False,
        verbose=verbose,
    )

    # Retrieve modes, orbital phase and frequency from the returned list
    orbital_eccentricity = retval[-2]["e"]
    # Throw error if eccentricity at the end of inspiral is definitely unsafe
    if orbital_eccentricity[-1] > ECCENTRICITY_LEVEL_ISCO_ERROR:
        raise IOError(
            f"""ERROR: You entered a very large initial eccentricity
{eccentricity}. The orbital eccentricity at the end of inspiral was
{orbital_eccentricity[-1]}. The merger-ringdown attachment with a
quasicircular will be questionable."""
        )
    # Warn user if eccentricity at the end of inspiral is potentially unsafe
    if orbital_eccentricity[-1] > ECCENTRICITY_LEVEL_ISCO_WARNING and verbose:
        print(
            f"""WARNING: You entered a very large initial eccentricity
{eccentricity}. The orbital eccentricity at the end of inspiral was
{orbital_eccentricity[-1]}. The merger-ringdown attachment with a quasicircular
model might be affected."""
        )

    if (f_window_mr_transition is None) or failsafe or (verbose > 1):
        if hybridize_using_avg_orbital_frequency:
            orbital_frequency = (
                retval[-2]["x"] ** 1.5 / ((mass1 + mass2) * lal.MTSUN_SI) / (2 * np.pi)
            )
        else:
            orbital_frequency = (
                retval[-2]["phidot"] / ((mass1 + mass2) * lal.MTSUN_SI) / (2 * np.pi)
            )
    modes_inspiral_numpy = retval[-1]

    if return_orbital_params_user:
        orbital_vars_dict = {
            key: pt.TimeSeries(retval[-2][key], delta_t=delta_t, epoch=retval[0][0])
            for key in return_orbital_params_user
        }

    # DEBUG
    if verbose > 5:
        el, em = mode_to_align_by
        mode_phase = gwnr.waveform.hybridize.compute_phase(
            modes_inspiral_numpy[mode_to_align_by]
        )
        mode_frequency = gwnr.waveform.hybridize.compute_frequency(mode_phase, delta_t)
        print(
            f"""DEBUG: Orbital freq at end of inspiral is {orbital_frequency[-1]}Hz,
mode-{el}{em} freq at the end of inspiral is {mode_frequency[-1]}Hz, max and min
mode-{el}{em} frequencies are {np.max(mode_frequency)}Hz and
{np.min(mode_frequency)}Hz, and the transition frequency (of {el},{em}-mode)
requested is {f_mr_transition}Hz, which should be less than the maximum freq of
{el}{em}-mode: {mode_frequency.max()}Hz."""
        )
        return (
            modes_inspiral_numpy,
            mode_phase,
            mode_frequency,
            orbital_frequency,
            orbital_eccentricity,
            orbital_vars_dict,
        )

    # In case the user-specified transition frequency is too high, and they
    # requested failsafe mode, we reset it to a reasonable value.
    if failsafe:
        el, em = mode_to_align_by
        mode_phase = gwnr.waveform.hybridize.compute_phase(
            modes_inspiral_numpy[mode_to_align_by]
        )
        mode_frequency = gwnr.waveform.hybridize.compute_frequency(mode_phase, delta_t)
        if mode_frequency.max() < f_mr_transition:
            if verbose:
                print(
                    f"""FAILSAFE: Maximum orbital freq during inspiral is
{orbital_frequency.max()}Hz, and max frequency of {el},{em}-mode is
{mode_frequency.max()}Hz, so we are resetting transition frequency from
{f_mr_transition}Hz to {mode_frequency.max()}Hz."""
                )
            f_mr_transition = mode_frequency.max()

    # If the user does not provide the width of hybridization window (
    # `f_window_mr_transition`) over which the inspiral should transition to
    # merger-ringdown, we switch schemes and hybridize over `num_hyb_orbits`
    # orbits instead.
    if f_window_mr_transition is None:
        f_window_mr_transition = (
            _get_transition_frequency_window(
                retval[-2]["phi"],
                orbital_frequency,
                delta_t,
                f_mr_transition / mode_to_align_by_em,
                num_hyb_orbits,
                keep_f_mr_transition_at_center,
                hybridize_using_avg_orbital_frequency,
                failsafe=failsafe,
                verbose=verbose,
            )
            * mode_to_align_by_em
        )

    # This is done to make use of the same hybridization code, that actually
    # assumes f_mr_transition to be at window's midpoint, to keep the
    # hybridization window's end at f_mr_transition
    if not keep_f_mr_transition_at_center:
        f_mr_transition -= f_window_mr_transition / 2.0

    # Generate NR surrogate waveform that will be our merger-ringdown, starting
    # from a frequency = 90% of
    max_retries = 4
    f_lower_mr = (f_mr_transition - f_window_mr_transition / 2) * (
        1.8 / mode_to_align_by_em
    )
    for _ in range(max_retries):
        try:
            if verbose:
                print(f"Generating MR waveform from {f_lower_mr}Hz...")
            hlm_mr = ls.SimInspiralChooseTDModes(
                coa_phase,  # phiRef
                delta_t,  # deltaT
                mass1 * lal.MSUN_SI,
                mass2 * lal.MSUN_SI,
                0,  # spin1x
                0,  # spin1y
                spin1z,
                0,  # spin2x
                0,  # spin2y
                spin2z,
                f_lower_mr,  # f_min
                f_lower_mr,  # f_ref
                distance * lal.PC_SI * 1.0e6,
                None,  # LALpars
                4,  # lmax
                getattr(ls, merger_ringdown_approximant),
            )
            break
        except:
            f_lower_mr *= 0.8
            continue

    modes_mr = {}
    while hlm_mr is not None:
        modes_mr[(hlm_mr.l, hlm_mr.m)] = hlm_mr.mode
        hlm_mr = hlm_mr.next

    modes_mr_numpy = {k: np.asarray(modes_mr[k].data.data) for k in modes_mr}

    try:
        retval = gwnr.waveform.hybridize.hybridize_modes(
            modes_inspiral_numpy,
            modes_mr_numpy,
            orbital_frequency,
            f_mr_transition,
            frq_width=f_window_mr_transition,
            delta_t=delta_t,
            modes_to_hybridize=modes_to_use,
            mode_to_align_by=mode_to_align_by,
            hybridize_using_avg_orbital_frequency=hybridize_using_avg_orbital_frequency,
            hybridize_aligning_merger_to_inspiral=hybridize_aligning_merger_to_inspiral,
            include_conjugate_modes=include_conjugate_modes,
            verbose=verbose,
        )
    except Exception as exc:
        print(
            f"""Inspiral + MergerRingdown attachment failed. Its very likely
that you entered a very large initial eccentricity {eccentricity}. The orbital
eccentricity at the end of inspiral was {orbital_eccentricity[-1]}
              """
        )
        raise exc
    modes_imr_numpy = retval[0]

    # Align modes at peak of (2, 2) mode
    if mode_to_align_by not in modes_imr_numpy:
        mode_to_align_by = list(modes_imr_numpy.keys())[0]
    idx_peak = abs(modes_imr_numpy[mode_to_align_by]).argmax()
    t_peak = idx_peak * delta_t

    itime = time.perf_counter()
    modes_imr = {}
    for el, em in modes_imr_numpy:
        modes_imr[(el, em)] = pt.TimeSeries(
            modes_imr_numpy[(el, em)], delta_t=delta_t, epoch=-1 * t_peak
        )
    if verbose:
        print(
            "Time taken to store in pycbc.TimeSeries is {} secs".format(
                time.perf_counter() - itime
            )
        )

    if verbose:
        print("hybridized.")

    if return_hybridization_info and return_orbital_params_user:
        return modes_imr, orbital_vars_dict, retval
    elif return_orbital_params_user:
        return modes_imr, orbital_vars_dict
    elif return_hybridization_info:
        return modes_imr, retval
    return modes_imr


def get_imr_esigma_waveform(
    mass1,
    mass2,
    f_lower,
    delta_t,
    f_ref=None,
    spin1z=0.0,
    spin2z=0.0,
    eccentricity=0.0,
    mean_anomaly=0.0,
    inclination=0.0,
    coa_phase=0.0,
    distance=1.0,
    modes_to_use=[(2, 2), (3, 3), (4, 4)],
    mode_to_align_by=(2, 2),
    f_mr_transition=None,
    f_window_mr_transition=None,
    num_hyb_orbits=0.25,
    hybridize_using_avg_orbital_frequency=False,
    keep_f_mr_transition_at_center=False,
    merger_ringdown_approximant="NRSur7dq4",
    return_hybridization_info=False,
    return_orbital_params=False,
    failsafe=True,
    verbose=False,
    **kwargs,
):
    """
    Returns IMR GW polarizations constructed using IMR ESIGMA modes

    Parameters:
    -----------
        mass1, mass2              -- Binary's component masses (in solar masses)
        f_lower                   -- Starting frequency of the waveform (in Hz)
        f_ref                     -- Reference frequency at which to define the
                                     waveform parameters.  We require that
                                     `f_ref <= f_lower`.
                                     `f_ref = f_lower` by default.
        delta_t                   -- Waveform's time grid-spacing (in s)
        spin1z, spin2z            -- z-components of component dimensionless
                                     spins (lies in [0,1))
        eccentricity              -- Initial eccentricity
        mean_anomaly              -- Mean anomaly of the periastron (in rad)
        inclination               -- Inclination (in rad), defined as the angle
                                     between the orbital angular momentum L and
                                     the line-of-sight
        coa_phase                 -- Coalesence phase of the binary (in rad)
        distance                  -- Luminosity distance to the binary (in Mpc)
        modes_to_use              -- GW modes to use. List of tuples (l, |m|)
        mode_to_align_by          -- GW mode to use to align inspiral and merger
                                     in phase and time
        f_mr_transition           -- Inspiral to merger transition GW frequency
                                     (Hz).
                                     Defaults to the minimum of the Kerr and
                                     Schwarzschild ISCO frequency
        f_window_mr_transition    -- Hybridization frequency window (in Hz).
                                     Disabled by the default value (None). In
                                     such a case, the hybridization proceeds
                                     over a window of `num_hyb_orbits` orbital
                                     cycles (1 orbital cycle ~ 2 GW cycles)
                                     that ends at the frequency value given by
                                     `f_mr_transition`.
                                     Also see `keep_f_mr_transition_at_center`
                                     to choose the position of `f_mr_transition`
                                     within this window.
        num_hyb_orbits            -- number of orbital cycles to hybridize over.
                                     Only used if f_window_mr_transition is not
                                     specified.
        hybridize_using_avg_orbital_frequency -- If True, the orbit averaged
                                     frequency during the inspiral is used to
                                     hybridize modes, instead of the modes'
                                     frequency.
        keep_f_mr_transition_at_center -- If True, `f_mr_transition` is kept at
                                     the center of the hybridization window.
                                     Otherwise, it's kept at the end of the
                                     window (default).
        merger_ringdown_approximant    -- Choose merger-ringdown model. Tested
                                     choices: [NRSur7dq4, SEOBNRv4PHM]
        return_hybridization_info -- If True, returns hybridization related data
        return_orbital_params     -- If True, returns the orbital evolution of
                                     all the orbital elements (in
                                     geometrized units). Can also be a list of
                                     orbital variable names to return
                                     only those specific variables. Available
                                     orbital variables names are:
                                  ['x', 'e', 'l', 'phi', 'phidot', 'r', 'rdot'].
                                     Note that these are available only for the
                                     inspiral portion of the waveform!
        failsafe                  -- If True, we make reasonable choices for the
                                     user, if the inputs to this method lead
                                     into exceptions.
        verbose                   -- Verbosity level. Available values are: 0, 1, 2

    Returns:
    --------
        hp, hc       -- Plus and cross IMR GW polarizations PyCBC TimeSeries
        orbital_vars_dict -- Dictionary of evolution of orbital elements.
                        Returned only if return_orbital_params is specified
        retval       -- Hybridization related data.
                        Returned only if return_hybridization_info is True
    """

    retval = get_imr_esigma_modes(
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        eccentricity=eccentricity,
        mean_anomaly=mean_anomaly,
        distance=distance,
        f_lower=f_lower,
        f_ref=f_ref,
        delta_t=delta_t,
        modes_to_use=modes_to_use,
        mode_to_align_by=mode_to_align_by,
        include_conjugate_modes=True,  # Always include conjugate modes while generating polarizations
        f_mr_transition=f_mr_transition,
        f_window_mr_transition=f_window_mr_transition,
        num_hyb_orbits=num_hyb_orbits,
        hybridize_using_avg_orbital_frequency=hybridize_using_avg_orbital_frequency,
        keep_f_mr_transition_at_center=keep_f_mr_transition_at_center,
        merger_ringdown_approximant=merger_ringdown_approximant,
        return_hybridization_info=return_hybridization_info,
        return_orbital_params=return_orbital_params,
        failsafe=failsafe,
        verbose=verbose,
    )
    if return_hybridization_info and return_orbital_params:
        modes_imr, orbital_vars_dict, retval = retval
    elif return_hybridization_info:
        modes_imr, retval = retval
    elif return_orbital_params:
        modes_imr, orbital_vars_dict = retval
    else:
        modes_imr = retval

    hp_ihc = modes_imr[(2, 2)] * 0  # Initialize with zeros
    if verbose:
        print("Shape of hp_ihc: {}".format(np.shape(hp_ihc)), flush=True)

    for el, em in modes_imr:
        ylm = lal.SpinWeightedSphericalHarmonic(inclination, coa_phase, -2, el, em)
        hp_ihc = hp_ihc + modes_imr[(el, em)] * ylm
        if verbose == 2:
            print(f"Adding mode {el}, {em} with ylm = {ylm}", flush=True)
            print(
                "... adding {}, {}".format(
                    modes_imr[(el, em)], modes_imr[(el, em)] * ylm
                ),
                flush=True,
            )
            print(f"hp after adding: {hp_ihc.data}", flush=True)

    hp = hp_ihc.real()
    hc = -1 * hp_ihc.imag()

    if return_hybridization_info and return_orbital_params:
        return hp, hc, orbital_vars_dict, retval
    elif return_hybridization_info:
        return hp, hc, retval
    elif return_orbital_params:
        return hp, hc, orbital_vars_dict
    return hp, hc


class FitMOmegaIMRAttachmentNonSpinning:
    called_once = False

    def __init__(self):
        self.called_once = False
        return

    @classmethod
    def fit_quadratic_poly(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_quadratic_poly")
            cls.called_once = True
        assert len(coeffs) == 2, "{} coeffs passed!".format(len(coeffs))
        a1, a2 = coeffs
        return (1.0 / 6**1.5) * (1.0 + a1 * eta + a2 * eta**2)

    @classmethod
    def fit_cubic_poly(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_cubic_poly")
            cls.called_once = True
        assert len(coeffs) == 3, "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3 = coeffs
        return (1.0 / 6**1.5) * (1.0 + a1 * eta + a2 * eta**2 + a3 * eta**3)

    @classmethod
    def fit_ratio_poly_44(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_poly_44")
            cls.called_once = True
        assert len(coeffs) == 6, "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2, b3 = coeffs
        return (
            (1.0 / 6**1.5)
            * (1.0 + a1 * eta + a2 * eta**2 + a3 * eta**3)
            / (1.0 + b1 * eta + b2 * eta**2 + b3 * eta**3)
        )

    @classmethod
    def fit_ratio_sqrt_poly_44(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_sqrt_poly_44")
            cls.called_once = True
        assert len(coeffs) == 6, "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2, b3 = coeffs
        s_eta = eta**0.5
        return (
            (1.0 / 6**1.5)
            * (1.0 + a1 * s_eta + a2 * s_eta**2 + a3 * s_eta**3)
            / (1.0 + b1 * s_eta + b2 * s_eta**2 + b3 * s_eta**3)
        )

    @classmethod
    def fit_ratio_sqrt_hyb1_poly_44(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_sqrt_hyb1_poly_44")
            cls.called_once = True
        assert len(coeffs) == 6, "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2, b3 = coeffs
        s_eta = eta**0.5
        return (
            (1.0 / 6**1.5)
            * (1.0 + a1 * eta + a2 * eta**2 + a3 * eta**3)
            / (1.0 + b1 * eta + b2 * eta**2 + b3 * eta**3)
        )

    @classmethod
    def fit_ratio_poly_43(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_poly_43")
            cls.called_once = True
        assert len(coeffs) == 5, "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2 = coeffs
        return (
            (1.0 / 6**1.5)
            * (1.0 + a1 * eta + a2 * eta**2 + a3 * eta**3)
            / (1.0 + b1 * eta + b2 * eta**2)
        )

    @classmethod
    def fit_ratio_sqrt_poly_43(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_sqrt_poly_43")
            cls.called_once = True
        assert len(coeffs) == 5, "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2 = coeffs
        s_eta = eta**0.5
        return (
            (1.0 / 6**1.5)
            * (1.0 + a1 * s_eta + a2 * s_eta**2 + a3 * s_eta**3)
            / (1.0 + b1 * s_eta + b2 * s_eta**2)
        )

    @classmethod
    def fit_ratio_sqrt_hyb1_poly_43(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_sqrt_hyb1_poly_43")
            cls.called_once = True
        assert len(coeffs) == 5, "{} coeffs passed!".format(len(coeffs))
        a1, a2, a3, b1, b2 = coeffs
        s_eta = eta**0.5
        return (
            (1.0 / 6**1.5)
            * (1.0 + a1 * eta * s_eta + a2 * eta**2 * s_eta + a3 * eta**3 * s_eta)
            / (1.0 + b1 * eta + b2 * eta**2)
        )

    @classmethod
    def fit_ratio_poly_34(cls, eta, coeffs):
        if not cls.called_once:
            logging.info("Using fit_ratio_poly_34")
            cls.called_once = True
        assert len(coeffs) == 5, "{} coeffs passed!".format(len(coeffs))
        a1, a2, b1, b2, b3 = coeffs
        return (
            (1.0 / 6**1.5)
            * (1.0 + a1 * eta + a2 * eta**2)
            / (1.0 + b1 * eta + b2 * eta**2 + b3 * eta**3)
        )
