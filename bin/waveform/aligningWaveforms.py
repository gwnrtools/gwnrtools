#!/usr/bin/env python

import UseNRinDA
from pycbc.waveform import amplitude_from_polarizations, phase_from_polarizations
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from pycbc.pnutils import mchirp_eta_to_mass1_mass2, eta_mass1_to_mass2
from pycbc.psd import *
from pycbc.waveform.utils import *
from pycbc.waveform import *
from pycbc.types import *
from pycbc.filter import *
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex': True})

######
golden_ratio = (5.**0.5 + 1.) / 2.


def shift_waveform_phase_time(hp, hc, t_shift, ph_shift):
    hpnew = TimeSeries(hp, epoch=hp._epoch, delta_t=hp.delta_t, dtype=hp.dtype)
    hcnew = TimeSeries(hc, epoch=hc._epoch, delta_t=hc.delta_t, dtype=hc.dtype)
    # Now apply phase shift
    if ph_shift != 0.:
        amplitude = amplitude_from_polarizations(hpnew, hc)
        phase = phase_from_polarizations(hpnew, hc)
        print("shifting by %f radians" % ph_shift)
        phase = phase + ph_shift
        hpnew = TimeSeries(amplitude * np.cos(phase),
                           epoch=hpnew._epoch,
                           delta_t=hpnew.delta_t,
                           dtype=hpnew.dtype)
        hcnew = TimeSeries(amplitude * np.sin(phase),
                           epoch=hcnew._epoch,
                           delta_t=hcnew.delta_t,
                           dtype=hcnew.dtype)
    # First apply time shift
    if t_shift != 0:
        id_shift = int(np.round(t_shift / hpnew.delta_t))
        print("shifting by %d" % id_shift)
        hpnew.roll(id_shift)
        hcnew.roll(id_shift)
    return hpnew, hcnew


def align_waveforms_amplitude_peak(hplus1, hcross1, hplus2, hcross2):
    #
    # Find amplitude peaks
    #
    hp1, hc1 = TimeSeries(hplus1), TimeSeries(hcross1)
    hp2, hc2 = TimeSeries(hplus2), TimeSeries(hcross2)
    amp1 = amplitude_from_polarizations(hp1, hc1)
    amp2 = amplitude_from_polarizations(hp2, hc2)
    # Get amplitude peaks
    amp1I = interp1d(amp1.sample_times, -1 * amp1.data)
    x0 = np.float64(
        np.where(amp1.data == max(amp1.data))[0][0] * amp1.delta_t +
        amp1._epoch)
    tmp = minimize_scalar(amp1I,
                          x0,
                          method='bounded',
                          bounds=(x0 - 10 * amp1.delta_t,
                                  x0 + 10 * amp1.delta_t))
    h1_max_amp_time = tmp['x']
    h1_max_amp = -1 * tmp['fun']
    amp2I = interp1d(amp2.sample_times, -1 * amp2.data)
    x0 = np.float64(
        np.where(amp2.data == max(amp2.data))[0][0] * amp2.delta_t +
        amp2._epoch)
    tmp = minimize_scalar(amp2I,
                          x0,
                          method='bounded',
                          bounds=(x0 - 10 * amp2.delta_t,
                                  x0 + 10 * amp2.delta_t))
    h2_max_amp_time = tmp['x']
    h2_max_amp = -1 * tmp['fun']
    print("h1 max time = %f, epoch = %f" %
          (h1_max_amp_time, float(hp1._epoch)))
    print("h2 max time = %f, epoch = %f" %
          (h2_max_amp_time, float(hp2._epoch)))
    # Amplitude location from the start
    t1 = np.float64(h1_max_amp_time - hp1._epoch)
    t2 = np.float64(h2_max_amp_time - hp2._epoch)
    t_shift = t1 - t2
    print("time shift = %f" % t_shift)
    #
    # Find phase shift
    #
    phs1 = phase_from_polarizations(hp1, hc1)
    phs2 = phase_from_polarizations(hp2, hc2)
    ph1 = phs1[int(np.round(t1 / hp1.delta_t))]
    ph2 = phs2[int(np.round(t2 / hp2.delta_t))]
    print("phase1 at peak idx = %d, = %f" %
          (int(np.round(t1 / hp1.delta_t)), ph1))
    print("phase2 at peak idx = %d, = %f" %
          (int(np.round(t2 / hp2.delta_t)), ph2))
    ph_shift = (ph1 - ph2) * 1
    print("phase shift = %f" % ph_shift)
    #
    # Enforced that phase at merger be 0 for BOTH
    hp1, hc1 = shift_waveform_phase_time(hp1, hc1, 0, -0 - 1. * ph1)
    hp2, hc2 = shift_waveform_phase_time(hp2, hc2, t_shift, -0 - 1. * ph2)
    #
    return hp1, hc1, hp2, hc2


def align_waveforms_at_frequency(hplus1, hcross1, hplus2, hcross2, falign):
    #
    # Find amplitude peaks
    #
    hp1 = TimeSeries(hplus1)
    hc1 = TimeSeries(hcross1)
    hp2 = TimeSeries(hplus2)
    hc2 = TimeSeries(hcross2)
    #
    freq1 = frequency_from_polarizations(hp1, hc1)
    f1I = interp1d(np.arange(len(freq1)), np.abs(freq1.data - falign))
    id_start = int(0.2 / freq1.delta_t)
    for idx in range(id_start, len(freq1)):
        if freq1[idx] > 2 * falign and freq1[idx + 1] > 2 * falign:
            break
    tmp = minimize_scalar(f1I,
                          id_start,
                          method='bounded',
                          bounds=(id_start, idx))
    f1_align_idx = int(np.round(tmp['x']))
    t1 = f1_align_idx * freq1.delta_t
    #
    freq2 = frequency_from_polarizations(hp2, hc2)
    f2I = interp1d(np.arange(len(freq2)), np.abs(freq2.data - falign))
    id_start = int(0.2 / freq2.delta_t)
    for idx in range(id_start, len(freq2)):
        if freq2[idx] > 2 * falign and freq2[idx + 1] > 2 * falign:
            break
    tmp = minimize_scalar(f2I,
                          id_start,
                          method='bounded',
                          bounds=(id_start, idx))
    f2_align_idx = int(np.round(tmp['x']))
    t2 = f2_align_idx * freq2.delta_t
    #
    t_shift = t1 - t2
    phs1 = phase_from_polarizations(hp1, hc1)
    phs2 = phase_from_polarizations(hp2, hc2)
    ph1 = phs1[f1_align_idx]
    ph2 = phs2[f2_align_idx]
    #
    print("f1 time = %f" % (t1))
    print("f2 time = %f" % (t2))
    print("phase1 at peak = %f" % (ph1))
    print("phase2 at peak = %f" % (ph2))
    ph_shift = (ph1 - ph2) * 1
    print("phase shift = %f" % ph_shift)
    #
    # Enforced that phase at merger be 0 for BOTH
    hp1, hc1 = shift_waveform_phase_time(hp1, hc1, 0, -0 - 1. * ph1)
    hp2, hc2 = shift_waveform_phase_time(hp2, hc2, 1 * t_shift, -0 - 1. * ph2)
    #
    return hp1, hc1, hp2, hc2


def align_waveforms_optimally(h_plus1,
                              h_cross1,
                              hplus2,
                              hcross2,
                              psd=None,
                              low_frequency_cutoff=None,
                              high_frequency_cutoff=None,
                              tsign=1,
                              phsign=-1):
    #
    h_plus2 = TimeSeries(hplus2,
                         epoch=hplus2._epoch,
                         delta_t=hplus2.delta_t,
                         dtype=hplus2.dtype)
    h_cross2 = TimeSeries(hcross2,
                          epoch=hcross2._epoch,
                          delta_t=hcross2.delta_t,
                          dtype=hcross2.dtype)
    #
    if psd is None:
        raise IOError("Need psd to be input for now!")
    f_low, f_high = low_frequency_cutoff, high_frequency_cutoff
    if f_high is None:
        f_high = 1. / h_plus1.delta_t / 2.
    #
    htilde = make_frequency_series(h_plus1)
    stilde = make_frequency_series(h_plus2)
    #
    # Determine the phase and time shifts for optimal match
    snr, corr, snr_norm = \
        matched_filter_core(htilde, stilde, psd, f_low, f_high, None)
    max_snr, max_id = snr.abs_max_loc()
    t_shift = -1 * max_id * h_plus2.delta_t
    ph_shift = np.angle(snr[max_id])
    #
    hp2, hc2 = \
        shift_waveform_phase_time(
            h_plus2,
            h_cross2,
            tsign * t_shift,
            phsign * ph_shift)
    return h_plus1, h_cross1, hp2, hc2


def make_inspiral_merger_plot(hp1,
                              hc1,
                              hp2,
                              hc2,
                              tscale=1.,
                              lp=0.9,
                              rp=1.1,
                              xlabel=r'$\mathrm{Time (M)}$',
                              ylabel=r'$\\frac{R}{M}\,h_{22}\,(t)$',
                              legend=None,
                              savefig='plot.pdf'):
    a1 = amplitude_from_polarizations(hp1, hc1)
    _, max_id1 = a1.abs_max_loc()
    a2 = amplitude_from_polarizations(hp2, hc2)
    _, max_id2 = a2.abs_max_loc()
    #
    # plot it
    fig = plt.figure(figsize=(8 / 2, 8 / golden_ratio / 2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])
    ax0.plot(np.arange(len(hp1)) * tscale,
             hp1,
             np.arange(len(hp2)) * tscale,
             hp2,
             lw=0.5)
    xmin, xmax = 0, min(max_id1, max_id2) * lp * tscale
    ax0.set_xlim(xmin, xmax)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    ax0.legend(legend, loc='best')
    #
    ax1 = plt.subplot(gs[1])
    ax1.plot(np.arange(len(hp1)) * tscale,
             hp1,
             np.arange(len(hp2)) * tscale,
             hp2,
             lw=0.5)
    xmin, xmax = min(max_id1, max_id2) * lp * \
        tscale, min(max_id1, max_id2) * rp * tscale
    ax1.set_xlim(xmin, xmax)
    print(xmin, xmax)
    ax1.set_xticks([
        int(xmin * 0.9 + 0.1 * xmax),
        int(xmin * 0.5 + 0.5 * xmax),
        int(xmin * 0.05 + 0.95 * xmax)
    ])
    #
    plt.tight_layout()
    plt.savefig(savefig)


def make_inspiral_merger_plot_3(hp1,
                                hc1,
                                hp2,
                                hc2,
                                hp3,
                                hc3,
                                tscale=1.,
                                lp=0.9,
                                rp=1.1,
                                xlabel=r'$\mathrm{Time (M)}$',
                                ylabel=r'$\\frac{R}{M}\,h_{22}\,(t)$',
                                legend=None,
                                savefig='plot.pdf'):
    a1 = amplitude_from_polarizations(hp1, hc1)
    _, max_id1 = a1.abs_max_loc()
    a2 = amplitude_from_polarizations(hp2, hc2)
    _, max_id2 = a2.abs_max_loc()
    #
    # plot it

    fig = plt.figure(figsize=(8 / 2, 8 / golden_ratio / 2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])
    ax0.plot(np.arange(len(hp2)) * tscale,
             hp2,
             '-',
             np.arange(len(hp3)) * tscale,
             hp3,
             '-',
             np.arange(len(hp1)) * tscale,
             hp1,
             'k--',
             lw=0.5)
    xmin, xmax = 0, min(max_id1, max_id2) * lp * tscale
    ax0.set_xlim(xmin, xmax)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    ax0.legend(legend, loc='best')
    #
    ax1 = plt.subplot(gs[1])
    ax1.plot(np.arange(len(hp2)) * tscale,
             hp2,
             '-',
             np.arange(len(hp3)) * tscale,
             hp3,
             '-',
             np.arange(len(hp1)) * tscale,
             hp1,
             'k--',
             lw=0.5)
    xmin, xmax = min(max_id1, max_id2) * lp * \
        tscale, min(max_id1, max_id2) * rp * tscale
    ax1.set_xlim(xmin, xmax)
    print(xmin, xmax)
    ax1.set_xticks([
        int(xmin * 0.9 + 0.1 * xmax),
        int(xmin * 0.5 + 0.5 * xmax),
        int(xmin * 0.05 + 0.95 * xmax)
    ])
    #
    plt.tight_layout()
    plt.savefig(savefig)


#####

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
fig_height_pt = 0.3 * 672.0
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_height_pt * inches_per_pt  # height in inches
fig_height = fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]
plt.rcParams.update(  # try to match font sizes of document
    {'axes.labelsize': 10,
     'text.fontsize': 10,
     'legend.fontsize': 10,
     'xtick.labelsize': 8,
     'ytick.labelsize': 8,
     'text.usetex': True,
     'figure.figsize': fig_size,
     'font.family': 'serif',
     'font.serif': ['palatino'],
     'savefig.dpi': 300
     })

f_low = 15.
f_high = 2048.

# True NR parameters
nrs1 = +0.84998783
nrs2 = +0.84947785
nrmc = 28.33690746
nret = 0.18752535
nrmt = nrmc / nret**0.6
nrm1, nrm2 = mchirp_eta_to_mass1_mass2(nrmc, nret)
nrq = nrm1 / eta_mass1_to_mass2(nret, nrm1)

nrwav = UseNRinDA.nr_waveform(
    filename=
    '/home/prayush/research/NR/EffectualnessStudies/NRwaveforms/SKS_d13.9-q3-sA_0_0_0.85_sB_0_0_0.85-Lev5/extracted-rhOverM_CcePITT_Asymptotic_GeometricUnits/CceR0420.dir/Y_l2_m2.dat',
    time_length=8)
nrwav.rescale_to_totalmass(nrmt)
nrhp = TimeSeries(nrwav.rescaled_hp, epoch=0)
nrhc = TimeSeries(nrwav.rescaled_hc, epoch=0)
nrh_max_amp_t, _ = nrwav.get_amplitude_peak()

# Get actual EOB waveform
hp0t, hc0t = get_td_waveform(approximant='SEOBNRv2',
                             mass1=nrm1,
                             mass2=nrm2,
                             spin1z=nrs1,
                             spin2z=nrs2,
                             distance=(nrm1 + nrm2) * lal.MRSUN_SI /
                             (1.e6 * lal.PC_SI),
                             delta_t=nrwav.dt,
                             f_lower=f_low - 1.)
hp0 = TimeSeries(np.zeros(nrwav.n),
                 delta_t=hp0t.delta_t,
                 dtype=hp0t.dtype,
                 epoch=0)
hc0 = TimeSeries(np.zeros(nrwav.n),
                 delta_t=hc0t.delta_t,
                 dtype=hc0t.dtype,
                 epoch=0)
hp0[:len(hp0t)] = hp0t * 0.5 * np.pi  # FIXME: Magic factors!!
hc0[:len(hc0t)] = hc0t * 0.5 * np.pi

# align at peak
hnrhp, hnrhc, hhp0, hhc0 = align_waveforms_amplitude_peak(nrhp, nrhc, hp0, hc0)

# align at 20 Hz
hnrhp, hnrhc, hhp0, hhc0 = align_waveforms_at_frequency(
    nrhp, nrhc, hp0, hc0, 20)
make_inspiral_merger_plot(
    hnrhp,
    hnrhc,
    hhp0,
    hhc0,
    legend=['NR', 'SEOBNRv2'],
    tscale=hnrhp.delta_t / nrmt / lal.MTSUN_SI,
    lp=0.96,
    rp=1.02,
    savefig='ExactParameters_LFAligned_q%.1f_sA%.2f_sB%.2f.png' %
    (nrq, nrs1, nrs2))

# Get BEST MATCH EOB waveform
s1 = 0.90702927
s2 = 0.93065105
mc = 28.238826
et = 0.14983673
mt = mc / et**0.6
m1, m2 = mchirp_eta_to_mass1_mass2(mc, et)

hp1t, hc1t = get_td_waveform(approximant='SEOBNRv2',
                             mass1=nrm1,
                             mass2=nrm2,
                             spin1z=nrs1,
                             spin2z=nrs2,
                             distance=(nrm1 + nrm2) * lal.MRSUN_SI /
                             (1.e6 * lal.PC_SI),
                             delta_t=nrwav.dt,
                             f_lower=f_low - 1.)
hp1 = TimeSeries(np.zeros(nrwav.n),
                 delta_t=hp1t.delta_t,
                 dtype=hp1t.dtype,
                 epoch=0)
hc1 = TimeSeries(np.zeros(nrwav.n),
                 delta_t=hc1t.delta_t,
                 dtype=hc1t.dtype,
                 epoch=0)
hp1[:len(hp1t)] = hp1t * 0.5 * np.pi  # FIXME: Magic factors!!
hc1[:len(hc1t)] = hc1t * 0.5 * np.pi

# Align the way match is maximized
psd = from_txt('/home/prayush/research/advLIGO_PSDs/ZERO_DET_high_P.dat',
               nrwav.n / 2 + 1, nrwav.df, f_low)

tsign = 1
phsign = -1
for tsign in [-1, 1]:
    for phsign in [-1, 1]:
        _, _, hhp1, hhc1 = align_waveforms_optimally(
            hnrhp,
            hnrhc,
            hp1,
            hc1,
            psd=psd,
            low_frequency_cutoff=f_low,
            high_frequency_cutoff=f_high,
            tsign=tsign,
            phsign=phsign)
        #
        make_inspiral_merger_plot(
            hnrhp,
            hnrhc,
            hhp1,
            hhc1,
            legend=['NR', 'SEOBNRv2'],
            tscale=hnrhp.delta_t / nrmt / lal.MTSUN_SI,
            lp=0.96,
            rp=1.02,
            savefig=
            'BestMatchParameters_LFAligned_q%.1f_sA%.2f_sB%.2f_%d_%d.png' %
            (nrq, nrs1, nrs2, tsign, phsign))
        #
        make_inspiral_merger_plot_3(
            hnrhp,
            hnrhc,
            hhp0,
            hhc0,
            hhp1,
            hhc1,
            legend=['NR', 'Same Params', 'Best Match'],
            tscale=hnrhp.delta_t / nrmt / lal.MTSUN_SI,
            lp=0.96,
            rp=1.02,
            savefig=
            'ExactBestMatchParameters_Aligned_q%.1f_sA%.2f_sB%.2f_%d_%d.png' %
            (nrq, nrs1, nrs2, tsign, phsign))
