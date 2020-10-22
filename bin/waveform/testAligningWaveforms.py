#!/usr/bin/env python

import UseNRinDA
from pycbc.waveform import amplitude_from_polarizations, phase_from_polarizations
from pycbc.pnutils import mchirp_eta_to_mass1_mass2
from pycbc.psd import *
from pycbc.waveform.utils import *
from pycbc.waveform import *
from pycbc.types import *
from pycbc.filter import *
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex': True})

#from aligningWaveforms import shift_waveform_phase_time


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
        hcnew = TimeSeries(-1 * amplitude * np.sin(phase),
                           epoch=hcnew._epoch,
                           delta_t=hcnew.delta_t,
                           dtype=hcnew.dtype)
    # First apply time shift
    print("shifting by %e secs" % t_shift)
    if t_shift != 0:
        idx_shift = int(np.round(t_shift / hpnew.delta_t))
        hpnew.roll(idx_shift)
        hcnew.roll(idx_shift)
        #hpnew._epoch = hpnew._epoch + t_shift
        #hcnew._epoch = hcnew._epoch + t_shift
    return hpnew, hcnew


def align_waveforms_optimally(h_plus1,
                              h_cross1,
                              hplus2,
                              hcross2,
                              psd=None,
                              low_frequency_cutoff=None,
                              high_frequency_cutoff=None,
                              tsign=-1,
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
    if psd is None or low_frequency_cutoff is None:
        raise IOError("Need psd to be input AND the low frequency cutoff!")
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
    t_shift = 1 * max_id * h_plus2.delta_t
    ph_shift = np.angle(snr[max_id])
    #
    hp2, hc2 = \
        shift_waveform_phase_time(
            h_plus2,
            h_cross2,
            tsign * t_shift,
            phsign * ph_shift)
    return h_plus1, h_cross1, hp2, hc2


f_low = 15

T = 8 * 2
sample_rate = 8192
N = T * sample_rate
n = N / 2 + 1

# Align the way match is maximized
psd = from_txt('/home/prayush/research/advLIGO_PSDs/ZERO_DET_high_P.dat', N,
               1. / T, f_low)

###################################
#hp1, hc1 = get_td_waveform(approximant='SEOBNRv1', mass1=50., mass2=50., spin1z=0.5, spin2z=-0.5, delta_t=1./sample_rate, f_lower=f_low)
#hp1._epoch = 0
#hc1._epoch = 0
# Generate sine wave instead
if True:
    T = 1. / 30.
    tvec = np.arange(0, 100 * T, 1. / sample_rate)
    hp1 = TimeSeries(np.cos(tvec * 2 * np.pi / T), delta_t=1. / sample_rate)
    hc1 = TimeSeries(-1 * np.sin(tvec * 2 * np.pi / T),
                     delta_t=1. / sample_rate)

orig_length = len(hp1)

h = TimeSeries(np.zeros(N),
               delta_t=hp1.delta_t,
               epoch=hp1._epoch,
               dtype=real_same_precision_as(hp1))
h[:len(hp1)] = hp1
hp1 = h
h = TimeSeries(np.zeros(N),
               delta_t=hc1.delta_t,
               epoch=hc1._epoch,
               dtype=real_same_precision_as(hc1))
h[:len(hc1)] = hc1
hc1 = h

#hp2, hc2 = get_td_waveform(approximant='SEOBNRv1', mass1=50., mass2=50., spin1z=0.5, spin2z=-0.25, delta_t=1./sample_rate, f_lower=f_low)
#hp2._epoch = 0
#hc2._epoch = 0
# Generate sine wave
if True:
    T = 1. / 30.
    tvec = np.arange(0, 100 * T, 1. / sample_rate)
    hp2 = TimeSeries(np.cos(tvec * 2 * np.pi / T), delta_t=1. / sample_rate)
    hc2 = TimeSeries(-1 * np.sin(tvec * 2 * np.pi / T),
                     delta_t=1. / sample_rate)

orig_length = max(orig_length, len(hp2))

h = TimeSeries(np.zeros(N),
               delta_t=hp2.delta_t,
               epoch=hp2._epoch,
               dtype=real_same_precision_as(hp2))
h[:len(hp2)] = hp2
hp2 = h
h = TimeSeries(np.zeros(N),
               delta_t=hc2.delta_t,
               epoch=hc2._epoch,
               dtype=real_same_precision_as(hc2))
h[:len(hc2)] = hc2
hc2 = h

# True NR parameters
nrs1 = 0
nrs2 = 0
nrmt = 1.2637662451913398e+02
nrq = 8.5
nret = nrq / (1. + nrq)**2
nrmc = nrmt * nret**0.6
nrm1, nrm2 = mchirp_eta_to_mass1_mass2(nrmc, nret)

#nrwav = UseNRinDA.nr_waveform(filename='/home/prayush/research/cita-papers/SEOBtesting/Paper1/plots/InvestigateSimulations/d11.5-q8.5-sA_0_0_0_sB_0_0_0/rhOverM_CcePITT_Asymptotic_GeometricUnits.h5', filetype='HDF', time_length=T)
#nrwav2 = UseNRinDA.nr_waveform(filename='/home/prayush/research/cita-papers/SEOBtesting/Paper1/plots/InvestigateSimulations/BBH_CFMS_d15.9_q3.50_sA_0_0_0_sB_0_0_0/rhOverM_CcePITT_Asymptotic_GeometricUnits.h5', filetype='HDF', time_length=T)
nrwav2 = UseNRinDA.nr_waveform(
    filename=
    '/home/prayush/research/cita-papers/SEOBtesting/Paper1/plots/InvestigateSimulations/SKS_d14.3-q3-sA_0_0_0.73132_sB_0_0_-0.85/rhOverM_CcePITT_Asymptotic_GeometricUnits.h5',
    filetype='HDF',
    time_length=T)

#nrwav2 = UseNRinDA.nr_waveform(filename='/home/prayush/research/cita-papers/SEOBtesting/Paper1/plots/InvestigateSimulations/d11.5-q8.5-sA_0_0_0_sB_0_0_0/rhOverM_Asymptotic_GeometricUnits.h5', filetype='HDF', cce=False, ex_order=3, time_length=T)
#nrwav = UseNRinDA.nr_waveform(filename='/home/prayush/research/cita-papers/SEOBtesting/Paper1/plots/InvestigateSimulations/BBH_CFMS_d15.9_q3.50_sA_0_0_0_sB_0_0_0/rhOverM_Asymptotic_GeometricUnits.h5', filetype='HDF', cce=False, ex_order=3, time_length=T)
nrwav = UseNRinDA.nr_waveform(
    filename=
    '/home/prayush/research/cita-papers/SEOBtesting/Paper1/plots/InvestigateSimulations/SKS_d14.3-q3-sA_0_0_0.73132_sB_0_0_-0.85/rhOverM_Asymptotic_GeometricUnits.h5',
    filetype='HDF',
    cce=False,
    ex_order=3,
    time_length=T)

# nrwav.rescale_to_totalmass(nrmt)
#nrhp = TimeSeries(nrwav.rescaled_hp, epoch=0)
#nrhc = TimeSeries(nrwav.rescaled_hc, epoch=0)
#nrh_max_amp_t,_ = nrwav.get_amplitude_peak()

# nrwav2.rescale_to_totalmass(nrmt)
#nrhp2 = TimeSeries(nrwav2.rescaled_hp, epoch=0)
#nrhc2 = TimeSeries(nrwav2.rescaled_hc, epoch=0)
#nrh_max_amp_t2,_ = nrwav2.get_amplitude_peak()

# Uncomment these
#hp1, hc1 = nrhp, nrhc
#hp2, hc2 = nrhp2, nrhc2

# Find max lengths
print("Finding max lengths")
for i in range(len(hp1)):
    if hp1[i] == 0 and hc1[i] == 0 and hp1[i + 1] == 0 and hc1[
            i + 1] == 0 and hp1[i + 2] == 0 and hc1[i + 2] == 0:
        break
i1 = i
for i in range(len(hp2)):
    if hp2[i] == 0 and hc2[i] == 0 and hp2[i + 1] == 0 and hc2[
            i + 1] == 0 and hp2[i + 2] == 0 and hc2[i + 2] == 0:
        break

orig_length = max(i, i1)
print("max length = %d, %f s" % (orig_length, orig_length / sample_rate))

################################################
htilde = make_frequency_series(hp1)
stilde = make_frequency_series(hp2)
_snr = zeros(N, dtype=complex_same_precision_as(htilde))

snr, corr, snr_norm = matched_filter_core(htilde,
                                          stilde,
                                          psd,
                                          f_low,
                                          None,
                                          None,
                                          out=_snr)
v2_norm = sigma(stilde, psd, f_low, None)

matchCplx = TimeSeries(snr.data * snr_norm / v2_norm,
                       dtype=complex_same_precision_as(snr),
                       delta_t=hp1.delta_t,
                       epoch=snr._epoch)

#thp1, thc1, thp2, thc2 = align_waveforms_optimally( hp1, hc1, hp2, hc2, psd=psd, low_frequency_cutoff=f_low)

# format the indices to be sampled, so they are denser near the peak
idx_length = 1 * orig_length
pltid = 0
low_mm = 0
for i in range(1400, idx_length, int(0.0005 / hp1.delta_t)):
    mCplx = matchCplx[i]
    #
    # just to reduce the number of plots
    if np.abs(mCplx) < 0.1:
        low_mm += 1
        if low_mm % 4 != 0:
            continue
    if pltid <= 41 or pltid >= 55:
        pltid += 1
        continue
    #
    t_shift = i * hp1.delta_t
    ph_shift = np.angle(snr[i])
    #
    #thp2, thc2 = shift_waveform_phase_time( hp2, hc2, 1 * t_shift, 1 * ph_shift)
    # print "\n+1 +1 overlaps = %f, %f" % (np.abs(overlap_cplx(hp1, thp2, psd=psd, low_frequency_cutoff=f_low)), np.abs(overlap_cplx(hc1, thc2, psd=psd, low_frequency_cutoff=f_low)))
    # print "match = ", match(hp1, thp2, psd=psd, low_frequency_cutoff=f_low)

    # Final choice -- optimal
    thp2, thc2 = shift_waveform_phase_time(hp2, hc2, -1 * t_shift,
                                           -1 * ph_shift)
    print(
        "-1 +1 overlaps = %f, %f" %
        (np.abs(overlap_cplx(hp1, thp2, psd=psd, low_frequency_cutoff=f_low)),
         np.abs(overlap_cplx(hc1, thc2, psd=psd, low_frequency_cutoff=f_low))))
    print("match = ", match(hp1, thp2, psd=psd, low_frequency_cutoff=f_low))

    #thp2, thc2 = shift_waveform_phase_time( hp2, hc2, 1 * t_shift, -1 * ph_shift)
    # print "+1 -1 overlaps = %f, %f" % (np.abs(overlap_cplx(hp1, thp2, psd=psd, low_frequency_cutoff=f_low)), np.abs(overlap_cplx(hc1, thc2, psd=psd, low_frequency_cutoff=f_low)))
    # print "match = ", match(hp1, thp2, psd=psd, low_frequency_cutoff=f_low)

    #thp2, thc2 = shift_waveform_phase_time( hp2, hc2, -1 * t_shift, -1 * ph_shift)
    # print "-1 -1 overlaps = %f, %f" % (np.abs(overlap_cplx(hp1, thp2, psd=psd, low_frequency_cutoff=f_low)), np.abs(overlap_cplx(hc1, thc2, psd=psd, low_frequency_cutoff=f_low)))
    # print "match = ", match(hp1, thp2, psd=psd, low_frequency_cutoff=f_low)
    # break
    # continue
    #
    fig = plt.figure(int(np.random.random() * 1e7))
    #
    ax = fig.add_subplot(211)
    ax.plot(matchCplx.sample_times, np.abs(matchCplx), 'k',
            matchCplx.sample_times[i], np.abs(mCplx), 'ro')
    ax.grid()
    ax.set_xlim(0 + np.float64(matchCplx._epoch),
                idx_length * hp1.delta_t + np.float64(matchCplx._epoch))
    #
    ax = fig.add_subplot(212)
    ax.plot(hp1.sample_times, hp1, thp2.sample_times, thp2)
    ax.legend(['Extp-N3', 'CCE'], loc='upper left')
    ax.grid()
    ax.set_xlim(
        0 + np.float64(min(hp1._epoch, thp2._epoch)),
        idx_length * hp1.delta_t + np.float64(max(hp1._epoch, thp2._epoch)))
    fig.savefig('plotsTesting/testplot%06d.png' % pltid, dpi=600)
    pltid += 1
