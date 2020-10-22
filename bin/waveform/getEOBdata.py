#!/usr/bin/env python

from pycbc.waveform import *
from pycbc.pnutils import *
from numpy import *

waveforms = []

for q in arange(1, 10.5, 0.5):
    M = 30.
    eta = q / (1. + q)**2
    m1, m2 = mtotal_eta_to_mass1_mass2(M, eta)
    hp, hc = get_td_waveform(approximant='SEOBNRv2',
                             mass1=m1,
                             mass2=m2,
                             f_lower=15,
                             delta_t=1. / 4096)
    waveforms.append([m1, m2, hp, hc])
    print('generated wveform for q = %f' % q)

for m1, m2, hp, hc in waveforms:
    savetxt('m%.2f_m%.2f_q%.1f_f15.txt' % (m1, m2, m1 / m2),
            list(zip(hp.sample_times.data, hp.data, hc.data)),
            fmt='%.18e\t%.18e\t%.18e')
    print('saved wveform for q = %f' % (m1 / m2))
