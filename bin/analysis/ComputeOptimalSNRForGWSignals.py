#!/usr/bin/env python
from pycbc.waveform import *
from pycbc.types import *
from pycbc.psd import *
from pycbc.filter import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--m1', type=float)
parser.add_argument('--m2', type=float)
parser.add_argument('--s1z', default=0., type=float)
parser.add_argument('--s2z', default=0., type=float)
parser.add_argument('--dist',
                    default=1.e3,
                    type=float,
                    help="distance to source in MegaParsecs")
parser.add_argument(
    '--incl',
    default=0.,
    type=float,
    help="inclination angle between orbital-L and line of sight")

parser.add_argument('--f-lower', type=float)
parser.add_argument('--approx', default='SEOBNRv2', type=str)
parser.add_argument('--delta-t', default=1. / 4096., type=float)
parser.add_argument('--delta-f', default=1. / 128., type=float)

args = parser.parse_args()

# INITIALIZE
approx = args.approx
f_lower = args.f_lower
dt = args.delta_t
df = args.delta_f
time_length = int(1. / df)
sample_rate = int(1. / dt)
N = time_length * sample_rate
n = N / 2 + 1

m1 = args.m1
m2 = args.m2
s1z = args.s1z
s2z = args.s2z
dist = args.dist
incl = args.incl

# GET WAVEFORM
if approx in td_approximants():
    hp, hc = get_td_waveform(approximant=approx,
                             mass1=m1,
                             mass2=m2,
                             spin1z=s1z,
                             spin2z=s2z,
                             distance=dist,
                             inclination=incl,
                             f_lower=f_lower,
                             delta_t=dt)
    time_length = len(hp) * dt
    df = 1. / time_length
    n = len(hp) / 2 + 1
    # print len(hp), time_length, df, n, time_length * sample_rate / 2 + 1
elif approx in fd_approximants():
    hp, hc = get_fd_waveform(approximant=approx,
                             mass1=m1,
                             mass2=m2,
                             spin1z=s1z,
                             spin2z=s2z,
                             distance=dist,
                             inclination=incl,
                             f_lower=f_lower,
                             delta_f=df)

# GET PSD
psd = from_string('aLIGOZeroDetHighPower', n, df, f_lower)

# COMPUTE OPTIMAL SNR
h = hp  # REPLACE WITH h = Fplus * hp + Fcross * hc
opt_snr = sigma(h, psd=psd, low_frequency_cutoff=f_lower)

print("Optimal SNR = %f" % (opt_snr))
