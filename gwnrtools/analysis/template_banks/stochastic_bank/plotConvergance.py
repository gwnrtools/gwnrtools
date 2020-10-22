#! /usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import os
import subprocess

import time
from time import gmtime, strftime
import datetime
from datetime import datetime

import string

import pylab as pb
import numpy as np
from numpy import sqrt

from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils

# setting some figure properties
# taken from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
fig_height_pt = 0.3 * 672.0
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_height_pt * inches_per_pt  # height in inches
#fig_height = fig_width * golden_mean      # height in inches
fig_size = [1.1 * fig_width, 1.1 * fig_height]
pb.rcParams.update(  # try to match font sizes of document
    {'axes.labelsize': 7,
   'text.fontsize': 7,
   'legend.fontsize': 8,
   'xtick.labelsize': 7,
   'ytick.labelsize': 7,
   'text.usetex': True,
   'figure.figsize': fig_size,
   'font.family': 'serif',
   'font.serif': ['palatino'],
   'savefig.dpi': 300
   })

NUMNEWPOINTS = 80

n = []
idx = []

print("Getting information about the banks\n\n")

i = 0
fname = 'banks/bank_%d.xml' % i
if not os.path.exists(fname):
    raise ValueError("Not even the 0th bank exists !")

n.append(
    np.int(subprocess.getoutput('ligolw_print -c mass1 %s | wc -l' % fname)))
idx.append(i)

print('new N = %d, new idx = %d' % (max(n), max(idx)))

FMT = '%H:%M:%S'
i += 1
fname = 'banks/bank_%d.xml' % i
while os.path.exists(fname):
    n.append(
        np.int(subprocess.getoutput('ligolw_print -c mass1 %s | wc -l' %
                                    fname)))
    idx.append(i)

    fname1 = 'banks/bank_%d.xml' % (i - 1)
    s1 = string.split(time.ctime(os.path.getmtime(fname1)))[3]
    fname2 = 'banks/bank_%d.xml' % i
    s2 = string.split(time.ctime(os.path.getmtime(fname2)))[3]
    tdelta = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
    print("\nTime taken to get the following bank was ", tdelta)
    print('new N = %d, new idx = %d' % (max(n), max(idx)))

    i += 1
    fname = 'banks/bank_%d.xml' % i

NUMBANKSMADE = i

s3 = string.split(strftime("%Y-%m-%d %H:%M:%S", gmtime()))[1]
tdelta = datetime.strptime(s3, FMT) - datetime.strptime(s2, FMT)
print("\nTime elapsed since the last bank was made = ", tdelta)

print("\nPlotting the NUBER OF POINTS vs NUMBER OF BANKS MADE\n\n")

pi = 0
pb.figure(pi)
pi += 1
pb.plot(idx, n, 'o-')
pb.xlabel('Number of banks made')
pb.ylabel('Number of points in the bank')
pb.savefig("plots/NumPointsNumBanks.png", dpi=300, format='png')

print("Calculating CONVERGENCE\n\n")
cvg = []
ncvg = []

for i in range(max(idx)):
    if i != max(idx):
        cvg.append((n[i + 1] - n[i]) / np.float64(NUMNEWPOINTS))
        ncvg.append(i + 1)
        print(('Acceptance while making bank %d is %f' %
               (ncvg[i], 100. * cvg[i])) + ' %')

print("\nPlotting the ACCEPTANCE RATIO vs THE BANK BEING MADE\n\n")

pb.figure(pi)
pi += 1
pb.semilogy(ncvg, cvg, 'o-')
pb.xlabel('Index of the bank being made')
pb.ylabel('Acceptance Ratio (per $%d$ steps)' % NUMNEWPOINTS)
pb.savefig('plots/ConVsNumBank.png', dpi=300, format='png')

print("\nPlotting the banks themselves\n\n")

for idx in range(NUMBANKSMADE):
    fname = 'banks/bank_%d.xml' % idx
    print("Reading bank %s" % fname)

    bdoc = ligolw_utils.load_filename(fname)
    btab = table.get_table(bdoc, lsctables.SimInspiralTable.tableName)

    m1 = []
    m2 = []
    mc = []
    et = []
    mt = []
    s1z = []
    s2z = []
    for p in btab:
        m1.append(max(p.mass1, p.mass2))
        m2.append(min(p.mass1, p.mass2))
        mt.append(p.mass1 + p.mass2)
        mc.append(p.mchirp)
        et.append(p.eta)
        s1z.append(p.spin1z)
        s2z.append(p.spin2z)

    print("Plotting bank %s" % fname)
    pb.figure(pi)
    pi += 1
    pb.plot(m1, m2, 'x')
    pb.xlim([min(m1) - .5, max(m1) + .5])
    pb.ylim([min(m2) - .5, max(m2) + .5])
    pb.grid()
    pb.title('Bank %d' % idx)
    pb.xlabel('Mass ($M_{\odot}$)')
    pb.ylabel('Mass ($M_{\odot}$)')
    pb.savefig('plots/bank_m1m2_%d.png' % idx, dpi=300, format='png')

    pb.figure(pi)
    pi += 1
    pb.plot(mc, et, 'x')
    pb.xlim([min(mc) - .5, max(mc) + .5])
    pb.ylim([min(et) - .01, max(et) + .01])
    pb.grid()
    pb.title('Bank %d' % idx)
    pb.xlabel('$\mathcal{M}_c$ ($M_{\odot}$)')
    pb.ylabel('$\eta$')
    pb.savefig('plots/bank_mcet_%d.png' % idx, dpi=300, format='png')

    pb.figure(pi)
    pi += 1
    pb.plot(mt, s1z, 'x')
    pb.xlim([0., max(mt) + .5])
    pb.ylim([min(s1z) - .05, max(s1z) + .05])
    pb.grid()
    pb.title('Bank %d' % idx)
    pb.xlabel('Total Mass ($M_{\odot}$)')
    pb.ylabel('$\mathrm{S}_{1,2}$')
    pb.savefig('plots/bank_mtotchi_%d.png' % idx, dpi=300, format='png')
