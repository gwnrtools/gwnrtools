#!/usr/bin/env python
# Copyright (C) 2014 Prayush Kumar, Heather Fong
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

import ccerun as CC
import sys
import os
import subprocess as cmd
import imp
sys.path.append('/home/prayush/src/UseNRinDA/scripts/setupCCEruns/')
sys.path.append('/home/p/pfeiffer/prayush/src/UseNRinDA/scripts/setupCCEruns/')
try:
    cmd.getoutput('module load git')
    head_dir = cmd.getoutput('git rev-parse --show-toplevel')
    sys.path.append(os.path.join(head_dir, 'scripts/setupCCEruns/'))
except BaseException:
    print("adding path to UseNRinDA using git rev-parse failed.. :(")

imp.reload(CC)

if sys.argv[1] == '-h':
    print("""\
######################################################
######################################################
**PostProcessOneRunRunAtOneLev.py

######################################################
#1- FULL path to data directory within which lie
    subdirectories with names Lev?

#2- FULL path to simulation directory within which lie
     subdirectories with names Lev?

#3- Lev, e.g. Lev3 Lev4 Lev5
""")
    exit()

#datadir = '/prayush/NR/CCE_2/SKS_d16.6-q3-sA_0_0_-0.6_sB_0_0_-0.4/'
#outdir = datadir
levdirs = ['Lev5']

datadir = sys.argv[1]
outdir = sys.argv[2]
levdirs = sys.argv[3:]
if not isinstance(levdirs, list):
    levdirs = [levdirs]

# Initialize container class for each Lev of the simulation
crun = {}
for ld in levdirs:
    ld_datadir = os.path.join(datadir, ld)
    ld_outdir = os.path.join(outdir, ld)
    datafile = cmd.getoutput('/bin/ls %s/ | grep .h5 | grep CceR' % ld_outdir)
    #
    print(ld_datadir, "\n", ld_outdir, "\n", datafile)
    crun[ld] = CC.cce_run(datafile=datafile,
                          datadir=ld_datadir,
                          pittnull=os.path.join(ld_datadir, datafile),
                          outdir=ld_outdir,
                          post_process_only=True,
                          verbose=True)

# Combine different segments of CCE
for ld in levdirs:
    tbc = crun[ld].is_to_be_continued_2()
    if tbc == True:
        print("At %s, CONTINUE to segmend %d" % (ld, crun[ld].nn - 1))
    else:
        print("At %s, STOP AFTER segmend %d" % (ld, crun[ld].nn - 1))

print("\n\n")
