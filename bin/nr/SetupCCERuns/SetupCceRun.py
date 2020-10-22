#!/usr/bin/env python

import os
import sys

import numpy as np
import subprocess as cmd

if sys.argv[1] == '-h':
    print("""\
######################################################
######################################################
**SetupCceRun.py -- run_cce_from_data.py:

######################################################
#1-Takes in an one head directory name for a sim.
    Assumptions:-
    It has a CceData sub-directory containing Lev? subdirs,
    each with CCE data files.
#2- Takes in the directory in which to set up the runs.

Creates the output directory with the same name as head,
within the directory passed as the second argument.

# The idea is to give this script as input a
  """)
    exit()

indir = sys.argv[1]  # .strip('/')
outdir = sys.argv[-1]
pwd = cmd.getoutput('pwd')

pittnull = '/home/p/pfeiffer/prayush/src/cactus_cce/Cactus/exe/cactus_pittnull'
#
levdirs = cmd.getoutput('ls %s/CceData' % indir).split()
print(cmd.getoutput('pwd'))
for levdir in levdirs:
    # Create the output directory
    workdir = '%s/%s/%s' % (outdir, indir.split('/')[-1], levdir)
    os.makedirs(workdir)
    #
    # What H5 files are there ?
    h5Files = cmd.getoutput('ls %s/CceData/%s/*.h5' % (indir, levdir)).split()
    outer_radii = \
        np.array([int(h5Files[idx][-7:-3]) for idx in range(len(h5Files))])
    workfile = h5Files[np.where(outer_radii == outer_radii.max())[0][0]]
    os.chdir(workdir)
    print("workfile is %s" % workfile)
    cmd.getoutput('cp %s .' % pittnull)
    cmd.getoutput('ln -s %s' % (workfile))
    cmd.getoutput(
        'cp /home/p/pfeiffer/prayush/scratch/projects/CCE_modeldir/highResCce700.par highResCce.par'
    )
    h5File = cmd.getoutput('ls *.h5')
    os.makedirs('highResCce')
    os.chdir(pwd)
    # Create Submit.input file
    fout = open('%s/Submit.input' % workdir, "w+")
    fout.write('#!/bin/env sh\n')
    fout.write('export OMP_NUM_THREADS=1\n\n')
    fout.write('H5FILEin=%s\n' % h5File)
    fout.write('H5FILEout=Cce.h5\n')
    fout.write('PARFILE=highResCce.par\n')
    fout.write('RUNDIR=.\n\n')
    fout.write('cp -L ${RUNDIR}/${H5FILEin} /dev/shm/${H5FILEout}\n')
    fout.write(
        '/scinet/gpc/mpi/openmpi/1.4.4-intel-v12.1/bin/mpirun -np 8 ${RUNDIR}/cactus_pittnull ${RUNDIR}/${PARFILE}\n'
    )
    fout.close()
    #

###
print('mkdir -p %s/%s/%s' % (outdir, indir.split('/')[-1], levdir))
