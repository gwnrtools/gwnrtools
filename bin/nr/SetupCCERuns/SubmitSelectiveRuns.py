#!/usr/bin/env python

import ccerun as CC
import sys
import os
import subprocess as cmd
from glob import glob
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
print("Using: ", CC.__file__)

# Get the simulation directory
dir = sys.argv[1]
datadir = sys.argv[2]
outdir = sys.argv[3]

# List the Lev sub-directories to probe
levdirs = ['Lev3', 'Lev4', 'Lev5']

# File to write names of simulations for which
# the first segment is still ongoing
fout = open('list_ongoing_1', 'w')

# Current working directory, relative to which
# one can provide dir, for instance.
PWD = cmd.getoutput('pwd')

# Initialize container class for each Lev of the simulation
crun = {}
NUM_JOBS_STARTED = 0

for ld in levdirs:
    os.chdir(PWD)
    os.chdir('%s/%s' % (dir, ld))
    #
    if len(glob('*-1')) == 0:
        print("NOT EVEN the first segment RUN.. PASSING")
        # os.chdir(PWD)
        # continue
        subfile = glob('*-1.input')[0]
        comm = '/opt/torque/bin/qsub -d %s -N %s ./%s >> sub.out' %\
            (os.path.join(PWD, dir, ld), dir + ld, subfile)
        print("Running command %s\n" % comm)
        cmd.getoutput(comm)
        NUM_JOBS_STARTED += 1
        fout.write('%s/%s\n' % (dir, ld))
        fout.flush()
    elif len(glob('*-1/Psi4_scri.L08Mm02.asc')) == 0:
        print("No Psi4 in segment ONE.. PASSING")
        # continue
        comm = 'rm -rf *-1 *-2*'
        cmd.getoutput(comm)
        print("removed empty *-1 segment")
        #
        subfile = glob('*-1.input')[0]
        comm = '/opt/torque/bin/qsub -d %s -N %s ./%s >> sub.out' %\
            (os.path.join(PWD, dir, ld), dir + ld, subfile)
        print("Running command %s\n" % comm)
        cmd.getoutput(comm)
        #
        NUM_JOBS_STARTED += 1
        fout.write('%s/%s\n' % (dir, ld))
        fout.flush()
    else:
        ld_datadir = os.path.join(datadir, dir, ld)
        ld_outdir = os.path.join(outdir, dir, ld)
        datafile = cmd.getoutput('/bin/ls %s/ | grep .h5 | grep CceR' %
                                 ld_outdir)
        #
        print(ld_datadir, "\n", ld_outdir, "\n", datafile)
        crun[ld] = CC.cce_run(datafile=datafile,
                              datadir=ld_datadir,
                              pittnull=os.path.join(ld_datadir, datafile),
                              outdir=ld_outdir,
                              post_process_only=True,
                              verbose=True)
        print("%s for run %s has completed first segment already" % (dir, ld))
        if len(glob('*-?')) == 1:
            print("ONLY ONE segment has run.. CONTINUING")
            comm = 'python %s' % glob('CceR*continuation.py')[0]
            print("Running command (could have completed too!) %s\n" % comm)
            comout_ = cmd.getoutput(comm)
            print(comout_)
            #
            NUM_JOBS_STARTED += 1
            fout.write('%s/%s\n' % (dir, ld))
            fout.flush()
        elif crun[ld].is_to_be_continued_2():
            print("Continuing...!")
            comm = 'python %s' % glob('CceR*continuation.py')[0]
            print("Running command (could have completed too!) %s\n" % comm)
            comout_ = cmd.getoutput(comm)
            print(comout_)
            #
            NUM_JOBS_STARTED += 0.5
            fout.write('%s/%s\n' % (dir, ld))
            fout.flush()
        else:
            print(" .. and it has COMPLETED!")
            continue
    #
    os.chdir(PWD)

# Close the file with list of simulations with
# ongoing first segment
fout.close()
print("STARTED %d JOBS..!" % NUM_JOBS_STARTED)
