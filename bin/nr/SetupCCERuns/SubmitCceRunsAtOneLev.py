#!/usr/bin/env python

import subprocess as cmd
import sys
import os
import ccerun

if sys.argv[1] == '-h':
    print("""\
######################################################
######################################################
**SubmitCceRunsAtOneLev.py

######################################################
#1- Directory containing all BBH Worldtube data, for each simulation within
     its own subdirectory
#2- Takes in 'Lev3', 'Lev5' etc

Submits EACH of the runs within PWD

Assumptions:-
  run directories have names with CF_ and SKS_ in them.

""")
    exit()

SUBMIT = False
#SCRIPT = os.path.join(os.path.dirname(sys.argv[0]), 'SetupCceRun.py')
PWD = cmd.getoutput('pwd')

#LEVDIRS = ['Lev3']
#LEVDIRS = ['Lev4']
#LEVDIRS = ['Lev5']

workdir = sys.argv[1]
LEVDIRS = [sys.argv[2]]

dirs = cmd.getoutput('ls %s | grep CF_d' % workdir).split()
dirs2 = cmd.getoutput('ls %s | grep SKS_d' % workdir).split()
for d in dirs2:
    dirs.append(d)

print(("Going to submit runs for ", dirs))
print("\n\n\n")

xx = {}
for d in dirs:
    print(("""
  # Initialize container class for %s
  """ % d))
    xx[d] = ccerun.nr_run_cce(datadir=os.path.join(workdir, d),
                              outdir=os.path.join(PWD, d))
    #
    print("""
  # Loop over levels to treat each individually
  """)
    for ld in LEVDIRS:
        print("""
    # SUBMIT the run to the local queue
    """)
        xx[d].submit_highestR_run_at_lev(ld, num_runs=2)
        #xx[d].setup_highestR_run_at_lev( ld, num_runs=2, submit=True )
        print("\n\n")

    #comm = 'python ~/src/scripts/run_cce_from_data.py /scratch/p/pfeiffer/prayush/RunsForCce/%s .' % d
    #comm = 'python %s %s/%s .' % (SCRIPT, workdir, d)
    #ret = cmd.getoutput(comm)
    # print "Status for %s: %s" % (d, ret)
