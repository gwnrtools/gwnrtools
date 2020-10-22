#!/usr/bin/env python

import subprocess as cmd
import sys
import os
import glob

if sys.argv[1] == '-h':
    print("""\
######################################################
######################################################
**ReSubmitCceRun.py -- resubmit_cce_sims.py:

######################################################
#1- TAkes the directory containing all CCE run directories

For each Lev? subdir in each run dir, it firsts tests if the
run needs to be continued from where it stopped or not.
""")
    exit()

workdir = sys.argv[1]
os.chdir(workdir)
pwd = cmd.getoutput('pwd')
#xx = glob.glob('CF_d15.8-q3-sA_0_0_-0.3_sB_0_0_-0.3')
#xx.extend( glob.glob('CF_d16.*'))
#xx.extend( glob.glob('CF_d17.*'))
dirs = glob.glob('d*')
#dirs.extend( xx )
#dirs = cmd.getoutput('ls %s | grep _d' % workdir).split()
subdirs = {}
for d in dirs:
    subdirs[d] = cmd.getoutput('ls %s/%s | grep Lev' % (workdir, d)).split()


def getSec(s):
    l = s.split(':')
    return int(l[0]) * 3600 + int(l[1]) * 60 + int(l[2])


for d in dirs:
    for ld in subdirs[d]:
        try:
            os.chdir('%s/%s' % (d, ld))
        except BaseException:
            continue
        print("\n\nInside %s" % cmd.getoutput('pwd'))
        # Check if the run has actually stopped
        try:
            ofiles = cmd.getoutput('ls *.o*').split('\n')
            max_mtime = 0.
            for ff in ofiles:
                mtime = os.stat(ff).st_mtime
                if mtime > max_mtime:
                    max_mtime = mtime
                    ofile = ff
            print("Checking outfile = %s" % ofile, file=sys.stderr)
            try:
                asked, used = cmd.getoutput('cat %s | grep walltime=' %
                                            ofile).split('\n')
                usedtime = int(getSec(used.split('=')[-1]))
                askedtime = int(getSec(asked.split('=')[-1]))
                if usedtime < askedtime:
                    #qq = int(cmd.getoutput('tail *_%s.o* | grep Killing | wc -l' % ld))
                    print("Warning: %s/%s run might not have stopped yet!" %
                          (d, ld),
                          file=sys.stdout)
                    askuser = eval(
                        input('Should we re-submit this run ([y]/n): '))
                    if 'y' not in askuser:
                        continue
                else:
                    print("%s run is incomplete. Execution time=%d" %
                          (ld, usedtime),
                          file=sys.stderr)
            except BaseException:
                print("Check did not work.Restarting anyway", file=sys.stderr)
        except BaseException:
            askuser = eval(
                input(
                    'Check indeterminate. Should we re-submit this run ([y]/n):'
                ))
            # if 'y' not in askuser: continue
        # Determine which is the next number
        nn = len(glob.glob('highResCce-*.par')) + 1
        #
        print("Moving highResCce.par to highResCce-%d.par" % nn)
        cmd.getoutput('mv highResCce.par highResCce-%d.par' % nn)
        #
        print("Copying highResCce-resubmit.par to highResCce.par")
        cmd.getoutput(
            'cp /home/p/pfeiffer/prayush/scratch/projects/CCE_modeldir/highResCce700-resubmit.par highResCce.par'
        )
        #
        print("Moving highResCce to highResCce-%d" % nn)
        cmd.getoutput('mv highResCce highResCce-%d' % nn)
        cmd.getoutput('rm -rf highResCce-0')
        cmd.getoutput('ln -s highResCce-%d highResCce-0' % nn)
        #
        # print "Appending the magic lines to par file"
        # cmd.getoutput("printf '%s\n%s\n' 'IO::recover = auto' 'IO::recover_dir = /path/to/output/from/last/Cce/run"' >>")
        #
        # print "Submitting new job!\n\n"
        #comm = 'qsub -d `pwd` -N %s_%s_%d -l nodes=1:ppn=8,walltime=48:00:00 ./Submit.input >> sub.out' % (d,ld,nn)
        # cmd.getoutput(comm)
        os.chdir(pwd)
