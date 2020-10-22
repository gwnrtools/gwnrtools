#!/usr/bin/env python

import subprocess as cmd
import sys
import os
import glob
import numpy as np
import h5py

if sys.argv[1] == '-h':
    print("""\
######################################################
######################################################
**TestCompletionCceRuns.py:

######################################################
#1- TAkes the directory containing all CCE run directories

For each Lev? subdir in each run dir, it tests if the
run needs to be continued from where it stopped or not.
""")
    exit()

workdir = sys.argv[1]
os.chdir(workdir)
pwd = cmd.getoutput('pwd')
xx = glob.glob('CF_*')
yy = glob.glob('d*sA_*sB_*')
dirs = glob.glob('SKS_*')
dirs.extend(xx)
dirs.extend(yy)
#dirs = cmd.getoutput('ls %s | grep _d' % workdir).split()
subdirs = {}
for d in dirs:
    subdirs[d] = cmd.getoutput('ls %s/%s | grep Lev' % (workdir, d)).split()

for d in dirs:
    for ld in subdirs[d]:
        try:
            os.chdir('%s/%s' % (d, ld))
        except BaseException:
            continue
        print("\n\nInside %s" % cmd.getoutput('pwd'))
        out_files = cmd.getoutput('ls *.o????*').split()
        testn = 0
        for f in out_files:
            testn = max(
                testn,
                len(
                    cmd.getoutput(
                        r'cat %s | grep "Requested\ timestep\ not\ in\ worldtube\ data\ file.\ Stopping"'
                        % f).split()))
        print("Found ", testn)
        if testn != 0:
            print("This has finished!")
        dd = np.loadtxt(open('highResCce/Psi4_scri.L02Mp02.asc'))
        fin = h5py.File(glob.glob('CceR*.h5')[0], 'r')
        #
        tend_data = fin['DrLapse.dat'].value[-1][0]
        tend_cce = dd[-1][0]
        print("data ends at %.12f, cce ends at %.12f" % (tend_data, tend_cce))
        if abs(tend_data - tend_cce) > 10.:
            print("NOT finished!!")
        else:
            print("finished")
        os.chdir(pwd)
        continue
        try:
            qq = int(cmd.getoutput('tail *_%s.o* | grep Killing | wc -l' % ld))
            if qq != 1:
                print("Warning: This run might not have stopped yet!")
            else:
                print("This run is stopped. Continuation needed.")
        except BaseException:
            print("Unable to determine run status. continue.")
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
        print("Submitting new job!\n\n")
        comm = 'qsub -d `pwd` -N %s_%s_%d -l nodes=1:ppn=8,walltime=48:00:00 ./Submit.input >> sub.out' % (
            d, ld, nn)
        cmd.getoutput(comm)
        os.chdir(pwd)


def getSec(s):
    l = s.split(':')
    return int(l[0]) * 3600 + int(l[1]) * 60 + int(l[2])


class cce_run():
    # {{{
    def __init__(self, workdir=None, dirname=None, verbose=True):
        self.dirname = dirname
        self.workdir = workdir
        self.verbose = verbose
        self.subdirs = cmd.getoutput('ls %s/%s | grep Lev' %
                                     (workdir, dirname)).split()
        self.nn = len(glob.glob('highResCce-*.par')) + 1
        self.ofiles = {}
        for ld in self.subdirs:
            self.ofiles[ld] = self.get_recent_output_file(ld)
            pwd = cmd.getoutput('pwd')
            os.chdir('%s/%s/%s' % (self.workdir, self.dirname, ld))
            self.nn[ld] = len(glob.glob('highResCce-*.par')) + 1
            os.chdir(pwd)

    #

    def get_recent_output_file(self, ld):
        pwd = cmd.getoutput('pwd')
        os.chdir('%s/%s/%s' % (self.workdir, self.dirname, ld))
        ofiles = cmd.getoutput('ls *.o*').split('\n')
        max_mtime = 0.
        for ff in ofiles:
            mtime = os.stat(ff).st_mtime
            if mtime > max_mtime:
                max_mtime, ofile = mtime, ff
        os.chdir(pwd)
        return ofile

    #

    def is_to_be_continued(self, ld):
        pwd = cmd.getoutput('pwd')
        os.chdir('%s/%s/%s' % (self.workdir, self.dirname, ld))
        asked, used = cmd.getoutput('cat %s | grep walltime=' %
                                    self.ofiles[ld]).split('\n')
        usedtime = int(getSec(used.split('=')[-1]))
        askedtime = int(getSec(asked.split('=')[-1]))
        os.chdir(pwd)
        if usedtime < askedtime:
            if usedtime < 120:
                if self.verbose:
                    print("Warning: %s/%s Run stopped too soon!" %
                          (self.dirname, ld),
                          file=sys.stdout)
                return False
            else:
                if self.verbose:
                    print("Run might have completed", file=sys.stdout)
                #qq = int(cmd.getoutput('tail %s | grep Killing | wc -l' % ofile))
                return False
        else:
            if self.verbose:
                print("Run needs to be restarted", file=sys.stdout)
            return True

    #

    def setup_rerun(self, ld):
        # ASSUMPTION, ALL RUNS ARE CONTINUOUSLY NUMBERED
        pwd = cmd.getoutput('pwd')
        os.chdir('%s/%s/%s' % (self.workdir, self.dirname, ld))
        # Determine which is the next number
        nn = len(glob.glob('highResCce-*.par')) + 1
        #
        if self.verbose:
            print("Moving highResCce.par to highResCce-%d.par" % nn)
        cmd.getoutput('mv highResCce.par highResCce-%d.par' % nn)
        #
        if self.verbose:
            print("Copying highResCce-resubmit.par to highResCce.par")
        cmd.getoutput(
            'cp /home/p/pfeiffer/prayush/scratch/projects/CCE_modeldir/highResCce700-resubmit.par highResCce.par'
        )
        #
        if self.verbose:
            print("Moving highResCce to highResCce-%d" % nn)
        cmd.getoutput('mv highResCce highResCce-%d' % nn)
        cmd.getoutput('rm -rf highResCce-0')
        cmd.getoutput('ln -s highResCce-%d highResCce-0' % nn)
        os.chdir(pwd)

    #

    def submit_rerun(self, ld):
        pwd = cmd.getoutput('pwd')
        os.chdir('%s/%s/%s' % (self.workdir, self.dirname, ld))
        if self.verbose:
            print("Submitting new job!\n\n")
        comm = 'qsub -d `pwd` -N %s_%s_%d -l nodes=1:ppn=8,walltime=48:00:00 ./Submit.input >> sub.out' % (
            self.dirname, ld, nn)
        cmd.getoutput(comm)
        os.chdir(pwd)

    # }}}
