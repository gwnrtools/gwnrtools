#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

NUM_OUTER_ITER = 100
NUM_TEST_BANKS = 200
NUM_SUB_BANKS_INITIAL = 40
NUM_SUB_BANKS_FINAL = 40
NUM_NEW_POINTS = 100
MM = 0.97

#########################################################################
########################## Write the DAG ###############################
#########################################################################
iid = 1
dagfilename = "banksim_%d.dag" % iid
dagfile = open(dagfilename, "w+")

tpid = 0
bnkid = 0

for iter in range(NUM_OUTER_ITER):
    # Write the testpoint-creation jobs that all run in the starting in parallel
    for tp in range(NUM_TEST_BANKS):
        dagfile.write("Job TEST%d script1.submit\n" % tpid)
        dagfile.write("VARS TEST%d ID=\"%d\"\n" % (tpid, tpid))
        dagfile.write("VARS TEST%d NUMNEWPOINTS=\"%d\"\n" %
                      (tpid, NUM_NEW_POINTS))
        dagfile.write("VARS TEST%d MM=\"%f\"\n" % (tpid, MM))
        if iter > 0:
            dagfile.write("PARENT COM%d CHILD TEST%d\n" % (bnkid - 1, tpid))
        dagfile.write("Retry TEST%d 5\n" % tpid)
        dagfile.write("PRIORITY TEST%d 1000\n\n" % tpid)
        tpid += 1

    for iid in range(NUM_TEST_BANKS):
        if bnkid != 0:
            NUM_SUB_BANKS = NUM_SUB_BANKS_FINAL
        else:
            NUM_SUB_BANKS = NUM_SUB_BANKS_INITIAL
        # Write the parent job to all sub-bank jobs. This is the one running script2.py
        dagfile.write("Job SPL%d script2.submit\n" % bnkid)
        dagfile.write("VARS SPL%d ID=\"%d\"\n" % (bnkid, bnkid))
        dagfile.write("VARS SPL%d NUMSUBBANKS=\"%d\"\n" %
                      (bnkid, NUM_SUB_BANKS))
        if iid > 0:
            dagfile.write("PARENT COM%d CHILD SPL%d\n" % (bnkid - 1, bnkid))
        else:
            for tp in range(tpid - NUM_TEST_BANKS, tpid):
                dagfile.write("PARENT TEST%d CHILD SPL%d\n" % (tp, bnkid))
        dagfile.write("Retry SPL%d 5\n" % bnkid)
        dagfile.write("PRIORITY SPL%d 100\n\n" % bnkid)

        # Write the child job to all sub-bank jobs. This is the one running script4.py
        dagfile.write("Job COM%d script4.submit\n" % bnkid)
        dagfile.write("VARS COM%d ID=\"%d\"\n" % (bnkid, bnkid))
        dagfile.write("VARS COM%d NUMSUBBANKS=\"%d\"\n" %
                      (bnkid, NUM_SUB_BANKS))
        dagfile.write("VARS COM%d MM=\"%f\"\n" % (bnkid, MM))
        dagfile.write("Retry COM%d 5\n" % bnkid)
        dagfile.write("PRIORITY COM%d 10\n\n" % bnkid)

        # Loop over NUM_SUB_BANKS different banks. Put a Job for each sub-bank. These
        # run script3.py
        for idx in range(NUM_SUB_BANKS):
            dagfile.write("\tJob JOB%d-%d script3.submit\n" % (bnkid, idx))
            dagfile.write("\tVARS JOB%d-%d ID=\"%d\"\n" % (bnkid, idx, bnkid))
            dagfile.write("\tVARS JOB%d-%d PARTID=\"%d\"\n" %
                          (bnkid, idx, idx))
            dagfile.write("\tPARENT JOB%d-%d CHILD COM%d\n" %
                          (bnkid, idx, bnkid))
            dagfile.write("\tPARENT SPL%d CHILD JOB%d-%d\n" %
                          (bnkid, bnkid, idx))
            dagfile.write("\tRetry JOB%d-%d 5\n" % (bnkid, idx))
            dagfile.write("\tPRIORITY JOB%d-%d %d\n\n" % (bnkid, idx, idx))

        bnkid += 1

dagfile.close()

#########################################################################
########################### Submit the DAG #############################
#########################################################################

#commands.getoutput("ssh sugar.phy.cita.utoronto.ca")
#commands.getoutput("condor_submit_dag %s > dag_%d.out 2>dag_%d.err" % (dagfilename, iid, iid)
