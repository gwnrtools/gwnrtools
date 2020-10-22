#!/usr/bin/env python
import sys
import os, logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',\
                     level=logging.INFO, stream=sys.stdout)
import time
_itime = time.time()

import argparse
import numpy as np
import glob

from glue import gpstime
from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import process as ligolw_process

#ctx = CUDAScheme()

__author__ = "Prayush Kumar <prayush@astro.cornell.edu>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])
#########################################################################
####################       Input parsing     #####################
#########################################################################
#{{{
parser = argparse.ArgumentParser(
    usage="%%prog [OPTIONS]",
    description="""
Reads in all match files for given testpoint set against current bank. Removes
all testpoints which have match > [MM] for at least one point in the existing
bank. Write the final testpoints as testpoints_sufficiently_far/test_%d.xml.
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# IO related inputs
parser.add_argument("--proposal-file-name",
                    dest="prop_file_name",
                    help="The current points file")
parser.add_argument("--new-proposal-file-name",
                    dest="new_prop_file_name",
                    help="The new points file")
parser.add_argument("--match-file-name-glob",
                    dest="match_file_name_glob",
                    help="glob for files that store matches")

# Physics related inputs
parser.add_argument("--minimal-match", dest="mm", default=0.97, type=float)
parser.add_argument("--elimination-dir",
                    metavar="STRING",
                    help="Where are hashes of eliminated proposals stored?",
                    default='testpoints_eliminated/')

# Miscellaneous
parser.add_argument("-V",
                    "--verbose",
                    action="store_true",
                    help="print extra debugging information",
                    default=False)
parser.add_argument("-C",
                    "--comment",
                    metavar="STRING",
                    help="add the optional STRING as the process:comment",
                    default='')

options = parser.parse_args()
#}}}

#########################################################################
#################### Opening input/output files/tables ##################
#########################################################################
logging.info("OPENING PROPOSAL FILE AND TABLES")
# Open the input proposals file and get the table
#{{{
if not options.prop_file_name:
    logging.info("No proposal points file-name given!")
    raise ValueError("No proposal points file-name given to %s" % PROGRAM_NAME)

if not os.path.exists(options.prop_file_name):
    logging.info("This proposal point file does not exist !")
    raise IOError(\
        "The proposal point file %s does not exist !" % options.prop_file_name)

logging.info("Opening proposals file %s" % options.prop_file_name)
prop_doc = ligolw_utils.load_filename(options.prop_file_name,
                                      contenthandler=table.use_in(
                                          ligolw.LIGOLWContentHandler),
                                      verbose=options.verbose)
try:
    prop_table = lsctables.SimInspiralTable.get_table(prop_doc)
except ValueError:
    raise IOError("Only sim_inspiral tables are understood for proposals..")

# Open the output proposals file to write new table
outdoc = ligolw.Document()
outdoc.appendChild(ligolw.LIGO_LW())
new_inspiral_table = lsctables.New(lsctables.SimInspiralTable,
                                   columns=prop_table.columnnames)
outdoc.childNodes[0].appendChild(new_inspiral_table)
out_proc_id = ligolw_process.register_to_xmldoc(
    outdoc, PROGRAM_NAME, options.__dict__, comment=options.comment).process_id
outname = options.new_prop_file_name
#}}}

sys.stdout.flush()


#########################################################################
####################### Functions to do things         ##################
#########################################################################
# Miscellaneous
def get_tag(wav):
    return str(wav.simulation_id.column_name)


def is_eliminated(wav):
    sid = get_tag(wav)
    if os.path.exists(os.path.join(options.elimination_dir, sid)): return True
    return False


def parse_match_file(mfile_name,
                     mvals_for_each_bank_point={},
                     mvals_for_each_test_point={}):
    if not os.path.exists(mfile_name):
        raise IOError("Provided file {} not found.".format(mfile_name))
    with open(mfile_name, 'r') as mfile:
        for line in mfile.readlines():
            line = line.split()
            btag, ptag = line[:2]
            if btag not in mvals_for_each_bank_point:
                mvals_for_each_bank_point[btag] = {}
            if ptag not in mvals_for_each_bank_point[btag]:
                mvals_for_each_bank_point[btag][ptag] = line[-1]
            if ptag not in mvals_for_each_test_point:
                mvals_for_each_test_point[ptag] = {}
            if btag not in mvals_for_each_test_point[ptag]:
                mvals_for_each_test_point[ptag][btag] = line[-1]
    return mvals_for_each_bank_point, mvals_for_each_test_point


def get_all_matches_against_point(p, mvals_dict):
    return np.array([mvals_dict[p][mval] for mval in mvals_dict[p]],
                    dtype=np.float128)


#########################################################################
#############################   Remove eliminated pts   #################
#########################################################################

##########################################################
### Note on algorithm to follow:-
## 0) match files are named as matches/match_A_B_C_D.dat, where:
##    - A := banks/bank_A.xml              (is the testpoints ID)
##    - B := banks_subparts/bank_B.xml  (is the subtestpoints ID)
##    - C := testpoints/test_A.xml               (is the bank ID)
##    - D := testpoints_subparts/test_A_B.xml (is the subbank ID)
## 1) options.match_file_name_glob will be "matches/matches_A_*.dat",
##    because we want to get matches against all testpoints and all bank tmplts.
## 2) Read in all match files indiscriminately with parse_match_file.
## 3) Recover matches for each testpoint with get_all_matches_against_point
## 4) Get Max values of these matches, for each testpoint respectively.
## 5) Append to new_inspiral_table all testpoints that have Max(Matches) < MM
##########################################################
## 1)
mfile_names = glob.glob(options.match_file_name_glob)
## 2)
mvals_for_each_bank_point = {}
mvals_for_each_test_point = {}
for mfile_name in mfile_names:
    mvals_for_each_bank_point, mvals_for_each_test_point =\
        parse_match_file(mfile_name, mvals_for_each_bank_point,\
            mvals_for_each_test_point)
## 3)
eliminated_testpoints = []
for tp in mvals_for_each_test_point:
    mvalues = get_all_matches_against_point(tp, mvals_for_each_test_point)
    ## 4)
    if np.max(mvalues) > options.mm:
        if options.verbose:
            logging.info("\t removing {}".format(tp))
        eliminated_testpoints.append(tp)
## 5)
for p in prop_table:
    ptag = get_tag(p)
    if ptag not in eliminated_testpoints and not is_eliminated(p):
        new_inspiral_table.append(p)

##########################################################
# write the xml doc to disk
proctable = table.get_table(outdoc, lsctables.ProcessTable.tableName)
proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(time.time())
ligolw_utils.write_filename(outdoc, outname)

cnt_eliminations = len(prop_table) - len(new_inspiral_table)
if options.verbose:
    logging.info("Written results to file: {}".format(outname))
    logging.info("Total {} test points eliminated, {} left.".format(
        cnt_eliminations, len(new_inspiral_table)))
    logging.info("Time taken: {} seconds".format(time.time() - _itime))

sys.stdout.flush()
