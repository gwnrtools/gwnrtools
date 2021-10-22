#! /usr/bin/env python

from pycbc.filter import match

import matplotlib
matplotlib.use('Agg')
import numpy as np

import sys
import os
import time
import subprocess

from optparse import OptionParser

from glue import gpstime

from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import process as ligolw_process
from glue import git_version

__author__ = "Prayush Kumar <prayush.kumar@gmail.com>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])
#################### Input parsing #####################
#{{{
parser = OptionParser(
    version=git_version.verbose_msg,
    usage="%prog [OPTIONS]",
    description=
    "Takes in the old bank file, the proposals file, and the match file, and creates the new bank file by prepending the old bank file with points in the proposals file that have matches below the MM."
)

parser = OptionParser()
parser.add_option("--iteration-id",
                  help="The index of the iteration",
                  type=int,
                  dest="iid")
parser.add_option("--num-sub-banks", help="The number of sub-banks", type=int)
parser.add_option('-m',
                  '--minimal-match',
                  metavar='MM',
                  help='minimal match',
                  default=0.95,
                  type=float,
                  dest="MM")

parser.add_option("-C",
                  "--comment",
                  metavar="STRING",
                  help="add the optional STRING as the process:comment",
                  default='')
parser.add_option("-V",
                  "--verbose",
                  action="store_true",
                  help="print extra debugging information",
                  default=False)

(options, args) = parser.parse_args()
#}}}

##############################################################################
######################### Get the PROPOSALS file and the table ###############
##############################################################################
################### Read in the proposal file, and get the sim table ############
# Get the proposal file
#{{{
if options.iid is not None:
    prop_file_name = "testpoints/test_%d.xml" % np.int(options.iid)
    iid = options.iid
else:
    idx = 0
    name1 = "testpoints/test_%d.xml" % idx
    idx += 1
    name2 = "testpoints/test_%d.xml" % idx
    while os.path.exists(name2):
        print(name1, name2)
        idx += 1
        name1 = name2
        name2 = "testpoints/test_%d.xml" % idx
    prop_file_name = name1
    iid = idx - 1

if not os.path.exists(prop_file_name):
    print("The prop file does not seem to exist or is named incorrectly")
    raise ValueError(
        "The prop-file %s does not exist. The value of iid passed was %d" %
        (prop_file_name, iid))

print("iid = %d" % iid)

in_prop_doc = ligolw_utils.load_filename(prop_file_name, options.verbose)
try:
    in_prop_table = table.get_table(in_prop_doc,
                                    lsctables.SimInspiralTable.tableName)
except ValueError:
    in_prop_table = table.get_table(in_prop_doc,
                                    lsctables.SnglInspiralTable.tableName)

#}}}

##############################################################################
########################### proposals match files ############################
##############################################################################
print("Opening proposals files")
matches = np.zeros(len(in_prop_table))
NUM_NEW_POINTS = 0

if options.num_sub_banks:
    num_sub_banks = options.num_sub_banks
else:
    idx = 0
    matfile_name = "matches/match_%d_part_%d.dat" % (iid, idx)
    while os.path.exists(matfile_name):
        idx += 1
        matfile_name = "matches/match_%d_part_%d.dat" % (iid, idx)

    num_sub_banks = idx

print("Total %d sub-bank match files" % num_sub_banks)
sys.stdout.flush()

for idx in range(num_sub_banks):
    filename = "matches/match_%d_part_%d.dat" % (iid, idx)
    print("Reading file %s" % filename)
    #sys.stdout.flush()
    matfile = open(filename, "r")
    data = np.loadtxt(matfile)
    matfile.close()
    for i in range(len(data)):
        if matches[i] < data[i]:
            matches[i] = data[i]

matfile_name = "matches/match_%d.dat" % iid
matfile = open(matfile_name, "w")
print("MM = %f" % options.MM)
sys.stdout.flush()
for match in matches:
    matfile.write("%12.18f\n" % match)
    if match < options.MM:
        NUM_NEW_POINTS += 1

matfile.close()

##############################################################################
####################### Open the OLD and NEW bank files, and write ###########
##############################################################################
old_bank_filename = "banks/bank_%d.xml" % iid
new_bank_filename = "banks/bank_%d.xml" % (iid + 1)

if NUM_NEW_POINTS == 0:
    print("No New Points. Copying the old bank file to the new name")
    subprocess.getoutput("cp %s %s" % (old_bank_filename, new_bank_filename))
else:
    print("%d New points to be added in the new file" % NUM_NEW_POINTS)
    # Open the Old bank file
    old_bank_doc = ligolw_utils.load_filename(old_bank_filename,
                                              options.verbose)
    try:
        old_bank_table = table.get_table(old_bank_doc,
                                         lsctables.SimInspiralTable.tableName)
    except ValueError:
        old_bank_table = table.get_table(old_bank_doc,
                                         lsctables.SnglInspiralTable.tableName)
    # Create the New bank file
    new_bank_doc = ligolw.Document()
    new_bank_doc.appendChild(ligolw.LIGO_LW())

    out_proc_id = ligolw_process.register_to_xmldoc(
        new_bank_doc,
        PROGRAM_NAME,
        options.__dict__,
        comment=options.comment,
        version=git_version.id,
        cvs_repository=git_version.branch,
        cvs_entry_time=git_version.date).process_id

    new_bank_table = lsctables.New(lsctables.SimInspiralTable,
                                   columns=[
                                       'mass1', 'mass2', 'mchirp', 'eta',
                                       'spin1x', 'spin1y', 'spin1z', 'spin2x',
                                       'spin2y', 'spin2z', 'inclination',
                                       'polarization', 'latitude', 'longitude',
                                       'bandpass', 'alpha', 'alpha1', 'alpha2',
                                       'process_id'
                                   ])
    new_bank_doc.childNodes[0].appendChild(new_bank_table)

    # Write to the New bank table
    for point in old_bank_table:
        point.process_id = out_proc_id
        new_bank_table.append(point)

    idx = 0
    for point in in_prop_table:
        if matches[idx] < options.MM:
            point.process_id = out_proc_id
            new_bank_table.append(point)

        idx += 1

    # Write the New bank to disk
    new_bank_proctable = table.get_table(new_bank_doc,
                                         lsctables.ProcessTable.tableName)
    new_bank_proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(time.time())
    ligolw_utils.write_filename(new_bank_doc, new_bank_filename)
