#! /usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import sys
import os
import time

from optparse import OptionParser

from glue import gpstime

from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import process as ligolw_process
from glue import git_version

__author__ = "Prayush Kumar <prkumar@cita.utoronto.ca>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])

#########################################################################
#################### Input parsing #####################
#########################################################################
#{{{
parser = OptionParser(
    version=git_version.verbose_msg,
    usage="%prog [OPTIONS]",
    description=
    "Takes in the iteration id. Reads in bank_id.xml. Writes bank_id_part_pid.xml, where each of these part files have bank-batch-size consecutive elements of the bank_id.xml. pid is part-id."
)

parser = OptionParser()
parser.add_option("--input-bank-file", help="Input bank file", type=str)
parser.add_option("--output-bank-file", help="Input bank file", type=str)

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

#########################################################################
################### Read in the bank file, and get the sim table ############
#########################################################################
# Get the bank file
#{{{
if options.input_bank_file is None or not os.path.exists(
        options.input_bank_file):
    raise ValueError("The bank file %s does not exist" %
                     options.input_bank_file)
else:
    in_bank_file = options.input_bank_file

in_bank_doc = ligolw_utils.load_filename(in_bank_file)
try:
    in_bank_table = table.get_table(in_bank_doc,
                                    lsctables.SimInspiralTable.tableName)
except ValueError:
    in_bank_table = table.get_table(in_bank_doc,
                                    lsctables.SnglInspiralTable.tableName)

#}}}

#########################################################################
############################ Write the output bank ######################
#########################################################################
if options.output_bank_file is None:
    raise ValueError("Out file name not given")
else:
    out_bank_file = options.output_bank_file

print("Writing output file %s" % out_bank_file)

out_bank_doc = ligolw.Document()
out_bank_doc.appendChild(ligolw.LIGO_LW())
out_proc_id = ligolw_process.register_to_xmldoc(
    out_bank_doc,
    PROGRAM_NAME,
    options.__dict__,
    comment=options.comment,
    version=git_version.id,
    cvs_repository=git_version.branch,
    cvs_entry_time=git_version.date).process_id
out_bank_table = lsctables.New(lsctables.SimInspiralTable,
                               columns=[
                                   'mass1', 'mass2', 'mchirp', 'eta', 'spin1x',
                                   'spin1y', 'spin1z', 'spin2x', 'spin2y',
                                   'spin2z', 'inclination', 'polarization',
                                   'latitude', 'longitude', 'bandpass',
                                   'alpha', 'alpha1', 'alpha2', 'process_id'
                               ])
out_bank_doc.childNodes[0].appendChild(out_bank_table)

## Write the input bank
for bank_point in in_bank_table:
    if bank_point.spin1z < -0.9:
        bank_point.spin1z = -0.9
    elif bank_point.spin1z > 0.9:
        bank_point.spin1z = 0.9
    if bank_point.spin2z < -0.9:
        bank_point.spin2z = -0.9
    elif bank_point.spin2z > 0.9:
        bank_point.spin2z = 0.9
    out_bank_table.append(bank_point)

#print "sub-bank file %s has %d points" % (subfile_name,len(out_subbank_table))
bank_proctable = table.get_table(out_bank_doc,
                                 lsctables.ProcessTable.tableName)
bank_proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(time.time())
ligolw_utils.write_filename(out_bank_doc, out_bank_file)
