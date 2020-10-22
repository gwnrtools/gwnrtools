#! /usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np

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
parser.add_option("--iteration-id",
                  help="The index of the iteration",
                  type=int,
                  dest="iid")
parser.add_option("--bank-batch-size",
                  help="No of bank points in each sub-job",
                  type=int)
parser.add_option("--num-sub-banks",
                  help="No of sub-banks=No of jobs",
                  type=int)

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
####################### Split the Bank #################################
#########################################################################

################### Read in the bank file, and get the sim table ############
# Get the bank file
#{{{
if options.iid is not None:
    bank_file_name = "banks/bank_%d.xml" % np.int(options.iid)
    iid = options.iid
else:
    idx = 0
    name1 = "banks/bank_%d.xml" % idx
    idx += 1
    name2 = "banks/bank_%d.xml" % idx
    while os.path.exists(name2):
        print(name1, name2)
        idx += 1
        name1 = name2
        name2 = "banks/bank_%d.xml" % idx
    bank_file_name = name1
    iid = idx - 1

if not os.path.exists(bank_file_name):
    print("The bank file does not seem to exist or is named incorrectly")
    raise ValueError(
        "The bank file %s does not exist. The iid passed in was %d" %
        (bank_file_name, iid))

print("iid = %d" % iid)
#sys.stdout.flush()

in_bank_doc = ligolw_utils.load_filename(bank_file_name, options.verbose)
try:
    in_bank_table = table.get_table(in_bank_doc,
                                    lsctables.SimInspiralTable.tableName)
except ValueError:
    in_bank_table = table.get_table(in_bank_doc,
                                    lsctables.SnglInspiralTable.tableName)

#}}}

######### Divide the bank into sub-banks #######
#{{{
if options.bank_batch_size and options.num_sub_banks:
    raise IOError(
        "--bank-batch-size and --num-sub-banks cant BOTH be specified")

if not options.bank_batch_size and not options.num_sub_banks:
    raise IOError(
        "ONE of --bank-batch-size and --num-sub-banks MUST be specified")

if options.bank_batch_size:
    bank_batch_size = options.bank_batch_size
    print("Going to split the bank in batches of %d points each" %
          bank_batch_size)
    #sys.stdout.flush()

    bank_batches = [
        in_bank_table[i:i + bank_batch_size]
        for i in range(0, len(in_bank_table), bank_batch_size)
    ]
    ######### Write each sub-part of the bank to bank_iid_part_pid.xml #######
    #{{{
    print("Writing the sub-parts to files")
    idx = 0
    for batch in bank_batches:
        subfile_name = "sub-banks/bank_%d_part_%d.xml" % (iid, idx)
        print("Writing sub-bank file %s" % subfile_name)
        out_subbank_doc = ligolw.Document()
        out_subbank_doc.appendChild(ligolw.LIGO_LW())
        out_proc_id = ligolw_process.register_to_xmldoc(
            out_subbank_doc,
            PROGRAM_NAME,
            options.__dict__,
            comment=options.comment,
            version=git_version.id,
            cvs_repository=git_version.branch,
            cvs_entry_time=git_version.date).process_id
        out_subbank_table = lsctables.New(
            lsctables.SimInspiralTable,
            columns=[
                'mass1', 'mass2', 'mchirp', 'eta', 'spin1x', 'spin1y',
                'spin1z', 'spin2x', 'spin2y', 'spin2z', 'inclination',
                'polarization', 'latitude', 'longitude', 'bandpass', 'alpha',
                'alpha1', 'alpha2', 'process_id'
            ])
        out_subbank_doc.childNodes[0].appendChild(out_subbank_table)
        for bank_point in batch:
            out_subbank_table.append(bank_point)

        subbank_proctable = table.get_table(out_subbank_doc,
                                            lsctables.ProcessTable.tableName)
        subbank_proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(
            time.time())
        ligolw_utils.write_filename(out_subbank_doc, subfile_name)
        idx += 1

    NUM_SUB_BANKS = idx
    #}}}
else:
    NUM_SUB_BANKS = options.num_sub_banks
    # Compute the numbers of templates that will go in sub-banks. If the total
    # number is not a multiple of the number of sub-banks, then there will some
    # banks with k points, and remaining with k+1 points.
    k = np.int(len(in_bank_table) / NUM_SUB_BANKS)
    kplusOne = k + 1
    print("bank_batch_sizeS = %d,%d" % (k, kplusOne))
    # Compute the number of sub-banks which will have k, and those that will have
    # k+1 points in each.
    NBkplusOne = len(in_bank_table) % NUM_SUB_BANKS
    NBk = NUM_SUB_BANKS - NBkplusOne
    # Create 2 batches. In first batch, store groups of points that go into
    # sub-banks which have k points in each, while the second batch has those
    # groups of points which go into sub-banks that have k+1 points each.
    bank_batches_k = [in_bank_table[i:i + k] for i in range(0, NBk * k, k)]
    bank_batches_kplusOne = [
        in_bank_table[i:i + k + 1]
        for i in range(NBk * k, len(in_bank_table), k + 1)
    ]

    # Just checking
    if (len(bank_batches_k) + len(bank_batches_kplusOne)) != NUM_SUB_BANKS or (
            NBk * k + NBkplusOne * kplusOne) != len(in_bank_table):
        raise ArithmeticError(
            "Something wrong with determination of sub-bank lengths.\nNsub=%d,k=%d,kplusOne=%d,NBk=%d,NBkplusOne=%d,N=%d"
            %
            (NUM_SUB_BANKS, k, kplusOne, NBk, NBkplusOne, len(in_bank_table)))
    ######### Write each sub-part of the bank to bank_iid_part_pid.xml #######
    #{{{
    print("Writing the sub-parts to files")
    idx = 0
    for batch in bank_batches_k:
        subfile_name = "sub-banks/bank_%d_part_%d.xml" % (iid, idx)
        print("Writing sub-bank file %s" % subfile_name)
        out_subbank_doc = ligolw.Document()
        out_subbank_doc.appendChild(ligolw.LIGO_LW())
        out_proc_id = ligolw_process.register_to_xmldoc(
            out_subbank_doc,
            PROGRAM_NAME,
            options.__dict__,
            comment=options.comment,
            version=git_version.id,
            cvs_repository=git_version.branch,
            cvs_entry_time=git_version.date).process_id
        out_subbank_table = lsctables.New(
            lsctables.SimInspiralTable,
            columns=[
                'mass1', 'mass2', 'mchirp', 'eta', 'spin1x', 'spin1y',
                'spin1z', 'spin2x', 'spin2y', 'spin2z', 'inclination',
                'polarization', 'latitude', 'longitude', 'bandpass', 'alpha',
                'alpha1', 'alpha2', 'process_id'
            ])
        out_subbank_doc.childNodes[0].appendChild(out_subbank_table)
        for bank_point in batch:
            out_subbank_table.append(bank_point)

        #print "sub-bank file %s has %d points" % (subfile_name,len(out_subbank_table))
        subbank_proctable = table.get_table(out_subbank_doc,
                                            lsctables.ProcessTable.tableName)
        subbank_proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(
            time.time())
        ligolw_utils.write_filename(out_subbank_doc, subfile_name)
        idx += 1

    for batch in bank_batches_kplusOne:
        subfile_name = "sub-banks/bank_%d_part_%d.xml" % (iid, idx)
        print("Writing sub-bank file %s" % subfile_name)
        out_subbank_doc = ligolw.Document()
        out_subbank_doc.appendChild(ligolw.LIGO_LW())
        out_proc_id = ligolw_process.register_to_xmldoc(
            out_subbank_doc,
            PROGRAM_NAME,
            options.__dict__,
            comment=options.comment,
            version=git_version.id,
            cvs_repository=git_version.branch,
            cvs_entry_time=git_version.date).process_id
        out_subbank_table = lsctables.New(
            lsctables.SimInspiralTable,
            columns=[
                'mass1', 'mass2', 'mchirp', 'eta', 'spin1x', 'spin1y',
                'spin1z', 'spin2x', 'spin2y', 'spin2z', 'inclination',
                'polarization', 'latitude', 'longitude', 'bandpass', 'alpha',
                'alpha1', 'alpha2', 'process_id'
            ])
        out_subbank_doc.childNodes[0].appendChild(out_subbank_table)
        for bank_point in batch:
            out_subbank_table.append(bank_point)

        #print "sub-bank file %s has %d points" % (subfile_name,len(out_subbank_table))
        subbank_proctable = table.get_table(out_subbank_doc,
                                            lsctables.ProcessTable.tableName)
        subbank_proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(
            time.time())
        ligolw_utils.write_filename(out_subbank_doc, subfile_name)
        idx += 1

    NUM_SUB_BANKS = idx
    #}}}

#}}}

sys.stdout.flush()
