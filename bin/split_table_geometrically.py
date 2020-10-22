#!/usr/bin/env python
import time
import sys
import os, logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',\
                     level=logging.INFO, stream=sys.stdout)

import numpy as np
import argparse

from glue import gpstime
from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import process as ligolw_process

PROGRAM_NAME = os.path.abspath(sys.argv[0])
__author__ = "Prayush Kumar <prayush@astro.cornell.edu>"

#########################################################################
#################### Input parsing #####################
#########################################################################
#{{{
parser = argparse.ArgumentParser(
    usage="%%prog [OPTIONS]",
    description="""
Takes in a template bank. Sorts all templates according to chirp mass. Then it
splits those templates into bins of width m_min_i*(1 + mchirp_window), where
m_min_i is the minimum chirp mass of the i'th bin.

For i=0, i.e. the global mchirp minimum is provided by user. Binning stops with
m_min_i * (1 + mchirp_window) >= m_max, where m_max is the global maximum
allowed chirp mass that user provides.

There should be N = log(m_max / m_min_0) / log(1 + mchirp_window) bins, and as
many sub-banks created.
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#~ parser.add_argument("--iteration-id", dest="iid",
#~ help="The index of the iteration",
#~ type=int)
parser.add_argument("-t",
                    "--tmplt-bank",
                    metavar='file',
                    help='template bank to split')

parser.add_argument('-n',
                    '--num',
                    metavar='SAMPLES',
                    help='number of output banks',
                    type=int)
parser.add_argument(
    '--mchirp-min',
    metavar='MC_MIN',
    help="""Global minimum for mchirp. This should be consistent
with the bank provided.""",
    type=float)
parser.add_argument(
    '--mchirp-max',
    metavar='MC_MAX',
    help="""Global maximum for mchirp. This should be consistent
with the bank provided.""",
    type=float)
parser.add_argument('-w',
                    '--mchirp-window',
                    metavar='MC_WIN',
                    help="""Fractional window on mchirp parameter. If waveform
parameters differ by more than this window, the overlap is set to 0.""",
                    default=0.01,
                    type=float)

parser.add_argument("-V",
                    "--verbose",
                    action="store_true",
                    help="print extra debugging information",
                    default=False)
parser.add_argument("--comment", help="Comment string to be stored in proc_id")
parser.add_argument("-e",
                    "--named",
                    help="Starting string in the names of final XMLs")

options = parser.parse_args()
#}}}
#########################################################################
#################### Initialize #####################
#########################################################################
logging.info('{}'.format(options.named))
indoc = ligolw_utils.load_filename(options.tmplt_bank,\
            contenthandler=table.use_in(ligolw.LIGOLWContentHandler),
            verbose=options.verbose)
try:
    template_bank_table = lsctables.SnglInspiralTable.get_table(indoc)
    tabletype = lsctables.SnglInspiralTable
except:
    template_bank_table = lsctables.SimInspiralTable.get_table(indoc)
    tabletype = lsctables.SimInspiralTable

#########################################################################
#################### Split & Write  #####################
#########################################################################
#{{{
mc_min = options.mchirp_min
mc_max = options.mchirp_max
mc_win = options.mchirp_window

mchirps_in_bank = np.array([p.mchirp for p in template_bank_table])
if mc_min > np.min(mchirps_in_bank) or mc_max < np.max(mchirps_in_bank):
    raise IOError("Provided bank has mchirps outside the provided range!")

n_should_be = int(np.ceil(np.log(mc_max / mc_min) / np.log(1. +\
                                              options.mchirp_window)))
if options.num != n_should_be:
    raise IOError(\
      """num_in={} and n_should_be={}, therefore, Inconsistent choice of
    (mc_min={}, mc_max={}, mc_window={}) passed""".format(\
        options.num, n_should_be, mc_min, mc_max, mc_win))

mc_bins_edges = np.array([mc_min * (1. +\
                    options.mchirp_window)**i for i in range(options.num + 1)])
if mc_bins_edges[-1] >= mc_max and mc_bins_edges[-2] < mc_max:
    mc_bins_edges[-1] = mc_max
else:
    raise IOError("Could not split mchirp into bins suitably")

if options.verbose:
    logging.info("Mchirp bin edges chosen are: {}".format(mc_bins_edges))
    sys.stdout.flush()
for i in range(len(mc_bins_edges) - 1):
    # create a blank xml document and add points that fall within the i'th bin
    outdoc = ligolw.Document()
    outdoc.appendChild(ligolw.LIGO_LW())
    new_inspiral_table = lsctables.New(tabletype,
                                       columns=template_bank_table.columnnames)
    outdoc.childNodes[0].appendChild(new_inspiral_table)

    out_proc_id = ligolw_process.register_to_xmldoc(
        outdoc, PROGRAM_NAME, options.__dict__,
        comment=options.comment).process_id

    for p in template_bank_table:
        if p.mchirp >= mc_bins_edges[i] and p.mchirp < mc_bins_edges[i + 1]:
            p.process_id = out_proc_id
            new_inspiral_table.append(p)

    if options.verbose:
        logging.info("\t {} templates in sub-bank {}.".format(\
          len(new_inspiral_table), i))
    # write the xml doc to disk
    proctable = table.get_table(outdoc, lsctables.ProcessTable.tableName)
    proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(time.time())

    outname = options.named + '%06d.xml' % i
    ligolw_utils.write_filename(outdoc, outname)

logging.info("{}".format(i))
