#! /usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from numpy import loadtxt

import sys
import os
import time

from optparse import OptionParser

import qm

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
parser.add_option(
    "--unc-points-file",
    help=
    "File containing mc-etai-chi values of uncoverde points, in THAT order in each column",
    type=str)
parser.add_option("--etas-file",
                  help="File containing allowed Eta values",
                  type=str)
parser.add_option("--input-bank-file", help="Input bank file", type=str)
parser.add_option("--output-bank-file", help="Input bank file", type=str)
parser.add_option("--mchirp-cluster-window",
                  help="Clustering Window on mchirp",
                  type=float,
                  default=0.)
parser.add_option("--first-chi",
                  action="store_true",
                  help="used if the uncovered point is placed based on chi",
                  default=False)
parser.add_option("--first-eta",
                  action="store_true",
                  help="used if the uncovered point is placed based on eta",
                  default=False)

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
##### Get the Input Bank, uncovered points, and eta vales ###############
#########################################################################

################### Read in the bank file, and get the sim table ############
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

################### Read in the uncovered points list ############
if options.unc_points_file is None or not os.path.exists(
        options.unc_points_file):
    raise ValueError("The uncovered points file %s does not exist" %
                     options.unc_points_file)
else:
    unc_file = options.unc_points_file

unc_fil = open(unc_file, 'r')
unc_dat = loadtxt(unc_fil)

################### Read in the eta-chi list ############
if options.etas_file is None or not os.path.exists(options.etas_file):
    raise ValueError("The etas file %s does not exist" % options.etas_file)
else:
    etas_file = options.etas_file

etas_fil = open(etas_file, 'r')
etas_dat = loadtxt(etas_fil)

#########################################################################
############################ Process the uncovered points ###############
#########################################################################

############### Organize the uncovered points to nearest-eta bins #######
## Sort the etas list
# Assume its already sorted in eta, from lower to higher
#etas_dat.sort()

## Remove duplicates from etas list
alletas = []
allchis = []
for eta, chi in etas_dat:
    if chi not in allchis:
        alletas.append(eta)
        allchis.append(chi)

if options.first_chi:
    allrest = list(zip(allchis, alletas))
    allsorted = sorted(allrest)
    allchis = [point[0] for point in allsorted]
    alletas = [point[1] for point in allsorted]

    ## Create a blank list for chirp mass storage
    allmts = []
    for eta in alletas:
        allmts.append([])

    ## Loop over all the uncovered points
    for i in range(len(unc_dat)):
        new_chi = unc_dat[i][2]
        ## For each uncovered point, add it to its BOTH neighboring chi values
        # Find the two neighboring chi values
        for j in range(len(allchis)):
            if j == 0 and new_chi <= allchis[0]:
                allmts[j].append(unc_dat[i][0])
                break
            elif j == (len(allchis) - 1) and new_chi >= allchis[j]:
                allmts[j].append(unc_dat[i][0])
                break
            elif new_chi >= allchis[j] and new_chi <= allchis[j + 1]:
                allmts[j].append(unc_dat[i][0])
                allmts[j + 1].append(unc_dat[i][0])
                break

    mchirp_window = options.mchirp_cluster_window
    ## Loop over alletas
    for i in range(len(alletas)):
        ## Loop over all mts for each eta
        allmts[i].sort()
        for j in range(len(allmts[i])):
            if j >= len(allmts[i]):
                break
            ## For all mts that are within the mchirp window, include only the average
            thismt = allmts[i][j]
            print("thismt = ", thismt)
            thesemts = [thismt]
            for k in range(j + 1, len(allmts[i])):
                if k >= len(allmts[i]):
                    break
                if abs(thismt - allmts[i][k]) < (mchirp_window * thismt):
                    thesemts.append(allmts[i][k])
                    allmts[i].pop(k)
            allmts[i][j] = sum(thesemts) / len(thesemts)
            print("avg mt = ", allmts[i][j])
elif options.first_eta:
    allrest = list(zip(alletas, allchis))
    allsorted = sorted(allrest)
    allchis = [point[1] for point in allsorted]
    alletas = [point[0] for point in allsorted]

    ## Create a blank list for chirp mass storage
    allmts = []
    for eta in alletas:
        allmts.append([])

    ## Loop over all the uncovered points
    for i in range(len(unc_dat)):
        new_eta = unc_dat[i][1]
        ## For each uncovered point, add it to its BOTH neighboring chi values
        # Find the two neighboring eta values
        for j in range(len(alletas)):
            if j == 0 and new_eta <= alletas[0]:
                allmts[j].append(unc_dat[i][0])
                break
            elif j == (len(alletas) - 1) and new_eta >= alletas[j]:
                allmts[j].append(unc_dat[i][0])
                break
            elif new_eta >= alletas[j] and new_eta <= alletas[j + 1]:
                allmts[j].append(unc_dat[i][0])
                allmts[j + 1].append(unc_dat[i][0])
                break

    mchirp_window = options.mchirp_cluster_window
    ## Loop over alletas
    for i in range(len(alletas)):
        ## Loop over all mts for each eta
        allmts[i].sort()
        for j in range(len(allmts[i])):
            if j >= len(allmts[i]):
                break
            ## For all mts that are within the mchirp window, include only the average
            thismt = allmts[i][j]
            print("thismt = ", thismt)
            thesemts = [thismt]
            for k in range(j + 1, len(allmts[i])):
                if k >= len(allmts[i]):
                    break
                if abs(thismt - allmts[i][k]) < (mchirp_window * thismt):
                    thesemts.append(allmts[i][k])
                    allmts[i].pop(k)
            allmts[i][j] = sum(thesemts) / len(thesemts)
            print("avg mt = ", allmts[i][j])
else:
    raise IOError(
        "must supply which of eta or chi is the one to be used to choose the new point to be added"
    )

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
    out_bank_table.append(bank_point)

## Write the reduced list of uncovered points
for i in range(len(alletas)):
    chi = allchis[i]
    eta = alletas[i]
    print("Adding %d points for chi = %f" % (len(allmts[i]), chi))
    for j in range(len(allmts[i])):
        mt = allmts[i][j]
        mc = mt * eta**0.6
        m1, m2 = qm.mchirp_eta_to_m1_m2(mc, eta)
        point = lsctables.SimInspiral()
        point.mass1 = m1
        point.mass2 = m2
        point.mchirp = mc
        point.eta = eta
        point.spin1x = 0
        point.spin1y = 0
        point.spin1z = chi
        point.spin2x = 0
        point.spin2y = 0
        point.spin2z = chi
        point.inclination = 0
        point.polarization = 0
        point.latitude = 0
        point.longitude = 0
        point.bandpass = j
        point.alpha = 0
        point.alpha1 = 0
        point.alpha2 = 0
        point.process_id = out_proc_id
        out_bank_table.append(point)

#print "sub-bank file %s has %d points" % (subfile_name,len(out_subbank_table))
bank_proctable = table.get_table(out_bank_doc,
                                 lsctables.ProcessTable.tableName)
bank_proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(time.time())
ligolw_utils.write_filename(out_bank_doc, out_bank_file)
