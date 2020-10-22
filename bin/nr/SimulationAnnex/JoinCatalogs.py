#!/usr/bin/env python

import time
import os
import sys

from optparse import OptionParser

from glue import gpstime, git_version
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import process as ligolw_process
#from pylal import series as lalseries

import glob
import numpy as np


@lsctables.use_in
class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
    pass


__author__ = "Prayush Kumar <prkumar@cita.utoronto.ca>"

PROGRAM_NAME = os.path.abspath(sys.argv[0])

# ##################################################################
### Function Definitions ###
# ##################################################################


def add_strings(strlist, fill=None):
    s = ''
    for st in strlist:
        s = s + st
        if fill is not None:
            s = s + fill
    return s[:-1]


def get_metadatafiles(nr_tag):
    levdirs = glob.glob(options.nr_input_dir + '/' + nr_tag +
                        '/Production/BBH*/Lev?')
    levstrs, metadatafiles = [], {}
    for levdir in levdirs:
        levname = levdir.split('/')[-1]
        levstrs.append(levname)
        metadatafiles[levname] = open(levdir + '/metadata.txt', 'r')
    #
    return metadatafiles


def get_data_from_metadatafile(fin):
    #fin = open(metadatafile,'r')
    lines = fin.readlines()
    for i in range(len(lines)):
        if 'relaxed-mass1' in lines[i]:
            m1line = lines[i]
            print(m1line)
            m1 = np.float64(m1line.split()[-1])
        if 'relaxed-mass2' in lines[i]:
            m2line = lines[i]
            print(m2line)
            m2 = np.float64(m2line.split()[-1])
        if 'relaxed-orbital-frequency =' in lines[i]:
            omegaline = lines[i]
            print(omegaline)
            omega = np.float64(omegaline.split()[-1])
        if 'relaxed-spin1' in lines[i]:
            chi1line = lines[i]
            print(chi1line)
        if 'relaxed-spin2' in lines[i]:
            chi2line = lines[i]
            print(chi2line)
        if 'relaxed-measurement-time =' in lines[i]:
            trelaxline = lines[i]
            print(trelaxline)
            trelax = np.float64(trelaxline.split()[-1])
    #
    sys.stdout.flush()
    sys.stderr.flush()
    chi1x = np.float64(chi1line.split()[-3][:-1]) / m1**2
    chi1y = np.float64(chi1line.split()[-2][:-1]) / m1**2
    chi1z = np.float64(chi1line.split()[-1]) / m1**2
    chi2x = np.float64(chi2line.split()[-3][:-1]) / m2**2
    chi2y = np.float64(chi2line.split()[-2][:-1]) / m2**2
    chi2z = np.float64(chi2line.split()[-1]) / m2**2
    return [m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, omega, trelax]


def get_waveform_location(
        p,
        extrap_filename='h_from_Psi4_scri.L02Mp02.dat',
        cce_filename='rhOverM_CcePITT_Asymptotic_GeometricUnits.h5'):
    tag = p.waveform
    subdir = tag.split('-')[-1]
    dirname = add_strings(tag.split('-')[:-1], '-')
    workdir = options.nr_input_dir
    #
    if options.use_hdf:
        h22file = workdir + '/' + dirname + '/' + subdir + '/' + cce_filename
    else:
        h22file = workdir + '/' + dirname + '/' + subdir + \
            '/highResCce.joined/' + extrap_filename
    if os.path.exists(h22file) and os.path.getsize(h22file) > 0:
        print("Waveform found for %s at %s" % (tag, h22file), file=sys.stdout)
        return h22file
    else:
        print("Waveform %s NOT found for %s" % (h22file, tag), file=sys.stdout)
        return p.numrel_data


# ##################################################################
### option parsing ###
# ##################################################################
parser = OptionParser(
    version=git_version.verbose_msg,
    usage="%prog [OPTIONS]",
    description="""Takes in a list of directory names as tags. Reads in the
    metadata for the simulation in it, for different Levs. Stores the tag and
    this information in an xml file.""")

parser.add_option("-x",
                  "--input-catalogs",
                  help="Names of the xml file to append the information to",
                  default=None)
parser.add_option("-t", "--output-catalog", help='output file name')
parser.add_option(
    "--nr-input-dir",
    metavar='DIR',
    help='Main dir with nr sim',
    default='/home/p/pfeiffer/prayush/scratch/projects/CCE/ChuAlignedSpinning/'
)
parser.add_option("--use-hdf",
                  action="store_true",
                  help="Store ascii file or HDF5 file location?",
                  default=False)

parser.add_option(
    "-Q",
    "--upper-q-threshold",
    help='Dont include points with mass-ratio above this threshold',
    type=float,
    default=-2.)

parser.add_option("-V",
                  "--verbose",
                  action="store_true",
                  help="print extra debugging information",
                  default=False)

options, argv_frame_files = parser.parse_args()

print(options.input_catalogs)
print("testing")

if options.input_catalogs is not None:
    input_tables = []
    input_catalogs = options.input_catalogs.split()
    if len(input_catalogs) == 0:
        raise IOError("Please provide catalogs to join")
    for input_catalog in input_catalogs:
        if options.verbose:
            print("using >> %s" % input_catalog)
        indoc = ligolw_utils.load_filename(input_catalog,
                                           contenthandler=LIGOLWContentHandler,
                                           verbose=options.verbose)
        #
        try:
            input_table = lsctables.SnglInspiralTable.get_table(indoc)
            inputtabletype = lsctables.SnglInspiralTable
        except BaseException:
            input_table = lsctables.SimInspiralTable.get_table(indoc)
            inputtabletype = lsctables.SimInspiralTable
        #
        # print tabletype
        length = len(input_table)
        input_tables.append([input_catalog, input_table])
else:
    raise IOError(
        "Please give a table to add the information about NR waveforms to.")

# ##################################################################

# ##################################################################
# From each input table, write over all rows with valid waveform locations.
# create a blank xml document and add the process id
outdoc = ligolw.Document()
outdoc.appendChild(ligolw.LIGO_LW())
proc_id = ligolw_process.register_to_xmldoc(
    outdoc,
    PROGRAM_NAME,
    options.__dict__,
    ifos=["G1"],
    version=git_version.id,
    cvs_repository=git_version.branch,
    cvs_entry_time=git_version.date).process_id

out_table = lsctables.New(
    inputtabletype,
    columns=[
        'mass1', 'mass2', 'mchirp', 'eta', 'spin1x', 'spin1y', 'spin1z',
        'spin2x', 'spin2y', 'spin2z', 'inclination', 'polarization',
        'latitude', 'longitude', 'bandpass', 'alpha', 'alpha1', 'alpha2',
        'process_id', 'waveform', 'numrel_data', 'numrel_mode_min',
        'numrel_mode_max', 't_end_time', 'f_lower'
    ])
outdoc.childNodes[0].appendChild(out_table)

# Copy the INPUT table
for input_catalog, input_table in input_tables:
    if options.verbose:
        print("Reading from %s" % input_catalog, file=sys.stderr)
    #
    for point in input_table:
        # Apply the mass-ratio threshold
        qth = options.upper_q_threshold
        if point.eta < (qth / (1. + qth)**2):
            if options.verbose:
                print("  -- Not including %s" % point.waveform,
                      file=sys.stdout)
                sys.stdout.flush()
            continue
        # Check if the waveform location is specified.
        # if not os.path.exists(point.numrel_data): continue
        # if not (os.path.getsize(point.numrel_data) > 0): continue
        # If cehck passed, proceed to appending it to the final table
        npoint = lsctables.SimInspiral()
        # Initialize columns
        for nn in out_table.columnnames:
            if 'process_id' in nn:
                npoint.process_id = proc_id
            elif 'waveform' in nn:
                npoint.waveform = 'NR'
            else:
                npoint.__setattr__(nn, 0)
        # Copy over columns
        for nn in point.__slots__:
            if hasattr(point, nn):
                npoint.__setattr__(nn, point.__getattribute__(nn))
        #
        out_table.append(npoint)
    #
    if options.verbose:
        print(len(out_table), " points copied", file=sys.stderr)

if options.verbose:
    print("Total %d points in final table" % len(out_table), file=sys.stderr)
# write the xml doc to disk
proctable = lsctables.ProcessTable.get_table(outdoc)
proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(time.time())

outname = options.output_catalog + '.xml'
ligolw_utils.write_filename(outdoc, outname)

print(len(out_table))
