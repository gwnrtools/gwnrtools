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

### option parsing ###

parser = OptionParser(
    version=git_version.verbose_msg,
    usage="%prog [OPTIONS]",
    description="""Takes in a list of directory names as tags. Reads in the
    metadata for the simulation in it, for different Levs. Stores the tag and
    this information in an xml file.""")

parser.add_option("-i", "--input-tags", help="Names of the tags")
parser.add_option("-x",
                  "--input-catalog",
                  help="Names of the xml file to append the information to",
                  type=str,
                  default=None)
parser.add_option("-t",
                  "--output-catalog",
                  metavar='file',
                  help='output file name')
parser.add_option(
    "--nr-input-dir",
    metavar='DIR',
    help='Main dir with nr sim',
    default='/mnt/raid-project/nr/tonyc/next100Runs/confg_AlignedSpinning/')

parser.add_option('-n',
                  '--num',
                  metavar='SAMPLES',
                  help='number of templates in the output banks',
                  type=int)
parser.add_option("-V",
                  "--verbose",
                  action="store_true",
                  help="print extra debugging information",
                  default=False)

options, argv_frame_files = parser.parse_args()

if options.input_catalog is not None:
    indoc = ligolw_utils.load_filename(options.input_catalog,
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
else:
    inputtabletype = lsctables.SimInspiralTable

# Re-write the input table files
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
if options.input_catalog is not None:
    for point in input_table:
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


def get_metadatafiles(nr_tag):
    levdirs = glob.glob(options.nr_input_dir + '/' + nr_tag +
                        '/Production/BBH*/Lev?')
    levstrs, metadatafiles = [], {}
    for levdir in levdirs:
        levname = levdir.split('/')[-1]
        if os.path.exists(os.path.join(levdir, 'metadata.txt')):
            levstrs.append(levname)
            metadatafiles[levname] = open(levdir + '/metadata.txt', 'r')
    #
    print(metadatafiles)
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


# Add NEW waveforms
print("ALL tags = ", options.input_tags.split())

for tag in options.input_tags.split():
    if options.verbose:
        print("checking tag = ", tag)
    dont_add = False
    for point in out_table:
        if tag in point.waveform:
            dont_add = True
            if options.verbose:
                print("This tag already cataloged", file=sys.stderr)
            break
    if dont_add:
        continue
    mfiles = get_metadatafiles(tag)
    if options.verbose:
        print("metadatafile = ", mfiles)
    for levname in list(mfiles.keys()):
        npoint = lsctables.SimInspiral()
        # Initialize columns
        for nn in out_table.columnnames:
            if 'process_id' in nn:
                npoint.process_id = proc_id
            elif 'waveform' in nn:
                npoint.waveform = tag + '-' + levname
            elif 'numrel_data' in nn:
                npoint.numrel_data = 'FILL'  # <- CCE hdata
            elif 'numrel_mode_min' in nn:
                npoint.numrel_mode_min = 2
            elif 'numrel_mode_max' in nn:
                npoint.numrel_mode_max = 8
            else:
                npoint.__setattr__(nn, 0)
        #
        print("Reading for %s %s" % (tag, levname))
        npoint.mass1, npoint.mass2, \
            npoint.spin1x, npoint.spin1y, npoint.spin1z, \
            npoint.spin2x, npoint.spin2y, npoint.spin2z, \
            npoint.f_lower, npoint.t_end_time = \
            get_data_from_metadatafile(mfiles[levname])
        npoint.eta = npoint.mass1 * npoint.mass2 / \
            (npoint.mass1 + npoint.mass2)**2
        npoint.mchirp = (npoint.mass1 + npoint.mass2) * npoint.eta**0.6
        out_table.append(npoint)

# write the xml doc to disk
proctable = lsctables.ProcessTable.get_table(outdoc)
proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(time.time())

outname = options.output_catalog + '.xml'
ligolw_utils.write_filename(outdoc, outname)

print(len(out_table))
