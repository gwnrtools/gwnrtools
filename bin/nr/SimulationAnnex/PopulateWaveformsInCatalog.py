#!/usr/bin/env python

import time
import os
import sys
import string
from optparse import OptionParser

from glue import gpstime, git_version
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import process as ligolw_process

import glob
import numpy as np


@lsctables.use_in
class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
    pass


__author__ = "Prayush Kumar <prayush@astro.cornell.edu>"

PROGRAM_NAME = os.path.abspath(sys.argv[0])

### Function Definitions ###


def add_strings(strlist, fill=None):
    s = ''
    for st in strlist:
        s = s + st
        if fill is not None:
            s = s + fill
    return s[:-1]


def get_metadatafiles(nr_tag):
    levtag = options.lev_tag
    levdirs = glob.glob(options.nr_input_dir + '/' + nr_tag + '/' + levtag)
    if len(levdirs) == 0:
        raise IOError("Lev directories not found. Please check the --lev-tag")
    levstrs, metadatafiles = [], {}
    for levdir in levdirs:
        levname = levdir.split('/')[-1]
        try:
            metadatafiles[levname] = open(levdir + '/metadata.txt', 'r')
        except BaseException:
            continue
        levstrs.append(levname)
    return metadatafiles


def get_data_from_metadatafile(fin, old_format_for_spins=False):
    #fin = open(metadatafile,'r')
    lines = fin.readlines()
    for i in range(len(lines)):
        if 'relaxed-mass1' in lines[i]:
            m1line = lines[i]
            if options.verbose:
                print(m1line)
            m1 = np.float64(m1line.split()[-1])
        if 'relaxed-mass2' in lines[i]:
            m2line = lines[i]
            if options.verbose:
                print(m2line)
            m2 = np.float64(m2line.split()[-1])
        if 'relaxed-orbital-frequency =' in lines[i]:
            omegaline = lines[i]
            if options.verbose:
                print(omegaline)
            omega = np.float64(omegaline.split()[-1])
        if 'relaxed-dimensionless-spin1' in lines[i]:
            chi1line = lines[i]
            if options.verbose:
                print(chi1line)
        if 'relaxed-dimensionless-spin2' in lines[i]:
            chi2line = lines[i]
            if options.verbose:
                print(chi2line)
        if 'relaxed-measurement-time =' in lines[i]:
            trelaxline = lines[i]
            if options.verbose:
                print(trelaxline)
            trelax = np.float64(trelaxline.split()[-1])
    sys.stdout.flush()
    sys.stderr.flush()
    chi1x = np.float64(chi1line.split()[-3][:-1])
    chi1y = np.float64(chi1line.split()[-2][:-1])
    chi1z = np.float64(chi1line.split()[-1])
    chi2x = np.float64(chi2line.split()[-3][:-1])
    chi2y = np.float64(chi2line.split()[-2][:-1])
    chi2z = np.float64(chi2line.split()[-1])
    if old_format_for_spins:
        chi1x /= (m1**2)
        chi1y /= (m1**2)
        chi1z /= (m1**2)
        chi2x /= (m2**2)
        chi2y /= (m2**2)
        chi2z /= (m2**2)
    return [m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, omega, trelax]


def get_waveform_location(
        p,
        cce_filename_ascii='h_from_Psi4_scri.L02Mp02.dat',
        cce_filename='rhOverM_CcePITT_Asymptotic_GeometricUnits.h5',
        extrapolated_filename='rhOverM_Asymptotic_GeometricUnits.h5',
        finite_radii_filename='rh_FiniteRadii_CodeUnits.h5',
        wavetypes='cce',
        wavename='',
        allow_symlinks=True):
    """
  Goes through all wavetypes IN ORDER GIVEN, and returns the location of
  first waveform found on disk
    """
    if len(wavetypes) == 0:
        wavetypes = ['dummy']
    for idx, wavetype in enumerate(wavetypes):
        if wavename != '':
            filename = wavename
        elif wavetype == 'cce':
            filename = cce_filename
        elif wavetype == 'extrapolated':
            filename = extrapolated_filename
        elif wavetype == 'finite-radius':
            filename = finite_radii_filename
        else:
            raise RuntimeError("Couldn't find which NR wave file to read.")
        #
        tag = p.waveform
        if options.verbose:
            print("Trying wavetype %s for %s " % (wavetype, tag),
                  file=sys.stdout)
        #
        subdir = tag.split('-')[-1]
        dirname = add_strings(tag.split('-')[:-1], '-')
        workdir = options.nr_input_dir
        # FIXME
        if options.verbose:
            print(tag, subdir, dirname, workdir)
        #
        if options.use_hdf:
            h22file = workdir + '/' + dirname + '/' + subdir + '/' + filename
        else:
            h22file = workdir + '/' + dirname + '/' + subdir + \
                '/highResCce.joined/' + filename
        if os.path.exists(h22file) and os.path.getsize(h22file) > 0:
            if options.verbose:
                print("Waveform found for %s at %s" % (tag, h22file),
                      file=sys.stdout)
            return h22file
        elif allow_symlinks and os.path.islink(h22file):
            if options.verbose:
                print("Waveform SYMLINK found for %s at %s" % (tag, h22file),
                      file=sys.stdout)
            return h22file
        else:
            if options.verbose:
                print("Waveform %s NOT found for %s" % (h22file, tag),
                      file=sys.stdout)
    return None


### option parsing ###

parser = OptionParser(
    version=git_version.verbose_msg,
    usage="%prog [OPTIONS]",
    description="""Takes in a list of directory names as tags. Reads in the
    metadata for the simulation in it, for different Levs. Stores the tag and
    this information in an xml file.""")

parser.add_option(
    "--nr-input-dir",
    metavar='DIR',
    help='Main dir with nr sim',
    default='/home/p/pfeiffer/prayush/scratch/projects/CCE/ChuAlignedSpinning/'
)
parser.add_option("-i", "--input-tags", help="Names of the tags")
parser.add_option(
    "-l",
    "--lev-tag",
    help=
    "RegEx for the Subdirectory of the simulation directory containing the waveform files",
    default='Production/BBH*/Lev?')
parser.add_option(
    "-w",
    "--wavetype",
    help="Please provide one of cce, extrapolated, finite-radius",
    default='')
parser.add_option(
    "-n",
    "--wave-name",
    help="Please provide one of cce, extrapolated, finite-radius",
    default='')

parser.add_option("-x",
                  "--input-catalog",
                  help="Names of the xml file to append the information to",
                  type=str,
                  default=None)
parser.add_option("-t", "--output-catalog", help='output file name')

parser.add_option("--use-hdf",
                  action="store_true",
                  help="Store ascii file or HDF5 file location?",
                  default=False)
parser.add_option(
    "--use-symlinks",
    action="store_true",
    help="Catalog symlinks even if the linked file does not exist",
    default=False)
parser.add_option("--use-highest-lev",
                  action="store_true",
                  help="Use only the highest lev for each simulation",
                  default=False)

parser.add_option(
    "--restrict-zero-spins",
    action="store_true",
    help=
    "Use only the non-spinning waveforms. transverse-spin-threshold is used as the spin-threshold",
    default=False)
parser.add_option(
    "--restrict-aligned-spins",
    action="store_true",
    help=
    "Use only the aligned-spin simulation. transverse spin threshold is reqd",
    default=False)
parser.add_option("--transverse-spin-threshold",
                  type=float,
                  help="Magnitude of x,y spins below which they are set to 0",
                  default=1.e-4)
parser.add_option("--store-path-relative-to", type=str, default='')

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
    try:
        input_table = lsctables.SnglInspiralTable.get_table(indoc)
        inputtabletype = lsctables.SnglInspiralTable
    except BaseException:
        input_table = lsctables.SimInspiralTable.get_table(indoc)
        inputtabletype = lsctables.SimInspiralTable
    # print tabletype
    length = len(input_table)
else:
    print(
        "Waning: No input table given to append to, will construct one from scratch"
    )
    inputtabletype = lsctables.SimInspiralTable
    #raise IOError("Please give a table to add the information about NR waveforms to.")

# Re-write the input table files.
# Create a blank xml document and add the process id
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

wavetypes = options.wavetype.split()

FAILED_METADATA = []
FAILED_DATA_LOCATION = []

if options.input_catalog is not None:
    # Fill in the INPUT table
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
        npoint.numrel_data = get_waveform_location(
            npoint,
            wavetypes=wavetypes,
            wavename=options.wave_name,
            allow_symlinks=options.use_symlinks)

        if npoint.numrel_data is None:
            if options.verbose:
                print("NO WAVE FOUND for %s. SKIPPING.." % npoint.waveform,
                      file=sys.stderr)
                sys.stderr.flush()
            del npoint
            continue
        out_table.append(npoint)
    #
    if options.verbose:
        print(len(out_table), " points copied", file=sys.stderr)
else:
    # Construct the table from scratch
    for tag in options.input_tags.split():
        if options.verbose:
            print("\n\n ##################################################")
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
        mkeys = list(mfiles.keys())
        mkeys.sort(reverse=True)
        for levname in mkeys:
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
            if options.verbose:
                print("Reading for %s %s" % (tag, levname))
            try:
                npoint.mass1, npoint.mass2, \
                    npoint.spin1x, npoint.spin1y, npoint.spin1z, \
                    npoint.spin2x, npoint.spin2y, npoint.spin2z, \
                    npoint.f_lower, npoint.t_end_time = \
                    get_data_from_metadatafile(mfiles[levname])
            except BaseException:
                FAILED_METADATA.append(tag)
                del npoint
                continue
            npoint.eta = npoint.mass1 * npoint.mass2 / \
                (npoint.mass1 + npoint.mass2)**2
            npoint.mchirp = (npoint.mass1 + npoint.mass2) * npoint.eta**0.6
            ##
            try:
                npoint.numrel_data = get_waveform_location(
                    npoint,
                    wavetypes=wavetypes,
                    wavename=options.wave_name,
                    allow_symlinks=options.use_symlinks)
            except BaseException:
                FAILED_DATA_LOCATION.append(tag)
                del npoint
                continue
            if npoint.numrel_data is None:
                FAILED_DATA_LOCATION.append(tag)
                if options.verbose:
                    print("NO WAVE FOUND for %s. SKIPPING.." % npoint.waveform,
                          npoint.numrel_data,
                          file=sys.stderr)
                    sys.stderr.flush()
                del npoint
                continue
            #
            if options.restrict_zero_spins:
                sthreshold = options.transverse_spin_threshold
                if abs(npoint.spin1x) > sthreshold or abs(
                        npoint.spin1y) > sthreshold or abs(
                            npoint.spin1z) > sthreshold or abs(
                                npoint.spin2x) > sthreshold or abs(
                                    npoint.spin2y) > sthreshold or abs(
                                        npoint.spin2z) > sthreshold:
                    continue

            if options.restrict_aligned_spins:
                tsthreshold = options.transverse_spin_threshold
                if abs(npoint.spin1x) > tsthreshold or abs(
                        npoint.spin1y) > tsthreshold or abs(
                            npoint.spin2x) > tsthreshold or abs(
                                npoint.spin2y) > tsthreshold:
                    continue
                if abs(npoint.spin1z) < tsthreshold and abs(
                        npoint.spin2z) < tsthreshold:
                    continue

            if len(options.store_path_relative_to) != 0:
                npoint.numrel_data = string.split(
                    npoint.numrel_data, options.store_path_relative_to)[-1]

            out_table.append(npoint)
            if options.use_highest_lev:
                break

# write the xml doc to disk
proctable = lsctables.ProcessTable.get_table(outdoc)
proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(time.time())

outname = options.output_catalog + '.xml'
ligolw_utils.write_filename(outdoc, outname)

print("\n\n Total %d simulations cataloged" % len(out_table))
print("\n\n Simulations for which metadata reading failed:\n", FAILED_METADATA)
print("\n\n Simulations for which data location was not found:\n",
      FAILED_DATA_LOCATION)
