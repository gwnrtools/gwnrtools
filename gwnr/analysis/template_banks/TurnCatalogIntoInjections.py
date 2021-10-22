#!/usr/bin/env python

# Copyright (c) 2018, Prayush Kumar
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#

from gwnr.nr import nr_strain as nr_wave

import numpy as np
import glob
import time
import os
import sys

from optparse import OptionParser

import lal
from glue import gpstime, git_version
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import process as ligolw_process


@lsctables.use_in
class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
    pass


MAX_NR_LENGTH = 100000


def nextpow2(x):
    return int(2**np.ceil(np.log2(x)))


def add_strings(strlist, fill=None):
    s = ''
    for st in strlist:
        s = s + st
        if fill is not None:
            s = s + fill
    return s[:-1]


def get_metadatafiles(nr_tag):
    levtag = options.nr_input_dir + '/' + nr_tag + '/' + options.lev_tag
    levdirs = glob.glob(levtag)
    # FIXME
    print("In ", levtag, type(levtag))
    print("Found ", levdirs)
    if len(levdirs) == 0:
        raise IOError("Lev directories not found. Please check the --lev-tag")
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
        cce_filename_ascii='h_from_Psi4_scri.L02Mp02.dat',
        cce_filename='rhOverM_CcePITT_Asymptotic_GeometricUnits.h5',
        extrapolated_filename='rhOverM_Asymptotic_GeometricUnits.h5',
        finite_radii_filename='rh_FiniteRadii_CodeUnits.h5',
        wavetype='cce',
        allow_symlinks=True):
    if wavetype == 'cce':
        filename = cce_filename
    elif wavetype == 'extrapolated':
        filename = extrapolated_filename
    elif wavetype == 'finite-radius':
        filename = finite_radii_filename

    tag = p.waveform
    subdir = tag.split('-')[-1]
    dirname = add_strings(tag.split('-')[:-1], '-')
    workdir = options.nr_input_dir
    # FIXME
    print(tag, subdir, dirname, workdir)
    #
    if options.use_hdf:
        h22file = workdir + '/' + dirname + '/' + subdir + '/' + filename
    else:
        h22file = workdir + '/' + dirname + '/' + subdir + \
            '/highResCce.joined/' + filename
    if os.path.exists(h22file) and os.path.getsize(h22file) > 0:
        print("Waveform found for %s at %s" % (tag, h22file), file=sys.stdout)
        return h22file
    elif allow_symlinks and os.path.islink(h22file):
        print("Waveform SYMLINK found for %s at %s" % (tag, h22file),
              file=sys.stdout)
        return h22file
    else:
        print("Waveform %s NOT found for %s" % (h22file, tag), file=sys.stdout)
        return p.numrel_data


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
    help='DEFUNCT: Main dir with nr sim',
    default='/home/p/pfeiffer/prayush/scratch/projects/CCE/ChuAlignedSpinning/'
)

parser.add_option("--f-lower", type=float, help="Low f cutoff", default=15.)
parser.add_option("--sample-rate",
                  type=float,
                  help="WF sample rate",
                  default=4096)
parser.add_option("--upper-mass-threshold",
                  type=float,
                  help="Upper limit for total mass sampling",
                  default=150.)
parser.add_option("--mass-sampling-step", type=float, help="dM", default=1.)

parser.add_option("-x",
                  "--input-catalog",
                  help="Names of the xml file to append the information to",
                  type=str,
                  default=None)
parser.add_option("-t", "--output-catalog", help='output file name')

parser.add_option("-F",
                  "--force-file-exists",
                  action="store_true",
                  help="Only add injections if the NR data file exists",
                  default=False)

parser.add_option("--transverse-spin-threshold",
                  type=float,
                  help="Magnitude of x,y spins below which they are set to 0",
                  default=1.e-4)

parser.add_option(
    "--zero-transverse-spins",
    action="store_true",
    help="if transverse spins are smaller than spin-threshold, set them to 0",
    default=True)

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
    except:
        input_table = lsctables.SimInspiralTable.get_table(indoc)
        inputtabletype = lsctables.SimInspiralTable
    #
    # print tabletype
    length = len(input_table)
else:
    print(
        "Waning: No input table given to append to, will construct one from scratch"
    )
    inputtabletype = lsctables.SimInspiralTable
    #raise IOError("Please give a table to add the information about NR waveforms to.")

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

if options.input_catalog is not None:
    # Fill in the INPUT table
    for point in input_table:
        if not os.path.exists(point.numrel_data):
            print("waveform file %s for %s does NOT exist! " %
                  (point.numrel_data, point.waveform))
            if options.force_file_exists:
                print("(SKIPPING)\n")
                continue
        mw = point.f_lower
        mLowMassCutoff = mw / np.pi / lal.MTSUN_SI / options.f_lower
        while True:
            total_mass = mLowMassCutoff
            estimated_length_pow2 = nextpow2(MAX_NR_LENGTH * total_mass *
                                             lal.MTSUN_SI)
            nrwav = nr_wave(filename=point.numrel_data,
                            modeLmax=2,
                            sample_rate=options.sample_rate,
                            time_length=estimated_length_pow2,
                            totalmass=total_mass,
                            inclination=0,
                            phi=0,
                            distance=1e6,
                            ex_order=3,
                            verbose=options.verbose)
            m_lower = nrwav.get_lowest_binary_mass(1100, options.f_lower)
            if m_lower > mLowMassCutoff:
                mLowMassCutoff = m_lower
            break

        # Add injections with given mass sampling
        idx = 0
        for mass in np.arange(mLowMassCutoff, options.upper_mass_threshold,
                              options.mass_sampling_step):
            # Check if NR wave can be rescaled to the lowest total mass
            if idx == 0:
                if verbose:
                    print(" .. scaling for lowest M to check ..",
                          file=sys.stdout)
                try:
                    nrwav.rescale_to_totalmass(mass)
                except IOError:
                    print(" ... FAILED to GENERATE, moving on.. ",
                          file=sys.stdout)
                    sys.stdout.flush()
                    continue
                sys.stdout.flush()
            #
            idx += 1
            #
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
            # Rescale total mass for the injection
            mfac = mass / (npoint.mass1 + npoint.mass2)
            npoint.mass1 = npoint.mass1 * mfac
            npoint.mass2 = npoint.mass2 * mfac
            npoint.mchirp = npoint.mchirp * mfac
            #
            # Zero out transverse spins if asked
            if options.zero_transverse_spins:
                if abs(npoint.spin1x) <= options.transverse_spin_threshold:
                    npoint.spin1x = 0
                if abs(npoint.spin1y) <= options.transverse_spin_threshold:
                    npoint.spin1y = 0
                if abs(npoint.spin2x) <= options.transverse_spin_threshold:
                    npoint.spin2x = 0
                if abs(npoint.spin2y) <= options.transverse_spin_threshold:
                    npoint.spin2y = 0
            else:
                print("NOT zero-ing small x,y spins")

            out_table.append(npoint)
        #
        if options.verbose:
            print(len(out_table), " points copied", file=sys.stderr)
else:
    raise IOError("Need an input catalog!")

# write the xml doc to disk
proctable = lsctables.ProcessTable.get_table(outdoc)
proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(time.time())

outname = options.output_catalog + '.xml'
ligolw_utils.write_filename(outdoc, outname)

print(len(out_table))
