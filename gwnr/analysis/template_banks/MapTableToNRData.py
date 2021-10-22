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

import time
import os
import sys

from optparse import OptionParser

from glue import git_version
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils

import h5py
import numpy as np


@lsctables.use_in
class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
    pass


__author__ = "Prayush Kumar <prkumar@cita.utoronto.ca>"

PROGRAM_NAME = os.path.abspath(sys.argv[0])

__itime__ = time.time()

################################################################################
### Function Definitions ###
################################################################################


def invert_tabletype(tabletype):
    if tabletype == lsctables.SnglInspiralTable:
        return lsctables.SimInspiralTable
    if tabletype == lsctables.SimInspiralTable:
        return lsctables.SnglInspiralTable
    raise IOError("Input table type neither Sim or Sngl Inspiral")


def new_row(tabletype):
    if tabletype == lsctables.SnglInspiralTable:
        return lsctables.SnglInspiral()
    if tabletype == lsctables.SimInspiralTable:
        return lsctables.SimInspiral()
    raise IOError("Input table type neither Sim or Sngl Inspiral")


def does_this_map(p, c, param='eta', return_err=False, verbose=False):
    """
  Judge if the given template 'p' matches the given catalog row 'c' for the param
    """
    val1 = p.__getattribute__(param)
    val2 = c.__getattribute__(param)

    if param in list(EPS_Params.keys()):
        eps = EPS_Params[param]
    else:
        eps = EPS_Default

    num_err = np.abs(val1 - val2)
    den_err = np.abs((val1 + val2) / 2.)
    if den_err == 0:
        frac_err = num_err
    else:
        frac_err = num_err / den_err

    if verbose:
        print("Param error: %.2e, Threshold: %.2e" % (frac_err, eps))

    if frac_err <= eps:
        if return_err:
            return True, frac_err
        else:
            return True
    else:
        if return_err:
            return False, frac_err
        else:
            return False


################################################################################
### option parsing ###
################################################################################
parser = OptionParser(version=git_version.verbose_msg,
                      usage="%prog [OPTIONS]",
                      description="""\
Takes in a sngl inspiral table of templates, and a sim inspiral catalog table 
for the NR simulations. Matches the physical parameters of the templates with 
the simulations, and writes the mappings in the input mapping file.

This script is meant for constructing the mapping file, which will be used in
the template generation code, for NR-hybrid template banks.
    """)

parser.add_option("-c",
                  "--input-catalog",
                  help="XML file with NR catalog information",
                  type=str,
                  default=None)
parser.add_option("-b",
                  "--input-bank",
                  help="Name of the template bank xml file",
                  type=str,
                  default=None)
parser.add_option("-m",
                  "--output-mapfile",
                  help='output file to store template-to-NR-data mappings')
parser.add_option("-a",
                  "--need-all-mappings",
                  action="store_true",
                  help="Fail if any template is not mapped",
                  default=False)
parser.add_option("-f",
                  "--force-file-exists",
                  action="store_true",
                  help="ensure that the NR data file exists for each case",
                  default=False)
parser.add_option("-V",
                  "--verbose",
                  action="store_true",
                  help="print extra debugging information",
                  default=False)

options, argv_frame_files = parser.parse_args()

################################################################################
# READ IN INPUT NR CATALOGS AND INPUT TEMPLATE BANK
################################################################################
if options.input_catalog is not None:
    indoc = ligolw_utils.load_filename(options.input_catalog,
                                       contenthandler=LIGOLWContentHandler,
                                       verbose=options.verbose)
    #
    try:
        input_catalog = lsctables.SnglInspiralTable.get_table(indoc)
        inputtabletype = lsctables.SnglInspiralTable
    except:
        input_catalog = lsctables.SimInspiralTable.get_table(indoc)
        inputtabletype = lsctables.SimInspiralTable
    #
    # print tabletype
    length_catalog = len(input_catalog)
else:
    raise IOError("No input CATALOG given for NR data..")

if options.input_bank is not None:
    indoc = ligolw_utils.load_filename(options.input_bank,
                                       contenthandler=LIGOLWContentHandler,
                                       verbose=options.verbose)
    #
    try:
        input_bank = lsctables.SnglInspiralTable.get_table(indoc)
        inputtabletype = lsctables.SnglInspiralTable
    except:
        input_bank = lsctables.SimInspiralTable.get_table(indoc)
        inputtabletype = lsctables.SimInspiralTable
    #
    # print tabletype
    length_bank = len(input_bank)
else:
    raise IOError("No TEMPLATE BANK given..")

if options.output_mapfile is not None:
    try:
        fout = h5py.File(options.output_mapfile, 'a')
    except:
        raise IOError("could not create OUTPUT MAP file %s" %
                      options.output_mapfile)
    else:
        fout.create_group('TemplateBankToNRMappings-DataLocation')
        gout = fout['TemplateBankToNRMappings-DataLocation']
else:
    raise IOError(
        "No file-name given for storing TEMPLATE to NR-DATA MAPPINGS")

################################################################################
# Define matching parameters AND acceptable difference levels
################################################################################
if options.verbose:
    print("""\
 Defining fractional level within which a template is considered matched
 with a given column in the NR catalog.
""",
          file=sys.stdout)

EPS_Params = {
    'eta': 1.e-7,
    'spin1x': 1.e-7,
    'spin1y': 1.e-7,
    'spin1z': 1.e-7,
    'spin2x': 1.e-7,
    'spin2y': 1.e-7,
    'spin2z': 1.e-7
}
EPS_Default = 1.e-7

if options.verbose:
    print("EPS_Detault = ", EPS_Default, "\n", EPS_Params, file=sys.stdout)
    sys.stdout.flush()

# This is the list of parameters which are to be matched within the error
# tolerances specified above
params_tested = [
    'eta', 'spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z'
]

################################################################################
# Figure out the mappings between template and NR data
################################################################################
if options.verbose:
    print("NOW MAPPING TEMPLATES..")
NOUTPUT = 1000

for idx, tmplt in enumerate(input_bank):
    if options.verbose and idx % NOUTPUT == 0:
        print("\n .. matching template %d" % idx, file=sys.stderr)
    #
    # Map template to catalog row
    for jdx, row in enumerate(input_catalog):
        is_map = True
        for param in params_tested:
            if not does_this_map(tmplt, row, param=param, verbose=False):
                is_map = False
                break
        if is_map:
            break
    #
    # If map is found, write it
    if is_map:
        colstring = str(tmplt.event_id)
        mapstring = str(row.numrel_data)
        #
        if options.verbose and idx % NOUTPUT == 0:
            print(" .. .. matched to %s" % mapstring, file=sys.stderr)
            print(" .. .. .. file exists: ",
                  os.path.exists(mapstring),
                  file=sys.stderr)
        #
        # Ensure that the mapped-to NR data file exists
        if options.force_file_exists:
            if not os.path.exists(mapstring):
                raise RuntimeError("File %s not found!\n" % mapstring)
            else:
                if os.path.getsize(mapstring) == 0:
                    raise RuntimeError("File %s has size 0\n" % mapstring)
        #
        gout.create_dataset(colstring, data=mapstring)
    else:
        # Else print out the maximum difference between the template's and catalog's parameters
        errs = []
        for param in params_tested:
            _, err = does_this_map(tmplt,
                                   row,
                                   param=param,
                                   return_err=True,
                                   verbose=False)
            errs.append(err)
        #
        if options.verbose:
            print(
                "Template %d does not match any of the rows in the NR catalog table."
                % idx,
                file=sys.stdout)
            print("MAX ERROR for rejection..: %.2e\n" % np.array(errs).max())
        #
        # Enforce that "ALL" mappings must be found!!
        if options.need_all_mappings:
            raise RuntimeError("Template %d could not be matched" % idx)
    #
    if options.verbose:
        sys.stdout.flush()
        sys.stderr.flush()


def get_dir_from_filepath(fp):
    ap = os.path.abspath(fp)
    ap = ap.split('/')[:-1]
    out = ''
    for s in ap:
        out = out + s + '/'
    return out


def get_file_from_filepath(fp):
    return fp.split('/')[-1]


if options.verbose:
    print("\n\n >> Total %d/%d templates mapped ..!" %
          (len(list(gout.keys())), len(input_bank)),
          file=sys.stdout)
    sys.stdout.flush()

# write the HDF mappling file to disk
fout.close()

print("""
\n\n
################################################################################

Please add the path to %s to your NR_CATALOG_PATH,
 and %s to NR_CATALOG_FILE with:

export NR_CATALOG_PATH=%s:$NR_CATALOG_PATH
export NR_CATALOG_FILE=%s

################################################################################
\n""" % (get_dir_from_filepath(
    options.output_mapfile), get_file_from_filepath(
        options.output_mapfile), get_dir_from_filepath(options.output_mapfile),
         get_file_from_filepath(options.output_mapfile)))

print(" Total %f seconds taken" % (time.time() - __itime__))
