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

from glue import gpstime, git_version
from glue.ligolw import ligolw, ilwd
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import process as ligolw_process


@lsctables.use_in
class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
    pass


itime = time.time()

### Function Definitions ###


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


### option parsing ###

parser = OptionParser(version=git_version.verbose_msg,
                      usage="%prog [OPTIONS]",
                      description="""\
    Takes in a Sim or Sngl InspiralTable file. Copies over all the columns to 
    a new file of the opposite table type. The columns which are not copyable,
    are printed out to the user. 
    
    In FUTURE: ALLOW USER to provide COLUMN MAPPINGS 
    
    This script is NOT meant to be part of any workflow, as table conversion is
    flaky at best.
    """)

parser.add_option("-x",
                  "--input-catalog",
                  help="Names of the xml file to append the information to",
                  type=str,
                  default=None)
parser.add_option("-t", "--output-catalog", help='output file name')

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

# Get output table type
outputtabletype = invert_tabletype(inputtabletype)

# Determined which columns can be copied over..
tmplt = input_table[0]
tslots = []
for col in tmplt.__slots__:
    try:
        tmp = tmplt.__getattribute__(col)
    except:
        continue
    else:
        tslots.append(col)
tslots = set(tslots)

outtmplt = new_row(outputtabletype)
oslots = set(outtmplt.__slots__)

outcols = list(oslots.intersection(tslots))
uncopiedcols = list(tslots - oslots)

if options.verbose:
    print("##### The followings columns ARE NOT copied over: ####\n",
          file=sys.stdout)
    print(uncopiedcols, file=sys.stdout)
    print("\n\n", file=sys.stdout)
    sys.stdout.flush()

if options.verbose:
    print("##### The followings columns ARE copied over: ####\n",
          file=sys.stdout)
    print(outcols, file=sys.stdout)
    print("\n\n", file=sys.stdout)
    sys.stdout.flush()

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

# Create output table
if outputtabletype == lsctables.SnglInspiralTable:
    outcols.append('event_id')

out_table = lsctables.New(outputtabletype, columns=outcols)
outdoc.childNodes[0].appendChild(out_table)

for idx, tmplt in enumerate(input_table):
    if options.verbose and idx % 1000 == 0:
        print(" .. copying row %d" % idx, file=sys.stderr)

    newp = new_row(outputtabletype)

    # First copy over all columns that can be possibly copied over
    for col in outcols:
        if col in tmplt.__slots__:
            newp.__setattr__(col, tmplt.__getattribute__(col))
        else:
            continue

    # Now create an identifier ID
    if outputtabletype == lsctables.SnglInspiralTable:
        newp.event_id = ilwd.ilwdchar("sngl_inspiral:event_id:%d" % idx)
        if options.verbose and idx % 1000 == 0:
            print(newp.event_id)

    newp.__setattr__('process_id', proc_id)

    # Add the point to the output table
    out_table.append(newp)

# write the xml doc to disk
proctable = lsctables.ProcessTable.get_table(outdoc)
proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(time.time())

outname = options.output_catalog
if '.xml' not in outname:
    outname = outname + '.xml'

ligolw_utils.write_filename(outdoc, outname, gz=outname.endswith('.gz'))

print(len(out_table))
print(" Total %f seconds taken" % (time.time() - itime))
