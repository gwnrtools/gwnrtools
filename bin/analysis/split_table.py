#!/usr/bin/env python

import pylab
from glue import git_version
from glue.ligolw.utils import process as ligolw_process
from glue.ligolw import utils as ligolw_utils
import time
import sys

from optparse import OptionParser

from glue import gpstime

from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw.ligolw import LIGOLWContentHandler


class mycontenthandler(LIGOLWContentHandler):
    pass


lsctables.use_in(mycontenthandler)

params = {'text.usetex': True}
pylab.rcParams.update(params)

### option parsing ###

parser = OptionParser(
    version=git_version.verbose_msg,
    usage="%prog [OPTIONS]",
    description="Creates a template bank and writes it to XML.")

parser.add_option('-n',
                  '--num',
                  metavar='SAMPLES',
                  help='number of templates in the output banks',
                  type=int)
parser.add_option('-s',
                  '--start-num',
                  metavar='SAMPLES',
                  help='Starting index for numbering output banks',
                  type=int,
                  default=0)
parser.add_option("-t",
                  "--tmplt-bank",
                  metavar='file',
                  help='template bank to split')
parser.add_option("-V",
                  "--verbose",
                  action="store_true",
                  help="print extra debugging information",
                  default=False)
parser.add_option("-e",
                  "--named",
                  help="Starting string in the names of final XMLs")

options, argv_frame_files = parser.parse_args()
print(options.named)

print(options.named)
indoc = ligolw_utils.load_filename(options.tmplt_bank,
                                   contenthandler=mycontenthandler)

try:
    template_bank_table = table.get_table(
        indoc, lsctables.SnglInspiralTable.tableName)
    tabletype = lsctables.SnglInspiralTable
except BaseException:
    template_bank_table = table.get_table(indoc,
                                          lsctables.SimInspiralTable.tableName)
    tabletype = lsctables.SimInspiralTable

# print tabletype
length = len(template_bank_table)
num_files = int(round(length / options.num + .5))

for num in range(num_files):

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

    sngl_inspiral_table = lsctables.New(
        tabletype, columns=template_bank_table.columnnames)
    outdoc.childNodes[0].appendChild(sngl_inspiral_table)

    for i in range(options.num):
        try:
            sngl_inspiral_table.append(template_bank_table.pop())
        except IndexError:
            break

    # write the xml doc to disk
    proctable = table.get_table(outdoc, lsctables.ProcessTable.tableName)
    proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(time.time())

    outname = options.named + str(num + options.start_num) + '.xml'
    ligolw_utils.write_filename(outdoc, outname)

    if len(sngl_inspiral_table) == 0:
        import subprocess as cmd
        print("Removing %s, since it has no elements" % outname)
        cmd.getoutput('rm -f %s' % outname)

print(num_files)
sys.exit(int(num_files))
