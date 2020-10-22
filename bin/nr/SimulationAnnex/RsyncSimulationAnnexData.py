#!/usr/bin/env python
import os
import sys
import subprocess as cmd
from optparse import OptionParser

from glue import git_version
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils


@lsctables.use_in
class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
    pass


####################################################################
__author__ = "Prayush Kumar <prayush@astro.cornell.edu>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])
####################################################################
### option parsing ###
####################################################################
parser = OptionParser(version=git_version.verbose_msg,
                      usage="%prog [OPTIONS]",
                      description="""
Copy over SimulationAnnex Data from a LIGOLW XML catalog of their locations.
    """)

parser.add_option("-x",
                  "--input-catalog",
                  help="Names of the xml file to append the information to",
                  type=str,
                  default=None)

parser.add_option("-s",
                  "--start",
                  help="Index to start from [default 0]",
                  type=int,
                  default=0)
parser.add_option("-e",
                  "--end",
                  help="Index to start from [default 0]",
                  type=int,
                  default=-1)

parser.add_option("--overwrite",
                  action="store_true",
                  help="Overwrite data",
                  default=False)

parser.add_option("-V",
                  "--verbose",
                  action="store_true",
                  help="print extra debugging information",
                  default=False)

options, argv_frame_files = parser.parse_args()
####################################################################
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
    print("Waning: No catalog XML given. Exiting.")
    exit()

####################################################################
FAILED_SIMS = {}
SKIPPED_SIMS = {}
for row_id, row in enumerate(input_table):
    if row_id < options.start:
        continue
    if options.end > 0 and row_id > options.end:
        continue
    try:
        ipwd = cmd.getoutput('pwd -P')
        _ = cmd.getoutput('mkdir -p {}'.format(row.waveform))
        local_wave_dir = row.waveform.replace('/', r'\/').replace(':', r'\:')
        remote_wave_file = row.numrel_data.replace('/',
                                                   r'\/').replace(':', r'\:')
        remote_wave_filename = os.path.split(remote_wave_file)[-1]
        remote_metadata_file = remote_wave_file.replace(
            remote_wave_filename, 'metadata.txt')
        if options.overwrite or not os.path.exists(
                os.path.join(local_wave_dir, 'metadata.txt')):
            cmd_to_run = 'cd {} && rsync --progress -avL -e ssh rrsync@black-holes.org:{} .'.format(
                local_wave_dir, remote_metadata_file)
            if os.path.exists(os.path.join(
                    local_wave_dir, 'metadata.txt')) and options.verbose:
                print("Metadata being OVERWRITTEN.")
            out = cmd.getoutput(cmd_to_run)
        else:
            if options.verbose:
                print("Metadata for {} already exists. Skipping..".format(
                    local_wave_dir))
        if options.overwrite or not os.path.exists(
                os.path.join(local_wave_dir, remote_wave_filename)):
            cmd_to_run = 'cd {} && rsync --progress -avL -e ssh rrsync@black-holes.org:{} .'.format(
                local_wave_dir, remote_wave_file)
            if options.verbose:
                print(cmd_to_run)
                print(row.numrel_data)
                print(row.waveform)
            if os.path.exists(os.path.join(
                    local_wave_dir, 'metadata.txt')) and options.verbose:
                print("Waveform data being OVERWRITTEN.")
            out = cmd.getoutput(cmd_to_run)
            if options.verbose:
                print(out)
            cmd.getoutput('cd {}'.format(ipwd))
        else:
            if options.verbose:
                print("Data {} for {} already exists. Skipping..".format(
                    remote_wave_filename, local_wave_dir))
            SKIPPED_SIMS[row.waveform] = row.numrel_data
            continue
    except BaseException:
        FAILED_SIMS[row.waveform] = row.numrel_data
        cmd.getoutput('cd {}'.format(ipwd))
        continue
    if 'No such file or directory' in out:
        FAILED_SIMS[row.waveform] = row.numrel_data
####################################################################
print("Failed sims ({}):\n".format(len(FAILED_SIMS)), FAILED_SIMS)
print("Skipped sims ({}):\n".format(len(SKIPPED_SIMS)), SKIPPED_SIMS)
