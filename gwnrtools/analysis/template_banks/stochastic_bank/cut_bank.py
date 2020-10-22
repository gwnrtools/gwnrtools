#! /usr/bin/env python

from glue import git_version
from glue.ligolw.utils import process as ligolw_process
from glue.ligolw import utils as ligolw_utils
from glue.ligolw import lsctables
from glue.ligolw import table
from glue.ligolw import ligolw
from glue import gpstime
from optparse import OptionParser
import time
import os
import sys
from numpy import pi
import matplotlib
matplotlib.use('Agg')

__author__ = "Prayush Kumar <prkumar@cita.utoronto.ca>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])
############## Functions ######################


class test:
    # def __init__(self):
    #self.m1 = m1
    #self.m2 = m2
    QM_MTSUN_SI = 4.9254909500000001e-06

    @classmethod
    def m1_m2_to_mchirp_eta(self, m1, m2):
        mt = m1 + m2
        et = m1 * m2 / mt**2
        mc = mt * et**0.6
        return (mc, et)

    def m1_m2_to_tau0_tau3(self, m1, m2):
        mt = m1 + m2
        et = m1 * m2 / mt**2
        tau3 = 1.0 / (8.0 * (pi * pi * options.f_min**5.0)**(1.0 / 3.0) *
                      mt**(2.0 / 3.0) * et)
        tau0 = 5.0 / (256.0 * pi * options.f_min**(8.0 / 3.0) *
                      mt**(5.0 / 3.0) * et)
        return (tau0, tau3)

    def m1_m2_to_tau0_tau3(self, m1, m2, f_min):
        mt = m1 + m2
        et = m1 * m2 / mt**2
        tau3 = 1.0 / (8.0 * (pi * pi * f_min**5.0)**(1.0 / 3.0) *
                      mt**(2.0 / 3.0) * et)
        tau0 = 5.0 / (256.0 * pi * f_min**(8.0 / 3.0) * mt**(5.0 / 3.0) * et)
        return (tau0, tau3)

    def mchirp_eta_to_m1_m2(mchirp, eta):
        mt = mchirp / eta**0.6
        tmp = (mt**2 * (1. - 4. * eta))**0.5
        m1 = (mt + tmp) / 2.
        m2 = (mt - tmp) / 2.
        return (m1, m2)


qm = test()


def within_region_nonspinning(m1, m2):
    eta = m1 * m2 / (m1 + m2)**2
    mc = (m1 + m2) * eta**0.6
    feta = -7011.5 * eta**4 + 5169.4 * eta**3 - \
        1469.6 * eta**2 + 217.76 * eta + 12.02
    if mc >= feta:
        return True
    else:
        return False


def within_region_alignedspin(mt, et, chi):
    feta = et**-0.6 * 12.9241 * \
        (1. + 15.1506 * et - 96.7042 * et**2 + 305.441 * et**3 - 342.732 * et**4) * \
        (1. + 0.304976 * chi + 0.140933 * chi **
         2 + 0.199598 * chi**3 + 0.151902 * chi**4)
    # lower feta a bit
    feta = feta - 2.
    print(feta, file=sys.stderr)
    if mt >= feta:
        return True
    else:
        return False


#########################################################################
#################### Input parsing #####################
#########################################################################
# {{{
parser = OptionParser(
    version=git_version.verbose_msg,
    usage="%prog [OPTIONS]",
    description=
    "Takes in the iteration id. Reads in bank_id.xml. Writes bank_id_part_pid.xml, where each of these part files have bank-batch-size consecutive elements of the bank_id.xml. pid is part-id."
)

parser = OptionParser()
parser.add_option("--input-bank", help="input bank", type=str)
parser.add_option("--output-bank", help="output bank", type=str)
parser.add_option("--columns", help="output bank columns", type=str)
parser.add_option("--table", help="sngl or sim", default='sim', type=str)
parser.add_option("--mass-cut", help="component mass cut", type=float)
parser.add_option("--mtotal-cut", help="total mass cut", type=float)
parser.add_option("--mchirp-cut", help="chirp mass cut", type=float)
parser.add_option("--eta-cut", help="eta cut", type=float)
parser.add_option("--upper-spinz-cut", help="spinz cut", type=float)
parser.add_option("--lower-spinz-cut", help="spinz cut", type=float)
parser.add_option("--extract-eta",
                  help="single eta to be extracted",
                  type=float)
parser.add_option("--extract-q", help="single q to be extracted", type=float)
parser.add_option(
    "--strict-region",
    action="store_true",
    help="choose templates in a region, which has to be specified",
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
# }}}

if options.input_bank is None or options.output_bank is None:
    raise IOError("PLease give input AND output bank names")

if options.mtotal_cut is None and options.mchirp_cut is None and options.eta_cut is None and options.mass_cut is None and options.extract_eta is None and options.extract_q is None and not options.strict_region and options.upper_spinz_cut is None and options.lower_spinz_cut is None:
    raise IOError("Please specify one cut to apply")

################### Read in the bank file, and get the sim table ############
bank_file_name = options.input_bank
in_bank_doc = ligolw_utils.load_filename(bank_file_name, options.verbose)
try:
    in_bank_table = table.get_table(in_bank_doc,
                                    lsctables.SimInspiralTable.tableName)
except ValueError:
    in_bank_table = table.get_table(in_bank_doc,
                                    lsctables.SnglInspiralTable.tableName)

################### Cut the bank amd write it to disk ############
subfile_name = options.output_bank
print("Writing sub-bank file %s" % subfile_name)
out_subbank_doc = ligolw.Document()
out_subbank_doc.appendChild(ligolw.LIGO_LW())
out_proc_id = ligolw_process.register_to_xmldoc(
    out_subbank_doc,
    PROGRAM_NAME,
    options.__dict__,
    comment=options.comment,
    version=git_version.id,
    cvs_repository=git_version.branch,
    cvs_entry_time=git_version.date).process_id

print(options.columns)
if options.columns is None:
    if options.table == 'sim':
        out_subbank_table = lsctables.New(
            lsctables.SimInspiralTable,
            columns=[
                'mass1', 'mass2', 'mchirp', 'eta', 'spin1x', 'spin1y',
                'spin1z', 'spin2x', 'spin2y', 'spin2z', 'inclination',
                'polarization', 'latitude', 'longitude', 'bandpass', 'alpha',
                'alpha1', 'alpha2', 'process_id'
            ])
    else:
        out_subbank_table = lsctables.New(
            lsctables.SnglInspiralTable,
            columns=[
                'mass1', 'mass2', 'mchirp', 'eta', 'spin1x', 'spin1y',
                'spin1z', 'spin2x', 'spin2y', 'spin2z'
            ]
        )  # ,'inclination','polarization','latitude','longitude','bandpass','alpha','alpha1','alpha2','process_id'])
else:
    cols = []
    for col in options.columns.split():
        cols.append(col)
    if options.table == 'sim':
        out_subbank_table = lsctables.New(lsctables.SimInspiralTable,
                                          columns=cols)
    else:
        out_subbank_table = lsctables.New(lsctables.SnglInspiralTable,
                                          columns=cols)

out_subbank_doc.childNodes[0].appendChild(out_subbank_table)
for bank_point in in_bank_table:
    m1 = bank_point.mass1
    m2 = bank_point.mass2
    mc, et = qm.m1_m2_to_mchirp_eta(m1, m2)
    mt = m1 + m2
    if options.strict_region:
        if within_region_alignedspin(mt, et, bank_point.spin1z):
            out_subbank_table.append(bank_point)
    elif options.mass_cut is not None:
        if m1 <= options.mass_cut and m2 <= options.mass_cut:
            out_subbank_table.append(bank_point)
    elif options.mtotal_cut is not None:
        if mt <= options.mtotal_cut:
            out_subbank_table.append(bank_point)
    elif options.mchirp_cut is not None:
        if mc <= options.mchirp_cut:
            out_subbank_table.append(bank_point)
    elif options.eta_cut is not None:
        if et <= options.eta_cut:
            out_subbank_table.append(bank_point)
    elif options.extract_eta is not None or options.extract_q is not None:
        if options.extract_q is not None:
            options.extract_eta = options.extract_q / \
                (1. + options.extract_q)**2
        if et <= 1.001 * options.extract_eta and et >= 0.999 * options.extract_eta:
            out_subbank_table.append(bank_point)
    elif options.lower_spinz_cut is not None or options.upper_spinz_cut is not None:
        if options.lower_spinz_cut is None:
            options.lower_spinz_cut = -1.
        if options.upper_spinz_cut is None:
            options.upper_spinz_cut = 1.
        if bank_point.spin1z >= options.lower_spinz_cut and bank_point.spin1z <= options.upper_spinz_cut and bank_point.spin2z >= options.lower_spinz_cut and bank_point.spin2z <= options.upper_spinz_cut:
            out_subbank_table.append(bank_point)
        else:
            if bank_point.spin1z > options.upper_spinz_cut:
                bank_point.spin1z = 0.9
            else:
                bank_point.spin1z = -0.9
            if bank_point.spin2z > options.upper_spinz_cut:
                bank_point.spin2z = 0.9
            else:
                bank_point.spin2z = -0.9
            out_subbank_table.append(bank_point)

subbank_proctable = table.get_table(out_subbank_doc,
                                    lsctables.ProcessTable.tableName)
subbank_proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(time.time())
ligolw_utils.write_filename(out_subbank_doc, subfile_name)
