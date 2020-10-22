#! /usr/bin/env python

from pycbc.waveform import get_fd_waveform
from pycbc.types import FrequencySeries, zeros
from pycbc.filter import match
import pycbc.psd

import matplotlib
matplotlib.use('Agg')
import numpy as np
from numpy import complex128, cos, float64, sin

import sys
import os

from optparse import OptionParser

import qm

from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from glue import git_version

print("STARTING THE MATCHING")
#ctx = CUDAScheme()

__author__ = "Prayush Kumar <prkumar@cita.utoronto.ca>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])
#################### Input parsing #####################
#{{{
parser = OptionParser(
    version=git_version.verbose_msg,
    usage="%prog [OPTIONS]",
    description=
    "Takes in a sub-bank file, and of proposal points. Computes overlaps between the systems in those files, and stores the maximum of these overlaps (for each proposal point) in a file match_id_part_pid.dat."
)

parser = OptionParser()
parser.add_option("--subbank-file-name",
                  help="The sub-bank file",
                  dest="subbank_file_name")
parser.add_option("--proposal-file-name",
                  help="The new points file",
                  dest="prop_file_name")
parser.add_option("--match-file-name",
                  help="The file to store matches",
                  dest="match_file_name")
parser.add_option("--subbank-batch-size",
                  help="No of bank points for which wavefors be pre-generated",
                  default=500,
                  type=int)

parser.add_option('-f',
                  '--low-frequency-cutoff',
                  metavar='FREQ',
                  help='low frequency cutoff of matched filter',
                  default=15.0,
                  type=float,
                  dest="f_min")
parser.add_option("-r",
                  "--sample-rate",
                  dest="sample_rate",
                  help="Sample Rate [Hz]",
                  default=4096,
                  type=int)
parser.add_option("-l",
                  "--signal-length",
                  help="The length of the signal (s)",
                  dest="signal_length",
                  default=256,
                  type=int)
parser.add_option("--mchirp-window", default=0.01, type=float)

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


# Miscellaneous
def outside_mchirp_window(bank, sim, w):
    #{{{
    bmchirp = None
    smchirp = None

    if hasattr(bank, "mchirp"):
        bmchirp = bank.mchirp
    elif hasattr(bank, "mass1") and hasattr(bank, "mass2"):
        bmchirp, eta = qm.m1_m2_to_mchirp_eta(bank.mass1, bank.mass2)
    elif hasattr(bank, "mtotal") and hasattr(bank, "eta"):
        bmchirp = bank.mtotal * (bank.eta**0.6)

    if hasattr(sim, "mchirp"):
        smchirp = sim.mchirp
    elif hasattr(sim, "mass1") and hasattr(sim, "mass2"):
        smchirp, eta = qm.m1_m2_to_mchirp_eta(sim.mass1, sim.mass2)
    elif hasattr(sim, "mtotal") and hasattr(sim, "eta"):
        smchirp = sim.mtotal * (sim.eta**0.6)

    if abs(smchirp - bmchirp) > (w * bmchirp):
        return True
    else:
        False
    #}}}


#########################################################################
####################### Functions to generate waveform ##################
#########################################################################
def generate_fplus_fcross(latitude, longitude, polarization):
    f_plus = -(1.0 / 2.0) * (1.0 + cos(latitude) * cos(latitude)) * cos(
        2.0 * longitude) * cos(2.0 * polarization) - cos(latitude) * sin(
            2.0 * longitude) * sin(2.0 * polarization)
    f_cross = (1.0 / 2.0) * (1.0 + cos(latitude) * cos(latitude)) * cos(
        2.0 * longitude) * sin(2.0 * polarization) - cos(latitude) * sin(
            2.0 * longitude) * cos(2.0 * polarization)
    return f_plus, f_cross


def generate_detector_strain(h_plus, h_cross, theta=0, phi=0, psi=0):
    f_plus, f_cross = generate_fplus_fcross(theta, phi, psi)
    #print "Fplus = %f, Fcross = %f" % (f_plus, f_cross)
    return (f_plus * h_plus + f_cross * h_cross)


def get_waveform(wav, f_min, dt, N):
    """This function will generate the waveform corresponding to the point taken as input"""
    #{{{
    m1 = wav.mass1
    m2 = wav.mass2

    s1x = wav.spin1x
    s1y = wav.spin1y
    s1z = wav.spin1z
    s2x = wav.spin2x
    s2y = wav.spin2y
    s2z = wav.spin2z

    inc = wav.inclination
    psi = wav.polarization
    theta = wav.latitude
    phi = wav.longitude

    df = 1. / (dt * N)
    f_max = min(1. / (2. * dt), 0.15 / ((m1 + m2) * qm.QM_MTSUN_SI))

    htild = get_fd_waveform(approximant="IMRPhenomC",
                            mass1=m1,
                            mass2=m2,
                            spin1z=s1z,
                            spin2z=s2z,
                            f_lower=f_min,
                            f_final=f_max,
                            delta_f=df)

    htilde = FrequencySeries(htild, delta_f=df, dtype=complex128, copy=True)
    href_padded = FrequencySeries(zeros(N / 2 + 1),
                                  delta_f=df,
                                  dtype=complex128,
                                  copy=True)
    href_padded[0:len(htilde)] = htilde

    return href_padded
    #}}}


#########################################################################
#################### Opening input/output files/tables ####################
#########################################################################

print("OPENING SUB-BANK/PROPOSAL FILE AND TABLES")
# Open the input sub-bank file and get the table
#{{{
if not options.subbank_file_name:
    print("No sub-bank file-name given!")
    raise ValueError("No sub-bank file-name given to %s" % PROGRAM_NAME)

if not os.path.exists(options.subbank_file_name):
    print("This sub-bank file does not exist !")
    raise IOError("The sub-bank file %s does not exist" %
                  options.subbank_file_name)

print("Opening sub-bank file %s" % options.subbank_file_name)
subbank_doc = ligolw_utils.load_filename(options.subbank_file_name,
                                         options.verbose)

try:
    subbank_table = table.get_table(subbank_doc,
                                    lsctables.SimInspiralTable.tableName)
except ValueError:
    subbank_table = table.get_table(subbank_doc,
                                    lsctables.SnglInspiralTable.tableName)

#}}}

sys.stdout.flush()

# Open the input proposals file and get the table
#{{{
if not options.prop_file_name:
    print("No proposal points file-name given!")
    raise ValueError("No proposal points file-name given to %s" % PROGRAM_NAME)

if not os.path.exists(options.prop_file_name):
    print("This proposal point file does not exist !")
    raise IOError("The proposal point file %s does not exist !" %
                  options.prop_file_name)

print("Opening proposals file %s" % options.prop_file_name)
prop_doc = ligolw_utils.load_filename(options.prop_file_name, options.verbose)

try:
    prop_table = table.get_table(prop_doc,
                                 lsctables.SimInspiralTable.tableName)
except ValueError:
    prop_table = table.get_table(prop_doc,
                                 lsctables.SnglInspiralTable.tableName)

#}}}

sys.stdout.flush()

#########################################################################
############################# Compute the Overlaps ######################
#########################################################################
# Initialize quantities for overlap calculations
f_min = options.f_min
signal_length = options.signal_length
sample_rate = options.sample_rate

dt = 1. / np.float64(sample_rate)
N = signal_length * sample_rate
print("f_min=%f, sig_len=%d, sample_rate=%d, dt=%f, N=%d" %
      (f_min, signal_length, sample_rate, dt, N))

# get the ZDHP psd
psd = pycbc.psd.from_asd_txt("/home/prayush/advLIGO_PSDs/ZERO_DET_high_P.txt",
                             N / 2 + 1, 1. / np.float64(signal_length), f_min)
#psd *= DYN_RANGE_FAC **2
psd = FrequencySeries(psd, delta_f=psd.delta_f, dtype=float64)
#print "PSD generated. First few points are %f, %f, %f..." % (psd[0],psd[1],psd[2])
sys.stdout.flush()

# Divide the sub-bank into batches of at most 500 points in each. These
# waveforms will be precomputed.
subbank_batch_size = options.subbank_batch_size
subbank_batches = [
    subbank_table[i:i + subbank_batch_size]
    for i in range(0, len(subbank_table), subbank_batch_size)
]

# Making a list with each element having a proposal point, and a list of matches
# with all sub-bank points. Initially this list is empty.
prop_point_matches = []
for prop_point in prop_table:
    prop_point_matches.append(([], prop_point))

idx = 0
print("subbank_batch_size = %d, Num of batches = %d" %
      (subbank_batch_size, len(subbank_batches)))
sys.stdout.flush()
for subbank_batch in subbank_batches:
    subbank_sims = []
    ind = 0
    for subbank_point in subbank_batch:
        #print "Getting waveform number %d in batch %d" % (ind,idx)
        #sys.stdout.flush()
        stilde = get_waveform(subbank_point, f_min, dt, N)
        subbank_sims.append((stilde, subbank_point))
        ind += 1
    print("Processing sub-bank batch %d (%d points)" %
          (idx, len(subbank_sims)))
    j = 0
    for prop_matches, prop_point in prop_point_matches:
        htilde = None
        print("\tsub-bank with point %d" % j)
        #sys.stdout.flush()
        k = 0
        for stilde, subbank_point in subbank_sims:
            # Check
            if options.mchirp_window and outside_mchirp_window(
                    prop_point, subbank_point, options.mchirp_window):
                prop_matches.append(0)
                k += 1
                continue
            if htilde is None:
                htilde = get_waveform(prop_point, f_min, dt, N)

            #print "\tcomputing overlap of proposal %d with subbank point %d" % (j,k)
            m, i = match(stilde, htilde, psd=psd, low_frequency_cutoff=f_min)
            prop_matches.append(m)
            k += 1

        j += 1
    idx += 1

print("Opening results file %s" % options.match_file_name)
if options.match_file_name:
    outfile = open(options.match_file_name, "w")
else:
    print("No Match file-name given to write the output in !")
    raise ValueError("No Match file-name given to write the output for %s" %
                     PROGRAM_NAME)

sys.stdout.flush()

for prop_matches, prop_point in prop_point_matches:
    outfile.write("%12.18f\n" % max(prop_matches))

outfile.close()
