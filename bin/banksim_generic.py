#!/usr/bin/env python
import sys
import os, logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',\
                     level=logging.INFO, stream=sys.stdout)
import time

_itime = time.time()

import argparse
import numpy as np

import gwnr.analysis as DA
import gwnr.waveform as WF

import lal
from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from glue.ligolw import ilwd

from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.waveform import td_approximants, fd_approximants
from pycbc.types import FrequencySeries, TimeSeries, zeros
from pycbc.filter import make_frequency_series, match
from pycbc.detector import overhead_antenna_pattern
import pycbc.psd

#ctx = CUDAScheme()

__author__ = "Prayush Kumar <prayush@astro.cornell.edu>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])
#########################################################################
####################       Input parsing     #####################
#########################################################################
#{{{
parser = argparse.ArgumentParser(
    usage="%%prog [OPTIONS]",
    description="""
Takes in a sub-bank and proposal points (as XML). Computes overlaps between
the systems in those files, and stores the maximum of these overlaps (for
each proposal point) in a file match_id_part_pid.dat.
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# IO related inputs
parser.add_argument("--bank-file-name",
                    dest="bank_file_name",
                    help="The bank file")
parser.add_argument("--proposal-file-name",
                    dest="prop_file_name",
                    help="The new points file")
parser.add_argument("--match-file-name",
                    dest="match_file_name",
                    help="The file to store matches")

# Physics related inputs
parser.add_argument("--bank-approximant",
                    help="Waveform Approximant for bank templates",
                    default="EccentricFD")
parser.add_argument("--proposal-approximant",
                    help="Waveform Approximant for proposed templates",
                    default="EccentricFD")

# Filtering related inputs
parser.add_argument("--bank-batch-size",
                    help="No of points for which wavefors be pre-generated",
                    default=100,
                    type=int)
parser.add_argument("--proposal-batch-size",
                    help="No of points for which wavefors be pre-generated",
                    default=100,
                    type=int)
parser.add_argument('-f',
                    '--low-frequency-cutoff',
                    metavar='FREQ',
                    dest="f_min",
                    help='low frequency cutoff of matched filter',
                    default=15.0,
                    type=float)
parser.add_argument("-r",
                    "--sample-rate",
                    dest="sample_rate",
                    help="Sample Rate [Hz]",
                    default=4096,
                    type=int)
parser.add_argument("-l",
                    "--signal-length",
                    dest="signal_length",
                    help="The length of the signal (s)",
                    default=128,
                    type=int)
parser.add_argument("--mchirp-window", default=0.01, type=float)
parser.add_argument("--eccentricity-window",
                    dest="ecc_window",
                    default=0.01,
                    type=float)
parser.add_argument("--minimal-match", dest="mm", default=0.97, type=float)
parser.add_argument("--psd-name",
                    metavar="STRING",
                    help="Name of PSD to be used.",
                    default="aLIGOZeroDetHighPower")
parser.add_argument("-E",
                    "--eliminate",
                    action="store_true",
                    help="""If enabled, and if any proposal point has overlap
over the desired threshold, then that proposal point is eliminated instantly""",
                    default=False)
parser.add_argument("--elimination-dir",
                    metavar="STRING",
                    help="Where are hashes of eliminated proposals stored?",
                    default='testpoints_eliminated/')

# Miscellaneous
parser.add_argument("-V",
                    "--verbose",
                    action="store_true",
                    help="print extra debugging information",
                    default=False)
parser.add_argument("-C",
                    "--comment",
                    metavar="STRING",
                    help="add the optional STRING as the process:comment",
                    default='')

options = parser.parse_args()
#}}}

#########################################################################
####################### Functions to generate waveform ##################
#########################################################################
# Miscellaneous
#outside_mchirp_window = DA.outside_mchirp_window
generate_fplus_fcross = overhead_antenna_pattern
generate_detector_strain = WF.generate_detector_strain


#############################
def outside_mchirp_window(bank, sim, w):
    #{{{
    bmchirp = None
    smchirp = None

    if hasattr(bank, "mchirp"):
        bmchirp = bank.mchirp
    elif hasattr(bank, "mass1") and hasattr(bank, "mass2"):
        bmchirp, eta = pnutils.mass1_mass2_to_mchirp_eta(
            bank.mass1, bank.mass2)
    elif hasattr(bank, "mtotal") and hasattr(bank, "eta"):
        bmchirp = bank.mtotal * (bank.eta**0.6)

    if hasattr(sim, "mchirp"):
        smchirp = sim.mchirp
    elif hasattr(sim, "mass1") and hasattr(sim, "mass2"):
        smchirp, eta = pnutils.mass1_mass2_to_mchirp_eta(sim.mass1, sim.mass2)
    elif hasattr(sim, "mtotal") and hasattr(sim, "eta"):
        smchirp = sim.mtotal * (sim.eta**0.6)

    if abs(smchirp - bmchirp) > (w * bmchirp):
        return True

    return False
    #}}}


def outside_ecc_window(bank, sim, w):
    #{{{
    b_ecc = 0
    s_ecc = 0
    if hasattr(bank, "alpha"):
        b_ecc = bank.alpha
    if hasattr(sim, "alpha"):
        s_ecc = sim.alpha
    if abs(s_ecc - b_ecc) > w:
        return True
    return False
    #}}}


def get_waveform(wav, approximant, f_min, dt, N):
    """This function will generate the waveform corresponding to the point
    taken as input"""
    #{{{
    m1 = wav.mass1
    m2 = wav.mass2

    s1x = wav.spin1x
    s1y = wav.spin1y
    s1z = wav.spin1z
    s2x = wav.spin2x
    s2y = wav.spin2y
    s2z = wav.spin2z

    ecc = wav.alpha
    mean_per_ano = wav.alpha1
    long_asc_nodes = wav.alpha2
    coa_phase = wav.coa_phase

    inc = wav.inclination
    dist = wav.distance

    df = 1. / (dt * N)
    f_max = min(1. / (2. * dt), 0.15 / ((m1 + m2) * lal.MTSUN_SI))

    if approximant in fd_approximants():
        try:
            hptild, hctild = get_fd_waveform(approximant=approximant,
                                             mass1=m1,
                                             mass2=m2,
                                             spin1x=s1x,
                                             spin1y=s1y,
                                             spin1z=s1z,
                                             spin2x=s2x,
                                             spin2y=s2y,
                                             spin2z=s2z,
                                             eccentricity=ecc,
                                             mean_per_ano=mean_per_ano,
                                             long_asc_nodes=long_asc_nodes,
                                             coa_phase=coa_phase,
                                             inclination=inc,
                                             distance=dist,
                                             f_lower=f_min,
                                             f_final=f_max,
                                             delta_f=df)
        except RuntimeError as re:
            for c in dir(wav):
                if "__" not in c and "get" not in c and "set" not in c and hasattr(
                        wav, c):
                    print(c, getattr(wav, c))
            raise RuntimeError(re)
        hptilde = FrequencySeries(hptild,
                                  delta_f=df,
                                  dtype=np.complex128,
                                  copy=True)
        hpref_padded = FrequencySeries(zeros(N / 2 + 1),
                                       delta_f=df,
                                       dtype=np.complex128,
                                       copy=True)
        hpref_padded[0:len(hptilde)] = hptilde
        hctilde = FrequencySeries(hctild,
                                  delta_f=df,
                                  dtype=np.complex128,
                                  copy=True)
        hcref_padded = FrequencySeries(zeros(N / 2 + 1),
                                       delta_f=df,
                                       dtype=np.complex128,
                                       copy=True)
        hcref_padded[0:len(hctilde)] = hctilde
        href_padded = generate_detector_strain(wav, hpref_padded, hcref_padded)
    elif approximant in td_approximants():
        #raise IOError("Time domain approximants not supported at the moment..")
        try:
            hp, hc = get_td_waveform(approximant=approximant,
                                     mass1=m1,
                                     mass2=m2,
                                     spin1x=s1x,
                                     spin1y=s1y,
                                     spin1z=s1z,
                                     spin2x=s2x,
                                     spin2y=s2y,
                                     spin2z=s2z,
                                     eccentricity=ecc,
                                     mean_per_ano=mean_per_ano,
                                     long_asc_nodes=long_asc_nodes,
                                     coa_phase=coa_phase,
                                     inclination=inc,
                                     distance=dist,
                                     f_lower=f_min,
                                     delta_t=dt)
        except RuntimeError as re:
            for c in dir(wav):
                if "__" not in c and "get" not in c and "set" not in c and hasattr(
                        wav, c):
                    print(c, getattr(wav, c))
            raise RuntimeError(re)
        hpref_padded = TimeSeries(zeros(N),
                                  delta_t=dt,
                                  dtype=hp.dtype,
                                  copy=True)
        hpref_padded[:len(hp)] = hp
        hcref_padded = TimeSeries(zeros(N),
                                  delta_t=dt,
                                  dtype=hc.dtype,
                                  copy=True)
        hcref_padded[:len(hc)] = hc
        href_padded_td = generate_detector_strain(wav, hpref_padded,
                                                  hcref_padded)
        href_padded = make_frequency_series(href_padded_td)
    return href_padded
    #}}}


def get_sim_hash(N=1, num_digits=10):
    return ilwd.ilwdchar(":%s:0" %
                         DA.get_unique_hex_tag(N=N, num_digits=num_digits))


def get_tag(wav):
    return str(wav.simulation_id.column_name)


def waveform_exists(wav, waves):
    sid = get_tag(wav)
    if sid in waves: return True
    return False


def is_eliminated(wav):
    sid = get_tag(wav)
    if os.path.exists(os.path.join(options.elimination_dir, sid)): return True
    return False


#########################################################################
#################### Opening input/output files/tables ##################
#########################################################################
logging.info("OPENING SUB-BANK/PROPOSAL FILE AND TABLES")
# Open the input sub-bank file and get the table
#{{{
if not options.bank_file_name:
    logging.info("No sub-bank file-name given!")
    raise ValueError("No sub-bank file-name given to %s" % PROGRAM_NAME)

if not os.path.exists(options.bank_file_name):
    logging.info("This sub-bank file does not exist !")
    raise IOError("The sub-bank file %s does not exist" %
                  options.bank_file_name)

if options.verbose:
    logging.info("..Opening bank file %s" % options.bank_file_name)
bank_doc = ligolw_utils.load_filename(options.bank_file_name,
                                      contenthandler=table.use_in(
                                          ligolw.LIGOLWContentHandler),
                                      verbose=options.verbose)
try:
    bank_table = lsctables.SimInspiralTable.get_table(bank_doc)
except ValueError:
    try:
        bank_table = lsctables.SnglInspiralTable.get_table(bank_doc)
        # We need to assign unique tags to all rows in this
        # table as we won't have those available under the
        # `simulation_id` column. We also addtionally include
        # sensible values for `inclination`, `distance`,
        # `latitude`, `longitude` and `polarization` fields!
        for row in bank_table:
            row.simulation_id = get_sim_hash()
            row.inclination = 0
            row.distance = 1.e6
            row.latitude = 0
            row.longitude = 0
            row.polarization = 0
    except ValueError:
        raise IOError("Only sim_inspiral tables are understood for banks..")
#}}}

# Open the input proposals file and get the table
#{{{
if not options.prop_file_name:
    logging.info("No proposal points file-name given!")
    raise ValueError("No proposal points file-name given to %s" % PROGRAM_NAME)

if not os.path.exists(options.prop_file_name):
    logging.info("This proposal point file does not exist !")
    raise IOError(\
        "The proposal point file %s does not exist !" % options.prop_file_name)

logging.info("Opening proposals file %s" % options.prop_file_name)
prop_doc = ligolw_utils.load_filename(options.prop_file_name,
                                      contenthandler=table.use_in(
                                          ligolw.LIGOLWContentHandler),
                                      verbose=options.verbose)
try:
    prop_table = lsctables.SimInspiralTable.get_table(prop_doc)
except ValueError:
    raise IOError("Only sim_inspiral tables are understood for proposals..")
#}}}

sys.stdout.flush()

#########################################################################
#############################   Compute Overlaps   ######################
#########################################################################
# Initialize quantities for overlap calculations
f_min = options.f_min
signal_length = options.signal_length
sample_rate = options.sample_rate

dt = 1. / float(sample_rate)
N = signal_length * sample_rate
n = N / 2 + 1
df = 1. / float(signal_length)
if options.verbose:
    logging.info("f_min={}, sig_len={}, sample_rate={}, dt={}, N={}".format(f_min,\
        signal_length,sample_rate,dt,N))

# GET psd
psd = pycbc.psd.from_string(options.psd_name, n, df, f_min)

##########################################################
### Note on algorithm to follow:-
## 0) Eliminate calculations as much as possible.
##    - use mchirp_window
##    - use elimination
##    - use self vs self
##    - if table is empty..
## 1) Generate waveform only and only if it is to be used in an overlap calc.
## 2) Once generated, store every waveform indexed by its HASH. No need for
##     separate bank and proposal dicts, as HASHes must be unique!
## 3) Finally compute matches

##########################################################
# Storage
waveforms = {}
matches = {}

##########################################################
# Split into batches
bank_batch_size = options.bank_batch_size
if len(bank_table) == 0:
    try:
        os.mknod(options.match_file_name)
    except OSError:
        pass
    logging.info("Bank file {} is empty. Exiting!".format(
        options.bank_file_name))
    sys.exit(0)
elif len(bank_table) <= bank_batch_size:
    bank_batches = [bank_table]
else:
    bank_batches=[bank_table[i:i+bank_batch_size] for i in range(0,\
                                              len(bank_table), bank_batch_size)]

# eliminate eliminated points from now only!
if options.eliminate:
    for p in prop_table:
        if is_eliminated(p):
            prop_table.remove(p)

prop_batch_size = options.proposal_batch_size
if len(prop_table) == 0:
    try:
        os.mknod(options.match_file_name)
    except OSError:
        pass
    logging.info("Proposal file {} is empty. Exiting!".format(
        options.prop_file_name))
    sys.exit(0)
elif len(prop_table) <= prop_batch_size:
    prop_batches = [prop_table]
else:
    prop_batches = [prop_table[i:i+prop_batch_size] for i in range(0,\
                                              len(prop_table), prop_batch_size)]


##########################################################
def append_one_match(bank, sim, mval):
    with open(options.match_file_name, "a") as myfile:
        myfile.write("%s\t%s\t%.12e\n" % (get_tag(bank), get_tag(sim), mval))
        myfile.flush()
    return


cnt_bank_generations = 0
cnt_test_generations = 0
cnt_match_evaluations = 0
cnt_eliminations = 0
for i, bank_batch in enumerate(bank_batches):
    bank_batch_waveforms = {}
    if options.verbose:
        logging.info("\t Processing bank batch {} of {} (size {})".format(i+1,\
            len(bank_batches), len(bank_batch)))
    for j, prop_batch in enumerate(prop_batches):
        prop_batch_waveforms = {}
        if options.verbose:
            logging.info("\t\t Processing proposal batch {} of {} (size {})".format(\
                j+1, len(prop_batches), len(prop_batch)))
        for k, pb in enumerate(bank_batch):
            for l, pp in enumerate(prop_batch):
                ## Avoid computing match as much as possible!
                if options.eliminate:
                    if is_eliminated(pp):
                        append_one_match(pb, pp, -1)
                        continue
                if options.mchirp_window and \
                    outside_mchirp_window(pp, pb, options.mchirp_window) and \
                    outside_ecc_window(pp, pb, options.ecc_window):
                    append_one_match(pb, pp, -1)
                    continue
                if get_tag(pp) == get_tag(pb):
                    append_one_match(pb, pp, 1)
                    continue

                ## Now, we really need to generate both of these waveforms!
                if options.verbose:
                    logging.info("\t\t\t Will need waves for ({}, {})".format(
                        k, l))

                if waveform_exists(pb, bank_batch_waveforms):
                    stilde = bank_batch_waveforms[get_tag(pb)]
                else:
                    cnt_bank_generations += 1
                    if options.verbose:
                        logging.info(
                            "\t\t\t\t Computing waves for ({}, o)".format(k))
                    stilde = get_waveform(pb, options.bank_approximant, f_min,
                                          dt, N)
                    bank_batch_waveforms[get_tag(pb)] = stilde

                if waveform_exists(pp, prop_batch_waveforms):
                    htilde = prop_batch_waveforms[get_tag(pp)]
                else:
                    cnt_test_generations += 1
                    if options.verbose:
                        logging.info(
                            "\t\t\t\t Computing waves for (o, {})".format(l))
                    htilde = get_waveform(pp, options.proposal_approximant,
                                          f_min, dt, N)
                    prop_batch_waveforms[get_tag(pp)] = htilde

                ## Compute match!
                mval, _ = match(stilde,
                                htilde,
                                psd=psd,
                                low_frequency_cutoff=f_min)
                append_one_match(pb, pp, mval)
                cnt_match_evaluations += 1

                ## If match is too high, prevent future match evaluations!
                if options.eliminate:
                    if mval > options.mm:
                        try:
                            os.mknod(options.elimination_dir + '/' +
                                     get_tag(pp))
                            cnt_eliminations += 1
                            if options.verbose:
                                "Eliminated {}".format(get_tag(pp))
                        except:
                            pass

if options.verbose:
    logging.info("Written results to file: {}".format(options.match_file_name))
    logging.info("Total {}+{} waves generated, {} matches evaluated.".format(\
        cnt_bank_generations, cnt_test_generations, cnt_match_evaluations))
    logging.info("Total {} test points eliminated.".format(cnt_eliminations))
    logging.info("Time taken: {} seconds".format(time.time() - _itime))

sys.stdout.flush()
