#! /usr/bin/env python
from pycbc.waveform import get_fd_waveform
from pycbc.types import FrequencySeries, zeros
from pycbc.filter import match
import pycbc.psd

import matplotlib
matplotlib.use('Agg')
import numpy as np
from numpy import complex128, cos, float64, floor, loadtxt, sin

import sys
import os
import time

from optparse import OptionParser

from pycbc.pnutils import *
import lal
from glue import gpstime

from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils

__author__ = "Prayush Kumar <prkumar@cita.utoronto.ca>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])

#########################################################################
#################### Input parsing #####################
#########################################################################
#{{{
parser = OptionParser(
    usage="%prog [OPTIONS]",
    description=
    "Takes in the iteration id. It chooses num-new-points new points. Each of these points are chosen in a way that they have an overlap < MM with all other points. This is done to ensure that these new points are not overlapping in terms of the area that they cover."
)

parser = OptionParser()
parser.add_option("--iteration-id",
                  help="The index of the iteration",
                  type=int,
                  dest="iid")
parser.add_option("--num-new-points",
                  help="No of bank points in each sub-job",
                  default=1000,
                  type=int)

parser.add_option('-m',
                  '--minimal-match',
                  metavar='MM',
                  help='minimal match',
                  default=0.95,
                  type=float,
                  dest="minimal_match")
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

parser.add_option(
    '-w',
    '--mchirp-window',
    metavar='MC_WIN',
    help=
    'Fractional window on mchirp parameter. If waveform parameters differ by more than this window, the overlap is set to 0.',
    default=0.01,
    type=float,
    dest="mchirp_window")

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
print("MM = %f, mchirp-window = %f" %
      (options.minimal_match, options.mchirp_window))
#ctx = CUDAScheme()

#########################################################################
################### Get new sample points ###############################
#########################################################################

######## Creating the new points file ############
#{{{
if options.iid is not None:
    iid = options.iid
    new_file_name = "testpoints/test_%d.xml" % iid
else:
    idx = 0
    name1 = "testpoints/test_%d.xml" % idx
    idx += 1
    name2 = "testpoints/test_%d.xml" % idx
    while os.path.exists(name2):
        print(name1, name2)
        idx += 1
        name1 = name2
        name2 = "testpoints/test_%d.xml" % idx
    new_file_name = name1
    iid = idx - 1

print("Storing the new sample points in %s" % new_file_name)
sys.stdout.flush()

new_points_doc = ligolw.Document()
new_points_doc.appendChild(ligolw.LIGO_LW())

out_proc_id = ligolw_process.register_to_xmldoc(
    new_points_doc, PROGRAM_NAME, options.__dict__,
    comment=options.comment).process_id

new_points_table = lsctables.New(lsctables.SimInspiralTable,
                                 columns=[
                                     'mass1', 'mass2', 'mchirp', 'eta',
                                     'spin1x', 'spin1y', 'spin1z', 'spin2x',
                                     'spin2y', 'spin2z', 'inclination',
                                     'polarization', 'latitude', 'longitude',
                                     'bandpass', 'alpha', 'alpha1', 'alpha2',
                                     'process_id'
                                 ])
new_points_doc.childNodes[0].appendChild(new_points_table)

#}}}
######## Functions to get new sample points #########
#{{{
# Functions to sample the different parameters
#{{{

## Open the file containing allowed values of eta
print("Opening the file containing the allowed set of eta values")
consfile = open('FinalEtaChi.dat', 'r')
consvalues = loadtxt(consfile)
print("Allowed set is", consvalues)
sys.stdout.flush()

etavalues = []
chivalues = []
for et, ch in consvalues:
    etavalues.append(et)
    chivalues.append(ch)

mass_min = 9.
mass_max = 200.

mtotal_max = 200.

#mchirp_max = (2. * mass_max) * (0.25**0.6)
mchirp_min = 24.64458  # (2. * mass_min) * (0.25**0.6)

eta_max = 0.25
eta_min = mass_max * mass_min / (mass_max + mass_min)**2.

smag_min = 0.0
smag_max = 0.0

sxyz_min = 0.0
sxyz_max = 0.0

inc_min = 0.
inc_max = 0.  #qm.QM_PI

pol_min = 0.
pol_max = 0.  #qm.QM_PI * 2.


def sample_mass():
    return ((mass_max - mass_min) * np.random.uniform() + mass_min)


def sample_mchirp():
    return ((mchirp_max - mchirp_min) * np.random.uniform() + mchirp_min)


def sample_bound(MIN, MAX):
    return ((MAX - MIN) * np.random.uniform() + MIN)


def sample_eta_uniform():
    return ((eta_max - eta_min) * np.random.uniform() + eta_min)


def sample_eta_chi():
    smplidx = int(floor(len(chivalues) * np.random.uniform()))
    return (etavalues[smplidx], chivalues[smplidx])


def sample_eta():
    return etavalues[int(floor(len(etavalues) * np.random.uniform()))]


def sample_smag():
    return ((smag_max - smag_min) * np.random.uniform() + smag_min)


def sample_sxyz():
    return ((sxyz_max - sxyz_min) * np.random.uniform() + sxyz_min)


def sample_inc():
    return ((inc_max - inc_min) * np.random.uniform() + inc_min)


def sample_pol():
    return ((pol_max - pol_min) * np.random.uniform() + pol_min)


def accept_point_boundary(mc, eta):
    # The following function describes the equation of the boundary of the region
    # which bounds the BBH systems that have 100% of their power in <= 40 waveform
    # cyclces. (For non-spinning systems). Also taking only points with mchirp
    # below 52.233 to not sample the region which is already covered by the
    # bank_0.xml
    feta = -63.5 * eta**2 + 65.9 * eta + 19.7
    if mc >= feta and mc < 52.233:
        return True
    else:
        return False


def get_new_sample_point():
    """This function returns an instance of lsctables.SimInspiral, with elements corresponding to various physical parameters uniformly sampled within their respective ranges. """
    p = lsctables.SimInspiral()
    p.alpha = -1
    p.alpha1 = -1
    p.alpha2 = -1

    p.eta, chi = sample_eta_chi()
    p.spin1z = chi
    p.spin2z = chi

    # Get the allowed range of mchirp for this eta
    #q = (0.5/p.eta) * (1. + sqrt(1. - 4.*p.eta)) - 1.
    mtot_max = mtotal_max  #mass_max * (1. + (1./q) )
    mtot_min = 1.5581 * chi**2 + 11.438 * chi + 63.875
    mtot = sample_bound(mtot_min, mtot_max)
    p.mchirp = mtot * p.eta**0.6
    p.mass1, p.mass2 = mchirp_eta_to_mass1_mass2(p.mchirp, p.eta)

    #mchirp_max = (mtot_max) * p.eta**0.6
    #mchirp_min = -63.5 * p.eta**2 + 65.9 * p.eta + 19.7 - 0.50

    #p.mchirp = sample_bound(mchirp_min,mchirp_max)
    #tm1,tm2 = qm.mchirp_eta_to_m1_m2(p.mchirp,p.eta)

    # Just in case the values fall outside the bank's range
    #while (tm1 < mass_min) or (tm1 > mass_max) or (tm2 < mass_min) or (tm2 > mass_max):
    #  p.eta = sample_eta()
    #  p.mchirp = sample_bound(mchirp_min,mchirp_max)
    #  tm1,tm2 = qm.mchirp_eta_to_m1_m2(p.mchirp,p.eta)

    #p.eta = eta
    #p.mass1 = m1
    #p.mass2 = m2
    #tm1 = sample_mass()
    #tm2 = sample_mass()
    #p.mass1 = max( tm1, tm2 )
    #p.mass2 = min( tm1, tm2 )

    #p.mchirp, p.eta = qm.m1_m2_to_mchirp_eta( p.mass1, p.mass2 )
    #while not accept_point( p.mchirp, p.eta ):
    #  tm1 = sample_mass()
    #  tm2 = sample_mass()
    #  p.mass1 = max( tm1, tm2 )
    #  p.mass2 = min( tm1, tm2 )
    #  p.mchirp, p.eta = qm.m1_m2_to_mchirp_eta( p.mass1, p.mass2 )

    p.spin1x = 0.  #sample_sxyz()
    p.spin1y = 0.  #sample_sxyz()
    #p.spin1z = sample_sxyz()

    #smag = np.sqrt( p.spin1x**2. + p.spin1y**2. + p.spin1z**2. )
    #if smag:
    #  newsmag = sample_smag()
    #  p.spin1x *= (newsmag/smag)
    #  p.spin1y *= (newsmag/smag)
    #  p.spin1z *= (newsmag/smag)

    p.spin2x = 0.  #sample_sxyz()
    p.spin2y = 0.  #sample_sxyz()
    #p.spin2z = sample_sxyz()

    #smag = np.sqrt( p.spin2x**2. + p.spin2y**2. + p.spin2z**2. )
    #if smag:
    #  newsmag = sample_smag()
    #  p.spin2x *= (newsmag/smag)
    #  p.spin2y *= (newsmag/smag)
    #  p.spin2z *= (newsmag/smag)

    p.inclination = sample_inc()
    p.polarization = sample_pol()
    p.latitude = 0.
    p.longitude = 0.

    return p


#}}}


# Miscellaneous
def within_mchirp_window(bank, sim, w):
    #{{{
    bmchirp = None
    smchirp = None

    if hasattr(bank, "mchirp"):
        bmchirp = bank.mchirp
    elif hasattr(bank, "mass1") and hasattr(bank, "mass2"):
        bmchirp, eta = mass1_mass2_to_mchirp_eta(bank.mass1, bank.mass2)
    elif hasattr(bank, "mtotal") and hasattr(bank, "eta"):
        bmchirp = bank.mtotal * (bank.eta**0.6)

    if hasattr(sim, "mchirp"):
        smchirp = sim.mchirp
    elif hasattr(sim, "mass1") and hasattr(sim, "mass2"):
        smchirp, eta = mass1_mass2_to_mchirp_eta(sim.mass1, sim.mass2)
    elif hasattr(sim, "mtotal") and hasattr(sim, "eta"):
        smchirp = sim.mtotal * (sim.eta**0.6)

    if abs(smchirp - bmchirp) < (w * bmchirp):
        return True
    else:
        False
    #}}}


################### Functions to generate waveform ##################
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
    f_max = min(1. / (2. * dt), 0.15 / ((m1 + m2) * lal.MTSUN_SI))

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


def reject_new_sample_point(new_point, points_table, MM, psd, f_min, dt, N,
                            in_mchirp_window):
    """This function takes in a new proposed point, and computes its overlaps with all points in the points_table. If the max of these overlaps is > MM, it returns True, else returns False. Which implies that if the new proposed point should be rejected from the set, it returns True, and False if that point should be kept."""
    matches = []
    hn = get_waveform(new_point, f_min, dt, N)
    if in_mchirp_window:
        mchirp_window = in_mchirp_window
    else:
        mchirp_window = 1.0

    for point in points_table:
        if within_mchirp_window(new_point, point, mchirp_window):
            #print "\tComputing overlaps with point number %d" % point.bandpass
            #sys.stdout.flush()
            hpt = get_waveform(point, f_min, dt, N)
            m, idx = match(hn, hpt, psd=psd, low_frequency_cutoff=f_min)
            matches.append(m)
        else:
            matches.append(0)

    if max(matches) > MM:
        return True
    else:
        return False


#}}}

################### Obtain the new sample points #######################
#{{{
num_new_points = np.int(options.num_new_points)

MM = options.minimal_match

f_min = options.f_min
signal_length = options.signal_length
sample_rate = options.sample_rate

dt = 1. / np.float64(sample_rate)
N = signal_length * sample_rate

# get the ZDHP psd
psd = pycbc.psd.from_asd_txt("/home/prayush/advLIGO_PSDs/ZERO_DET_high_P.txt",
                             N / 2 + 1, 1. / np.float64(signal_length), f_min)
#psd *= DYN_RANGE_FAC **2
psd = FrequencySeries(psd, delta_f=psd.delta_f, dtype=float64)
#print "PSD generated. First few points are %f, %f, %f..." % (psd[0],psd[1],psd[2])

print("Trying to choose points with overlap < %f" % MM)
print("f_min=%f, sig_len=%d, sample_rate=%d, dt=%f, N=%d" %
      (f_min, signal_length, sample_rate, dt, N))
#sys.stdout.flush()

cnt = 0
while cnt < num_new_points:
    print("%d points chosen" % cnt)
    #sys.stdout.flush()
    if cnt == 0:
        new_point = get_new_sample_point()
        new_point.bandpass = cnt
        new_point.process_id = out_proc_id
        new_points_table.append(new_point)
        cnt += 1
        continue

    k = 0
    new_point = get_new_sample_point()
    while reject_new_sample_point(new_point, new_points_table, MM, psd, f_min,
                                  dt, N, options.mchirp_window):
        print("Rejecting sample %d" % k)
        k += 1
        new_point = get_new_sample_point()

    new_point.bandpass = cnt
    new_point.process_id = out_proc_id
    new_points_table.append(new_point)
    cnt += 1

#}}}
############## Write the new sample points to XML #############
print("Writing %d new points to %s" % (len(new_points_table), new_file_name))
sys.stdout.flush()

new_points_proctable = table.get_table(new_points_doc,
                                       lsctables.ProcessTable.tableName)
new_points_proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(time.time())
ligolw_utils.write_filename(new_points_doc, new_file_name)
