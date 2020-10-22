#!/usr/bin/env python

import numpy as np

import sys
import os
import time

from optparse import OptionParser

from numpy.random import random

import lal
from glue import gpstime

from pycbc.pnutils import eta_mass1_to_mass2

from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import process as ligolw_process
from glue import git_version

__author__ = "Prayush Kumar <prkumar@cita.utoronto.ca>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])

# parsing the input arguments
parser = OptionParser(
    version=git_version.verbose_msg,
    usage="%prog [OPTIONS]",
    description=
    "Creates a set of random points in the spinning space, for the set of parameters that are not passed as input, and writes it to XML"
)

parser.add_option("--mass1",
                  help="Mass of the heavier object [Msun]",
                  dest="mass1",
                  type=float)
parser.add_option("--mass2",
                  help="Mass of the lighter object [Msun]",
                  dest="mass2",
                  type=float)
parser.add_option("--mchirp-min",
                  help="Lower limit for chirp mass [Msun]",
                  type=float)
parser.add_option("--mchirp-max",
                  help="Upper limit for chirp mass [Msun]",
                  type=float)
parser.add_option("--mtotal-min",
                  help="Lower limit for total mass [Msun]",
                  type=float)
parser.add_option("--mtotal-max",
                  help="Upper limit for total mass [Msun]",
                  type=float)
parser.add_option("--eta-min", help="Lower limit for eta", type=float)
parser.add_option("--eta-max", help="Upper limit for eta", type=float)
parser.add_option("--ecc-min",
                  help="Lower limit for eccentricity",
                  type=float,
                  default=0)
parser.add_option("--ecc-max",
                  help="Upper limit for eccentricity",
                  type=float,
                  default=0)
parser.add_option("--anomaly-min",
                  help="Lower limit for mean anomaly",
                  type=float,
                  default=0)
parser.add_option("--anomaly-max",
                  help="Upper limit for mean anomaly",
                  type=float,
                  default=0)
parser.add_option("--spin1x",
                  help="s1-x of the heavier object",
                  dest="spin1x",
                  type=float)
parser.add_option("--spin1y",
                  help="s1-y of the heavier object",
                  dest="spin1y",
                  type=float)
parser.add_option("--spin1z",
                  help="s1-z of the heavier object",
                  dest="spin1z",
                  type=float)
parser.add_option("--spin1z-min",
                  help="Max s1-z of the heavier object",
                  type=float,
                  default=-0.99)
parser.add_option("--spin1z-max",
                  help="Min s1-z of the heavier object",
                  type=float,
                  default=0.99)
parser.add_option("--spin1mag",
                  help="s1-magnitude of the heavier object",
                  dest="spin1mag",
                  type=float)
parser.add_option("--spin2x",
                  help="s2-x of the lighter object",
                  dest="spin2x",
                  type=float)
parser.add_option("--spin2y",
                  help="s2-y of the lighter object",
                  dest="spin2y",
                  type=float)
parser.add_option("--spin2z",
                  help="s2-z of the lighter object",
                  dest="spin2z",
                  type=float)
parser.add_option("--spin2z-min",
                  help="Min s2-z of the lighter object",
                  type=float,
                  default=-1.)
parser.add_option("--spin2z-max",
                  help="Max s2-z of the lighter object",
                  type=float,
                  default=1.)
parser.add_option("--spin2mag",
                  help="s2-magnitude of the lighter object",
                  dest="spin2mag",
                  type=float)
parser.add_option("--lambda1-min",
                  help="Lower limit for Lambda of M1",
                  type=float,
                  default=0)
parser.add_option("--lambda1-max",
                  help="Lower limit for Lambda of M1",
                  type=float,
                  default=0)
parser.add_option("--lambda2-min",
                  help="Lower limit for Lambda of M1",
                  type=float,
                  default=0)
parser.add_option("--lambda2-max",
                  help="Lower limit for Lambda of M1",
                  type=float,
                  default=4500)
parser.add_option("--sample-lambda1", action="store_true", default=False)
parser.add_option("--sample-lambda2", action="store_true", default=False)
parser.add_option(
    "--inclination",
    help=
    "inclination angle, between the orbital angular momentum and line of sight from the detector",
    dest="inclination",
    type=float)
parser.add_option(
    "--polarization",
    help=
    "polarization angle, between the radiatoin frame and the detector frame",
    dest="polarization",
    type=float)
parser.add_option("--num-points",
                  help="Number of points needed in the test-points file",
                  dest="npoints",
                  type=int)
parser.add_option("--sample-m1-m2",
                  action="store_true",
                  help="pass if you want to sample the m1-m2 axes",
                  default=True)
parser.add_option("--sample-mchirp-eta",
                  action="store_true",
                  help="pass if you want to sample the mchirp-eta axes",
                  default=False)
parser.add_option("--sample-mtotal-q",
                  action="store_true",
                  help="pass if you want to sample the mtotal-q axes",
                  default=False)
parser.add_option("-V",
                  "--verbose",
                  action="store_true",
                  help="Verbose output",
                  default=False)
parser.add_option("-C",
                  "--comment",
                  metavar="STRING",
                  help="add the optional STRING as the process:comment",
                  default='')

options, argv_frame_files = parser.parse_args()

print(options.mass1, options.mass2)
print(options.spin1x, options.spin1y, options.spin1z)
print(options.spin2x, options.spin2y, options.spin2z)
print(options.spin1z_min, options.spin1z_max)
print(options.spin2z_min, options.spin2z_max)
print(options.inclination, options.polarization)
print(options.mchirp_min, options.mchirp_max)
print(options.eta_min, options.eta_max)
print(options.npoints)
if not options.npoints:
    print("You must provide the number of sample points needed")
else:
    npoints = np.int(options.npoints)

# Create a blank xml document and add the process id
outdoc = ligolw.Document()
outdoc.appendChild(ligolw.LIGO_LW())

proc_id = ligolw_process.register_to_xmldoc(
    outdoc,
    PROGRAM_NAME,
    options.__dict__,
    comment=options.comment,
    version=git_version.id,
    cvs_repository=git_version.branch,
    cvs_entry_time=git_version.date).process_id

# Functions to sample the different parameters
mass1_min = 1.
mass1_max = 100.
mass2_min = 1.
mass2_max = 2.
#
mass_min = min(mass1_min, mass2_min)
mass_max = max(mass1_max, mass2_max)
mtotal_max = mass1_max + mass2_max
#
if options.mchirp_max:
    mchirp_max = options.mchirp_max
else:
    mchirp_max = mtotal_max * 0.25**0.6
if options.mchirp_min:
    mchirp_min = options.mchirp_min
else:
    mchirp_min = (2. * mass_min) * ((mass_max / mass_min) /
                                    (1. + (mass_max / mass_min))**2)**0.6
#
if options.mtotal_min:
    mtotal_min = options.mtotal_min
else:
    mtotal_min = mass1_min + mass2_min
if options.mtotal_max:
    mtotal_max = options.mtotal_max
else:
    mtotal_max = mass1_max + mass2_max

if options.eta_max:
    eta_max = options.eta_max
else:
    eta_max = 0.25
if options.eta_min:
    eta_min = options.eta_min
else:
    eta_min = (mass_max / mass_min) / (1. + (mass_max / mass_min))**2
#
q_min = 10. / eta_mass1_to_mass2(eta_max, 10.)
q_max = 10. / eta_mass1_to_mass2(eta_min, 10.)

print(mtotal_min, mtotal_max)
print(q_min, q_max)
print(mass_min, mass_max)

ecc_min = options.ecc_min
ecc_max = options.ecc_max
anom_min = options.anomaly_min
anom_max = options.anomaly_max

smag_min = 0.
smag_max = 1.0

sxyz_min = -1.0
sxyz_max = 0.

inc_min = 0.
inc_max = lal.PI

pol_min = 0.
pol_max = lal.PI * 2.


def m1_m2_to_mchirp_eta(m1, m2):
    m = m1 + m2
    eta = m1 * m2 / m**2
    mchirp = m * eta**0.6
    return mchirp, eta


def mchirp_eta_to_m1_m2(mchirp, eta):
    mtotal = mchirp / eta**0.6
    m1 = 0.5 * mtotal * (1. + (1. - 4. * eta)**0.5)
    m2 = 0.5 * mtotal * (1. - (1. - 4. * eta)**0.5)
    return m1, m2


def sample_mass1():
    return ((mass1_max - mass1_min) * random() + mass1_min)


def sample_mass2():
    return ((mass2_max - mass2_min) * random() + mass2_min)


def sample_mass():
    return ((mass_max - mass_min) * random() + mass_min)


def sample_smag():
    return ((smag_max - smag_min) * random() + smag_min)


def sample_sxyz():
    return ((sxyz_max - sxyz_min) * random() + sxyz_min)


def sample_inc():
    return ((inc_max - inc_min) * random() + inc_min)


def sample_pol():
    return ((pol_max - pol_min) * random() + pol_min)


def sample_range(MIN, MAX):
    return ((MAX - MIN) * random() + MIN)


# Determine which input options are missing, and which need to be randomly
# sampled.
sim_inspiral_table = lsctables.New(lsctables.SimInspiralTable,
                                   columns=[
                                       'mass1', 'mass2', 'mchirp', 'eta',
                                       'spin1x', 'spin1y', 'spin1z', 'spin2x',
                                       'spin2y', 'spin2z', 'inclination',
                                       'polarization', 'latitude', 'longitude',
                                       'bandpass', 'alpha', 'alpha1', 'alpha2',
                                       'alpha3', 'alpha4', 'process_id'
                                   ])
outdoc.childNodes[0].appendChild(sim_inspiral_table)


def accept_point_40cycregion(m1, m2):
    mc, et = m1_m2_to_mchirp_eta(m1, m2)
    feta = -63.5 * et**2 + 65.9 * et + 19.7 - 0.5
    if mc > feta:
        return True
    else:
        return False


def accept_point(m1, m2):
    mc, et = m1_m2_to_mchirp_eta(m1, m2)
    etlim = 10. / 121.  # q=10
    mtotlim = 200.
    mclim = 10.
    if (m1 + m2) <= mtotlim:
        return True
    else:
        return False


def accept_point_mchirp_eta(mchirp, eta):
    m1, m2 = mchirp_eta_to_m1_m2(mchirp, eta)
    if ((m1 + m2) <= mtotal_max) and (m1 > mass_min) and (m2 > mass_min) and (
            m1 < mass_max) and (m2 < mass_max):
        return True
    else:
        return False


def accept_point_mtotal_q(mtotal, q):
    eta = q / (1. + q)**2
    mchirp = mtotal * eta**0.6
    m1, m2 = mchirp_eta_to_m1_m2(mchirp, eta)
    if (mtotal <= mtotal_max) and (mtotal >= mtotal_min) and (
            eta >= eta_min) and (eta <= eta_max) and (m1 > mass1_min) and (
                m2 > mass2_min) and (m1 < mass1_max) and (m2 < mass2_max):
        return True
    else:
        print("sample mtotal = %f, q = %f REJECTED!" % (mtotal, q))
        return False


for i in np.arange(npoints):
    if i % 1000 == 0 and options.verbose:
        print("Point %d" % i, file=sys.stderr)
    smplpt = lsctables.SimInspiral()

    smplpt.process_id = proc_id

    # Using the field alpha as the index associated with the point. This value
    # remains the same, even if these points get split up between jobs
    smplpt.bandpass = i
    smplpt.alpha = i

    smplpt.alpha1 = sample_range(ecc_min, ecc_max)
    smplpt.alpha2 = sample_range(anom_min, anom_max)

    # Get the masses
    if options.sample_m1_m2:
        if options.mass1 is not None:
            smplpt.mass1 = options.mass1
        else:
            smplpt.mass1 = sample_mass1()
        if options.mass2 is not None:
            smplpt.mass2 = options.mass2
        else:
            smplpt.mass2 = sample_mass2()
        while not accept_point(smplpt.mass1, smplpt.mass2):
            if options.mass1 is not None and options.mass2 is not None:
                break
            if options.mass1 is not None:
                smplpt.mass1 = options.mass1
            else:
                smplpt.mass1 = sample_mass()
            if options.mass2 is not None:
                smplpt.mass2 = options.mass2
            else:
                smplpt.mass2 = sample_mass()

        temp1 = smplpt.mass1
        temp2 = smplpt.mass2
        smplpt.mass1 = max(temp1, temp2)
        smplpt.mass2 = min(temp1, temp2)

        smplpt.mchirp, smplpt.eta = m1_m2_to_mchirp_eta(
            smplpt.mass1, smplpt.mass2)

    # Sample the chirp-mass and eta coordinates
    if options.sample_mchirp_eta:
        tempmc = 0.1
        tempet = 0.1
        while not accept_point_mchirp_eta(tempmc, tempet):
            tempmc = sample_range(mchirp_min, mchirp_max)
            tempet = sample_range(eta_min, eta_max)

        smplpt.mchirp = tempmc
        smplpt.eta = tempet
        smplpt.mass1, smplpt.mass2 = mchirp_eta_to_m1_m2(
            smplpt.mchirp, smplpt.eta)

    if options.sample_mtotal_q:
        tempmt = sample_range(mtotal_min, mtotal_max)
        tempq = sample_range(q_min, q_max)
        while not accept_point_mtotal_q(tempmt, tempq):
            tempmt = sample_range(mtotal_min, mtotal_max)
            tempq = sample_range(q_min, q_max)

        smplpt.eta = tempq / (1. + tempq)**2
        smplpt.mchirp = tempmt * smplpt.eta**0.6
        smplpt.mass1, smplpt.mass2 = mchirp_eta_to_m1_m2(
            smplpt.mchirp, smplpt.eta)

    # Sample the spin of the heavier object
    if options.spin1x is not None:
        smplpt.spin1x = options.spin1x
    else:
        smplpt.spin1x = sample_sxyz()
    if options.spin1y is not None:
        smplpt.spin1y = options.spin1y
    else:
        smplpt.spin1y = sample_sxyz()
    if options.spin1z is not None:
        smplpt.spin1z = options.spin1z
    else:
        smplpt.spin1z = sample_range(options.spin1z_min, options.spin1z_max)
    # Sample the spin1 magnitude
    if (smplpt.spin1x or smplpt.spin1y or smplpt.spin1z
        ) and not (options.spin1x and options.spin1y and options.spin1z):
        s1mag = np.sqrt(smplpt.spin1x**2.0 + smplpt.spin1y**2.0 +
                        smplpt.spin1z**2.0)
        if options.spin1mag is not None:
            news1mag = options.spin1mag
        else:
            news1mag = sample_smag()
        #smplpt.spin1x *= (news1mag/s1mag)
        #smplpt.spin1y *= (news1mag/s1mag)
        #smplpt.spin1z *= (news1mag/s1mag)

    # Sample the spin of the lighter object
    if options.spin2x is not None:
        smplpt.spin2x = options.spin2x
    else:
        smplpt.spin2x = sample_sxyz()
    if options.spin2y is not None:
        smplpt.spin2y = options.spin2y
    else:
        smplpt.spin2y = sample_sxyz()
    if options.spin2z is not None:
        smplpt.spin2z = options.spin2z
    else:
        smplpt.spin2z = sample_range(options.spin2z_min, options.spin2z_max)
    # Sample the spin2 magnitude
    if (smplpt.spin2x or smplpt.spin2y or smplpt.spin2z
        ) and not (options.spin2x and options.spin2y and options.spin2z):
        s2mag = np.sqrt(smplpt.spin2x**2.0 + smplpt.spin2y**2.0 +
                        smplpt.spin2z**2.0)
        if options.spin2mag:
            news2mag = options.spin2mag
        else:
            news2mag = sample_smag()
        #smplpt.spin2x *= (news2mag/s2mag)
        #smplpt.spin2y *= (news2mag/s2mag)
        #smplpt.spin2z *= (news2mag/s2mag)

    # Get the tidal deformability of each object
    if options.sample_lambda1:
        smplpt.alpha3 = sample_range(options.lambda1_min, options.lambda1_max)
    else:
        smplpt.alpha3 = -1
    if options.sample_lambda2:
        smplpt.alpha4 = sample_range(options.lambda2_min, options.lambda2_max)
    else:
        smplpt.alpha4 = -1

    # Get the inclination angle
    if options.inclination is not None:
        smplpt.inclination = options.inclination
    else:
        smplpt.inclination = sample_inc()

    # Get the polarization angle
    if options.polarization is not None:
        smplpt.polarization = options.polarization
    else:
        smplpt.polarization = sample_pol()

    smplpt.latitude = 0.
    smplpt.longitude = 0.
    #
    sim_inspiral_table.append(smplpt)
    # print smplpt.spin1z, smplpt.spin2z

# Store the samples in the output file
proctable = table.get_table(outdoc, lsctables.ProcessTable.tableName)
proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(time.time())
outname = 'TestPoints.xml'
ligolw_utils.write_filename(outdoc, outname)
