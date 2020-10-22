#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')
import numpy as np

import sys
import os
import time

from optparse import OptionParser

from glue import gpstime

from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import process as ligolw_process
from glue import git_version

__author__ = "Prayush Kumar <prkumar@cita.utoronto.ca>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])


############## Functions ######################
class test:
    #def __init__(self):
    #self.m1 = m1
    #self.m2 = m2
    QM_MTSUN_SI = 4.9254909500000001e-06
    QM_PI = np.pi

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

    def mchirp_eta_to_m1_m2(self, mchirp, eta):
        mt = mchirp / eta**0.6
        tmp = (mt**2 * (1. - 4. * eta))**0.5
        m1 = (mt + tmp) / 2.
        m2 = (mt - tmp) / 2.
        return (m1, m2)


qm = test()

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
parser.add_option("--eta", help="Mass ratio ", dest="eta", type=float)
parser.add_option("--min-q", help="lower q bound", dest="min_q", type=float)
parser.add_option("--max-q", help="upper q bound", dest="max_q", type=float)
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
parser.add_option("--spin2mag",
                  help="s2-magnitude of the lighter object",
                  dest="spin2mag",
                  type=float)
parser.add_option("--min-sz",
                  help="lower spinZ bound",
                  dest="min_sz",
                  type=float,
                  default=-0.9)
parser.add_option("--max-sz",
                  help="upper spinZ bound",
                  dest="max_sz",
                  type=float,
                  default=0.9)
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
parser.add_option("-C",
                  "--comment",
                  metavar="STRING",
                  help="add the optional STRING as the process:comment",
                  default='')

options, argv_frame_files = parser.parse_args()

print(options.mass1, options.mass2)
print(options.min_q, options.max_q)
print(options.spin1x, options.spin1y, options.spin1z)
print(options.spin2x, options.spin2y, options.spin2z)
print(options.inclination, options.polarization)
print(options.npoints)
if not options.npoints:
    print("You must provide the number of sample points needed")
else:
    npoints = np.int(options.npoints)


def eta_to_q(eta):
    q = (1. + sqrt(1. - 4. * eta)) / (2. * eta) - 1.
    return q


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
mass_min = 3.
mass_max = 100.
mtotalMAX = 200.

smag_min = 1.0
smag_max = 1.0

sxyz_min = options.min_sz
sxyz_max = options.max_sz

inc_min = 0.
inc_max = qm.QM_PI

pol_min = 0.
pol_max = qm.QM_PI * 2.


def sample_bound(MIN, MAX):
    return ((MAX - MIN) * np.random.uniform() + MIN)


def sample_mass():
    return ((mass_max - mass_min) * np.random.uniform() + mass_min)


def sample_smag():
    return ((smag_max - smag_min) * np.random.uniform() + smag_min)


def sample_sxyz():
    return ((sxyz_max - sxyz_min) * np.random.uniform() + sxyz_min)


def sample_inc():
    return ((inc_max - inc_min) * np.random.uniform() + inc_min)


def sample_pol():
    return ((pol_max - pol_min) * np.random.uniform() + pol_min)


# Determine which input options are missing, and which need to be randomly
# sampled.
sim_inspiral_table = lsctables.New(lsctables.SimInspiralTable,
                                   columns=[
                                       'mass1', 'mass2', 'mchirp', 'eta',
                                       'spin1x', 'spin1y', 'spin1z', 'spin2x',
                                       'spin2y', 'spin2z', 'inclination',
                                       'polarization', 'latitude', 'longitude',
                                       'bandpass', 'alpha', 'alpha1', 'alpha2',
                                       'process_id'
                                   ])
outdoc.childNodes[0].appendChild(sim_inspiral_table)


def accept_point_nonspinning(m1, m2):
    mc, et = qm.m1_m2_to_mchirp_eta(m1, m2)
    #feta = -63.5 * et**2 + 65.9 * et + 19.7 - 0.5
    feta = 588.3 * et**3 - 402.2 * et**2 + 113.6 * et + 15.56 - 0.5
    etlim = 3. / 16.
    mtotal = m1 + m2
    if mc > feta and mtotal < mtotalMAX and et > etlim:
        return True
    else:
        return False


def accept_point_alignedspin(mtotal, eta, chi):
    fetachi = eta**-0.6 * \
              12.9241*(1. + 15.1506*eta - 96.7042*eta**2 + 305.441*eta**3 -\
              342.732*eta**4)*(1. + 0.304976*chi + 0.140933*chi**2 +\
              0.199598*chi**3 + 0.151902*chi**4)
    if mtotal >= fetachi:
        return True
    else:
        return False


for i in np.arange(npoints):
    smplpt = lsctables.SimInspiral()
    smplpt.process_id = proc_id
    # Using the field alpha as the index associated with the point. This value
    # remains the same, even if these points get split up between jobs
    smplpt.bandpass = i
    smplpt.alpha = i
    smplpt.alpha1 = -1
    smplpt.alpha2 = -1
    # Get the masses
    if options.mass1 is not None:
        smplpt.mass1 = options.mass1
    else:
        smplpt.mass1 = sample_mass()
    if options.mass2 is not None:
        smplpt.mass2 = options.mass2
    else:
        smplpt.mass2 = sample_mass()
    smplpt.mchirp, smplpt.eta = qm.m1_m2_to_mchirp_eta(smplpt.mass1,
                                                       smplpt.mass2)
    # Sample the spinZs
    if options.spin1x is not None:
        smplpt.spin1x = options.spin1x
    else:
        smplpt.spin1x = sample_sxyz()
    if options.spin1y is not None:
        smplpt.spin1y = options.spin1y
    else:
        smplpt.spin1y = sample_sxyz()
    if options.spin2x is not None:
        smplpt.spin2x = options.spin2x
    else:
        smplpt.spin2x = sample_sxyz()
    if options.spin2y is not None:
        smplpt.spin2y = options.spin2y
    else:
        smplpt.spin2y = sample_sxyz()
    chi = sample_sxyz()
    smplpt.spin1z = chi  #sample_sxyz()
    smplpt.spin2z = chi  #smplpt.spin1z

    if options.eta is not None:
        smplpt.eta = sample_bound(0.2, 0.25)
        mtot_max = 200.0
        mtot_min = 1.5581 * chi**2 + 11.438 * chi + 63.875
        mtot = sample_bound(mtot_min, mtot_max)
        smplpt.mchirp = mtot * smplpt.eta**0.6
        smplpt.mass1, smplpt.mass2 = qm.mchirp_eta_to_m1_m2(
            smplpt.mchirp, smplpt.eta)
    elif options.min_q is not None and options.max_q is not None:
        qsmpl = sample_bound(options.min_q, options.max_q)
        smplpt.eta = qsmpl / (1. + qsmpl)**2
        mtot_max = 200.0
        mtot_min = smplpt.eta**-0.6 * \
                    12.9241*(1. + 15.1506*smplpt.eta - 96.7042*smplpt.eta**2 +\
                    305.441*smplpt.eta**3 - 342.732*smplpt.eta**4) * \
                    (1. + 0.304976*chi + 0.140933*chi**2 +\
                    0.199598*chi**3 + 0.151902*chi**4)
        #mtot_min = 1.5581 * chi**2 + 11.438 * chi + 63.875
        mtot = sample_bound(mtot_min, mtot_max)
        smplpt.mchirp = mtot * smplpt.eta**0.6
        smplpt.mass1, smplpt.mass2 = qm.mchirp_eta_to_m1_m2(
            smplpt.mchirp, smplpt.eta)

    while not accept_point_alignedspin(smplpt.mchirp / smplpt.eta**0.6,
                                       smplpt.eta, chi):
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
        smplpt.mchirp, smplpt.eta = qm.m1_m2_to_mchirp_eta(
            smplpt.mass1, smplpt.mass2)
        # Sample the spin of the heavier object
        if options.spin1z is not None:
            smplpt.spin1z = options.spin1z
        else:
            smplpt.spin1z = sample_sxyz()
        # Sample the spin1 magnitude
        if (smplpt.spin1x or smplpt.spin1y or smplpt.spin1z
            ) and not (options.spin1x or options.spin1y or options.spin1z):
            s1mag = np.sqrt(smplpt.spin1x**2.0 + smplpt.spin1y**2.0 +
                            smplpt.spin1z**2.0)
            if options.spin1mag is not None:
                news1mag = options.spin1mag
            else:
                news1mag = sample_smag()
            smplpt.spin1x *= (news1mag / s1mag)
            smplpt.spin1y *= (news1mag / s1mag)
            smplpt.spin1z *= (news1mag / s1mag)

        # Sample the spin of the lighter object
        if options.spin2z is not None:
            smplpt.spin2z = options.spin2z
        else:
            smplpt.spin2z = sample_sxyz()
        # Sample the spin2 magnitude
        if (smplpt.spin2x or smplpt.spin2y or smplpt.spin2z
            ) and not (options.spin2x is not None or options.spin2y is not None
                       or options.spin2z is not None):
            s2mag = np.sqrt(smplpt.spin2x**2.0 + smplpt.spin2y**2.0 +
                            smplpt.spin2z**2.0)
            if options.spin2mag:
                news2mag = options.spin2mag
            else:
                news2mag = sample_smag()
            smplpt.spin2x *= (news2mag / s2mag)
            smplpt.spin2y *= (news2mag / s2mag)
            smplpt.spin2z *= (news2mag / s2mag)
        # Sample the z-component again
        chi = sample_sxyz()
        smplpt.spin1z = chi  #sample_sxyz()
        smplpt.spin2z = chi  #smplpt.spin1z
        if options.eta is not None:
            smplpt.eta = sample_bound(0.2, 0.25)
            mtot_max = 200.0
            mtot_min = 1.5581 * chi**2 + 11.438 * chi + 63.875
            mtot = sample_bound(mtot_min, mtot_max)
            smplpt.mchirp = mtot * smplpt.eta**0.6
            smplpt.mass1, smplpt.mass2 = qm.mchirp_eta_to_m1_m2(
                smplpt.mchirp, smplpt.eta)
        elif options.min_q is not None and options.max_q is not None:
            qsmpl = sample_bound(options.min_q, options.max_q)
            smplpt.eta = qsmpl / (1. + qsmpl)**2
            mtot_max = 200.0
            mtot_min = smplpt.eta**-0.6 * \
                        12.9241*(1. + 15.1506*smplpt.eta - 96.7042*smplpt.eta**2 +\
                        305.441*smplpt.eta**3 - 342.732*smplpt.eta**4) * \
                        (1. + 0.304976*chi + 0.140933*chi**2 +\
                        0.199598*chi**3 + 0.151902*chi**4)
            #mtot_min = 1.5581 * chi**2 + 11.438 * chi + 63.875
            mtot = sample_bound(mtot_min, mtot_max)
            smplpt.mchirp = mtot * smplpt.eta**0.6
            smplpt.mass1, smplpt.mass2 = qm.mchirp_eta_to_m1_m2(
                smplpt.mchirp, smplpt.eta)

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

    sim_inspiral_table.append(smplpt)

# Store the samples in the output file
proctable = table.get_table(outdoc, lsctables.ProcessTable.tableName)
proctable[0].end_time = gpstime.GpsSecondsFromPyUTC(time.time())
outname = 'TestPoints.xml'
ligolw_utils.write_filename(outdoc, outname)
