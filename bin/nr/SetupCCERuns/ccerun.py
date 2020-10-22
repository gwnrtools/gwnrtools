#!/bin/env python
# Copyright (C) 2014 Prayush Kumar, Heather Fong
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
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

import os
import sys
import numpy as np
from numpy import *
import time
import subprocess as cmd
import glob
import h5py
import matplotlib as plt  # FIXM
from matplotlib import use
use('Agg')
#import matplotlib.pyplot as plt
#import pylab as plt

#from numpy import array, where, arange, ceil, arctan2, arctan

try:
    from scipy.interpolate import interp1d
except ImportError:
    print("Dont uniformly sample output")

try:
    from DiscreteFunction import *
    from WaveFunction import *
    from DataAnalysis import *
except BaseException:
    print("Error importing Psi4->h integration modules")

try:
    import UseNRinDA
    import lal
    from glue.ligolw import ligolw, lsctables

    @lsctables.use_in
    class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
        pass

    from pycbc.waveform import *
    from pycbc.psd import from_txt
    from pycbc.filter import match
    from pycbc.types import *
except BaseException:
    print("Error importing LAL/PyCBC modules")

#########################################################################
__author__ = "Prayush Kumar <prkumar@cita.utoronto.ca>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])

#########################################################################
#########################################################################


def nextpow2(n):
    return 2**int(ceil(log2(n)))


def getSec(s):
    l = s.split(':')
    return int(l[0]) * 3600 + int(l[1]) * 60 + int(l[2])


def get_uniform_mass_range(m_lower, m_upper, m_sep):
    # {{{
    mlist = [m_lower]
    for m in np.arange(np.ceil(m_lower), np.floor(m_upper), m_sep):
        mlist.append(m)
    mlist.append(m_upper)
    return np.array(mlist)
    # }}}


def overlaps_vs_totalmass(wav1,
                          wav2,
                          psd=None,
                          mf_lower=-1.,
                          m_lower=-1.,
                          m_upper=100.,
                          m_delta=5.):
    # Need two wobjects of nr_waveform class.
    # Waveforms are rescaled to different total masses and their overlaps computed
    # Returns an array of total masses and overlaps
    # {{{
    # print min(wav1.rawhp), max(wav1.rawhp), max(wav2.rawhp), min(wav2.rawhp)
    if psd is None:
        raise IOError("Provide the PSD please!")
    if mf_lower < 0:
        print("Initial orbital frequencies will be deduced after blending")
    #
    t2_opt = [1000, 2000]
    t_option = [100, t2_opt[0], t2_opt[1], 50, 100]
    f_lower = 15. - 0.5
    # Calculate lowest total mass, from a) mf_lower, b) m_lower, c) calculate
    if mf_lower > 0:
        m_lower = mf_lower / f_lower / lal.MTSUN_SI
    elif m_lower <= 0:
        rescaled_mass, orbit_freq1 = wav1.get_orbital_frequency(t=max(t2_opt))
        rescaled_mass, orbit_freq2 = wav2.get_orbital_frequency(t=max(t2_opt))
        m_lower = max(orbit_freq1, orbit_freq2) * rescaled_mass / f_lower
    print(orbit_freq1, orbit_freq2, "lowest total Mass = %f" % m_lower)
    #
    overlaps = []
    mass_range = get_uniform_mass_range(m_lower, m_upper, m_delta)
    for mtot in mass_range:
        #wav1.rescale_to_totalmass( mtot )
        #wav2.rescale_to_totalmass( mtot )
        wav_blended1 = blend(wav1, mtot, wav1.sample_rate, wav1.time_length,
                             t_option)  # blending
        wav_blended2 = blend(wav2, mtot, wav1.sample_rate, wav1.time_length,
                             t_option)  # blending
        if len(wav_blended1) != len(wav_blended2):
            raise RuntimeError(
                "blending function return different sets of waveforms!!")
        tmp_overlaps = [mtot]
        for ii in range(len(wav_blended1)):
            hp1, hp2 = wav_blended1[ii], wav_blended2[ii]
            olap = overlap_between_waveforms(hp1, hp2, psd=psd)
            tmp_overlaps.append(olap)
            print("--In OvsM: window %d, overlap = %f" % (ii, olap))
        overlaps.append(tmp_overlaps)
    return overlaps
    # }}}


def overlap_between_waveforms(wav1, wav2, psd=None, f_lower=15.):
    # Return overlap between two TimeSEries with psd needed as a FrequencySeries
    # {{{
    try:
        if psd is None:
            psd = self.psd
    except BaseException:
        raise IOError("Please compute and store PSD")
    #
    len1, len2, lenp = len(wav1), len(wav2), len(psd)
    if len1 != len2:
        raise IOError("Length of waveforms not equal")
    if wav1.delta_t != wav2.delta_t:
        raise IOError("Mismatched wave sample rate")
    if len1 != 2 * lenp - 2:
        raise IOError("PSD length inconsistent with waveforms")
    #
    return match(wav1, wav2, psd=psd, low_frequency_cutoff=f_lower)[0]
    # }}}


def blend(hin, mm, sample, time, t_opt, WinID=-1):
    # Only dealing with real part, don't do hc calculations
    # t_opt is length-5 array describing multiples of mm
    # Returns length-5 array of TimeSeries (1 per blending)
    # {{{
    hp0, hc0 = hin.rescale_to_totalmass(mm)
    hp0._epoch = hc0._epoch = 0
    amp = TimeSeries(np.sqrt(hp0**2 + hc0**2), copy=True, delta_t=hp0.delta_t)
    max_a, max_a_index = amp.abs_max_loc()
    print("\n\n In blend:\nTotal Mass = %f, len(hp0,hc0) = %d, %d = %f s" %
          (mm, len(hp0), len(hc0), hp0.sample_times[-1] - hp0.sample_times[0]))
    print("Waveform max = %e, located at %d" % (max_a, max_a_index))
    #amp_after_peak = amp
    #amp_after_peak[:max_a_index] = 0
    mtsun = lal.MTSUN_SI
    amp_after_peak = amp[max_a_index:]
    iA, vA = min(enumerate(amp_after_peak),
                 key=lambda x: abs(x[1] - 0.01 * max_a))
    iA += max_a_index
    #iA, vA = min(enumerate(amp_after_peak),key=lambda x:abs(x[1]-0.01*max_a))
    iB, vB = min(enumerate(amp_after_peak),
                 key=lambda x: abs(x[1] - 0.1 * max_a))
    iB += max_a_index
    if iA <= max_a_index:
        print("iA = %d, iB = %d, vA = %e, vB = %e" % (iA, iB, vA, vB),
              file=sys.stdout)
        sys.stdout.flush()
        raise RuntimeError("Couldnt find amplitude threshold time iA")
        # do something
        #fout = open('hpdump.dat','w+')
        # for i in range( len(amp) ):
        #  if i > max_a_index and amp[i] == 0: break
        #  fout.write('%e\t%e\n' % (amp.sample_times[i],amp[i]))
        # fout.close()
        # Find the point the hard way
        target_amp = max_a * 0.01
        tmp_data = amp.data
        for idx in range(max_a_index, len(amp)):
            if tmp_data[idx] < target_amp:
                break
        iA = idx
        print("Newfound iA = %d" % iA)
        # Yet another way
        amp_after_peak = amp[max_a_index:]
        iA, vA = min(enumerate(amp_after_peak),
                     key=lambda x: abs(x[1] - 0.01 * max_a))
        iA += max_a_index
        print("Newfound iA another way = %d" % iA)
        raise RuntimeError("Had to find amplitude threshold the hard way")
    if iB <= max_a_index:
        raise RuntimeError("Couldnt find amplitude threshold time iB")
        # this doesn't happen yet
    print("NEW: iA = %d, iB = %d, vA = %e, vB = %e" % (iA, iB, vA, vB))
    t = [[t_opt[0] * mm, 500 * mm, hp0.sample_times.data[iA] / mtsun, hp0.sample_times.data[iA] / mtsun + t_opt[3] * mm],  # Prayush's E
         [t_opt[0] * mm, t_opt[1] * mm, hp0.sample_times.data[iA] / \
             mtsun, hp0.sample_times.data[iA] / mtsun + t_opt[3] * mm],
         [t_opt[0] * mm, t_opt[1] * mm, hp0.sample_times.data[iB] / \
             mtsun, hp0.sample_times.data[iB] / mtsun + t_opt[4] * mm],
         [t_opt[0] * mm, t_opt[2] * mm, hp0.sample_times.data[iA] / \
             mtsun, hp0.sample_times.data[iA] / mtsun + t_opt[3] * mm],
         [t_opt[0] * mm, t_opt[2] * mm, hp0.sample_times.data[iB] / mtsun, hp0.sample_times.data[iB] / mtsun + t_opt[4] * mm]]
    hphc = []
    hphc.append(hp0)
    for i in range(len(t)):
        if (WinID >= 0 and WinID < len(t)) and i != WinID:
            continue
        print("Testing window with t = ", t[i])
        hphc.append(
            hin.blending_function(hp0=hp0,
                                  t=t[i],
                                  sample_rate=sample,
                                  time_length=time))
    print("No of blending windows being tested = %d" % (len(hphc) - 1))
    return hphc
    # }}}


def blendTimeSeries(hp0, hc0, mm, sample, time, t_opt):
    # Only dealing with real part, don't do hc calculations
    # t_opt is length-5 array describing multiples of mm
    # Returns length-5 array of TimeSeries (1 per blending)
    # {{{
    from UseNRinDA import nr_waveform
    nrtool = nr_waveform()
    amp = TimeSeries(np.sqrt(hp0**2 + hc0**2), copy=True, delta_t=hp0.delta_t)
    max_a, max_a_index = amp.abs_max_loc()
    print("Waveform max = %e, located at %d" % (max_a, max_a_index))
    amp_after_peak = amp
    amp_after_peak[:max_a_index] = 0
    mtsun = lal.MTSUN_SI
    iA, vA = min(enumerate(amp_after_peak),
                 key=lambda x: abs(x[1] - 0.01 * max_a))
    iB, vB = min(enumerate(amp_after_peak),
                 key=lambda x: abs(x[1] - 0.1 * max_a))
    print(iA, iB)
    t = [[t_opt[0] * mm, 500 * mm, hp0.sample_times.data[iA] / mtsun, hp0.sample_times.data[iA] / mtsun + t_opt[3] * mm],  # Prayush's E
         [t_opt[0] * mm, t_opt[1] * mm, hp0.sample_times.data[iA] / \
             mtsun, hp0.sample_times.data[iA] / mtsun + t_opt[3] * mm],
         [t_opt[0] * mm, t_opt[1] * mm, hp0.sample_times.data[iB] / \
             mtsun, hp0.sample_times.data[iB] / mtsun + t_opt[4] * mm],
         [t_opt[0] * mm, t_opt[2] * mm, hp0.sample_times.data[iA] / \
             mtsun, hp0.sample_times.data[iA] / mtsun + t_opt[3] * mm],
         [t_opt[0] * mm, t_opt[2] * mm, hp0.sample_times.data[iB] / mtsun, hp0.sample_times.data[iB] / mtsun + t_opt[4] * mm]]
    hphc = []
    # hphc.append(hp0)
    for i in range(len(t)):
        print(t[i])
        hphc.append(
            nrtool.blending_function_Tukey(hp0=hp0,
                                           t=t[i],
                                           sample_rate=sample,
                                           time_length=time))
    print("No of blending windows being tested = %d" % len(hphc))
    return hphc
    # }}}


###############################################################################
# #############################################################################
###############################################################################
class cce_run():
    # {{{
    def __init__(
            self,
            datafile=None,
            datadir=None,
            outdir=None,
            pittnull='/home/p/pfeiffer/prayush/src/cactus_cce/Cactus/exe/cactus_pittnull',
            ccemodeldir='/home/p/pfeiffer/prayush/scratch/projects/CCE_modeldir/',
            cceparfile='highResCce700.par',
            nullnews_maxtimelevels=None,
            timestep=0.15,
            post_process_only=True,
            verbose=True):
        """
    ###############################################################################
    # #############################################################################
    ##  BASE CLASS: CONTAINS FUNCTIONS THAT ARE TO BE APPLIED TO ALL MODES OF *ONE*
    #               SIMULATION, IN A DIFFERENT AND SPECIFIC MANNER, WHICH WOULD
    #               DEPEND ON THE MODE. IT MANAGES THE HANDLING OF ALL MODES
    #               CONSISTENTLY.
    # E.g.:
    #  - STORING GROUND-LEVEL INFORMATION ABOUT CCE DATA FILE
    #  - SETTING UP CCE RUNS USING CACTUS'S PITTNULL CCE CODE
    #  - SETTING UP RE-RUNS, FROM INTERRUPTED CCE EVOLUTIONS
    #  - COLLECT UP NEWS AND PSI4 DATA FROM THE CODE'S OUTPUT INTO HDF5 FILES
    #  - INTEGRATE PSI4 TWICE TO GIVE STRAIN MODES. FFI INTEGRATION IS USED.
    #  - INTEGRATE NEWS ONCE TO GIVE STRAIN MODES. FFI INTEGRATION IS USED, BUT FOR
    #    M=0 USE TIME-DOMAIN INTEGRATION.
    #
    # Class containing the ground level information for an Cce boundary data file.
    # The inputs are:
    # datadir = EXACT directory with the Cce files
    # datafile= Name of the Cce file
    #
    # #############################################################################
    ###############################################################################
        """
        # {{{
        if not post_process_only:
            if datafile is None:
                raise IOError("Specify CCE data file")
            if datadir is None:
                raise IOError("Specify CCE data file directory")
            if not os.path.exists("%s/%s" % (datadir, datafile)):
                raise IOError("Input Cce data file cannot be located")
            if not os.path.exists(pittnull):
                raise IOError("Cannot located pittnull exe")
        if outdir is None:
            raise IOError("Where to set up the CCE run?")
        #
        self.datafile = datafile
        self.datadir = datadir
        self.outdir = outdir
        if not os.path.exists(outdir):
            if verbose:
                print("Outdirectory %s didn't exist..created." % outdir)
            cmd.getoutput('mkdir -p %s' % outdir)
        self.pittnull = pittnull
        #
        self.timestep = timestep
        self.maxtimelevels = nullnews_maxtimelevels
        self.ccemodeldir = ccemodeldir
        self.cceparfile = cceparfile
        #
        self.get_prefix()
        self.get_nn()
        #
        self.walltime = 48
        self.filetags = ['Psi4_scri.L0?M?0?.asc', 'NewsB_scri.L0?M?0?.asc']
        self.verbose = verbose
        # }}}

    #

    def get_datafile(self):
        # {{{
        return self.datafile
        # }}}

    def get_datadirname(self):
        # {{{
        return self.datadir
        # }}}

    def get_outdirname(self):
        # {{{
        return self.outdir
        # }}}

    def get_nn(self):
        # Assumption: Directories are to be named as prefix-1, prefix-2... Therefore
        # if no such directories exist, its believed to be a new run
        # {{{
        if not os.path.exists(self.outdir):
            self.nn = 1
            return
        pwd = cmd.getoutput('pwd')
        os.chdir(self.outdir)
        #
        xx, dd = glob.glob(self.prefix + '-?'), glob.glob(self.prefix + '-??')
        xx.sort()
        dd.sort()
        xx.extend(dd)
        if len(xx) == 0:
            self.nn = 1
        elif int(xx[-1].split('-')[-1]) != len(xx):
            print(xx, xx[-1])
            raise RuntimeError(
                "Are not run directories sequentially numbered for %s" %
                self.prefix)
        else:
            self.nn = len(xx) + 1
        #
        os.chdir(pwd)
        return self.nn
        # }}}

    def get_all_subdirs(self):
        # {{{
        pwd = cmd.getoutput('pwd')
        os.chdir(self.outdir)
        sd = glob.glob(self.prefix + '-?')
        dd = glob.glob(self.prefix + '-??')
        sd.sort()
        dd.sort()
        sd.extend(dd)
        self.subdirs = sd
        os.chdir(pwd)
        return sd
        # }}}

    def get_prefix(self):
        # {{{
        out = 'Cce'
        self.ExtractionRadius = int(self.datafile.split('.h5')[0][-4:])
        out += 'R%04d' % self.ExtractionRadius
        self.prefix = out
        # }}}

    def get_submitfilename(self, nn=None):
        # {{{
        if nn is None:
            nn = self.nn
        out = self.prefix
        out += '-%d.input' % nn
        return out
        # }}}

    def get_continuationfilename(self):
        # {{{
        return self.prefix + '-continuation.py'
        # }}}

    def get_parfilename(self, nn=None):
        # {{{
        if nn is None:
            nn = self.nn
        out = self.prefix
        out += '-%d.par' % nn
        return out
        # }}}

    def write_submitfile(self, walltime=24):
        # {{{
        pwd = cmd.getoutput('pwd')
        os.chdir(self.outdir)
        if walltime > 48 or walltime < 2:
            walltime = self.walltime
        out = """\
#!/bin/env sh
#PBS -l nodes=1:ppn=8,walltime=%d:00:00
#PBS -W umask=022
#PBS -e logs/%s
#PBS -o logs/%s

# ENV for CCE-PITTNULL
module load extras/64_6.4
module load intel/12.1.3
module load openmpi/1.4.4-intel-v12.1
module load gcc/4.6.1 python/2.7.2
module load petsc/3.2_intel_openmpi_cxx
module load gsl/1.13-intel
module load hdf5

# ENV for continuation script
source /home/p/pfeiffer/prayush/local/lal_master/etc/lscsoftrc

# Turn off implicit threading in Python, R
export OMP_NUM_THREADS=1

H5FILEin=%s
H5FILEout=Cce.h5
PARFILE=%s
#RUNDIR=.
RUNDIR=%s

cp -v -L ${RUNDIR}/${H5FILEin} /dev/shm/${H5FILEout}
cp -v -L ${RUNDIR}/cactus_pittnull ${RUNDIR}/${PARFILE} /dev/shm/
cd /dev/shm/

sleep %d && cp -r ./%s %s/ && ssh gpc03 'source /home/p/pfeiffer/prayush/scratch/src/SpEC/master/SpEC/MakefileRules/this_machine.env; source /home/p/pfeiffer/prayush/local/lal_master/etc/lscsoftrc; export PYTHONPATH=/home/p/pfeiffer/prayush/.local/lib/python2.7/site-packages/:$PYTHONPATH; cd %s; python %s/%s >> conout 2>>conerr' && exit &

/scinet/gpc/mpi/openmpi/1.4.4-intel-v12.1/bin/mpirun -np 8 ./cactus_pittnull ./${PARFILE}

wait

#/scinet/gpc/mpi/openmpi/1.4.4-intel-v12.1/bin/mpirun -np 8 ${RUNDIR}/cactus_pittnull ${RUNDIR}/${PARFILE}
""" %\
            (walltime,
             self.prefix + '-%d.out' % self.nn, self.prefix + '-%d.err' % self.nn,
             self.datafile, self.get_parfilename(), self.outdir,
             (walltime - 2) * 3600, self.prefix + '-%d' % self.nn, self.outdir,
                self.outdir, '.', self.get_continuationfilename())
        #
        fout = open(self.get_submitfilename(), "w+")
        fout.write(out)
        fout.close()
        os.chdir(pwd)
        return
        # }}}

    def write_parfile(self, recover_from=None):
        # {{{
        pwd = cmd.getoutput('pwd')
        if recover_from is None:
            recover_from = self.prefix + '-0'
        os.chdir(self.outdir)
        out = """\
#------------------------------------------------------------------------------
# A parfile to do CCE with sperical harmonic coefficients of the Cauchy metric
# from a file.
#
#------------------------------------------------------------------------------

ActiveThorns = "SymBase CoordBase CartGrid3D IOASCII IOBasic IOUtil Pugh PughReduce PughSlab PUGHInterp Time"
ActiveThorns = "Fortran MoL SpaceMask LocalReduce"
ActiveThorns = "ADMBase"
ActiveThorns = "Boundary"
ActiveThorns = "AEILocalInterp NullDecomp NullGrid NullInterp NullVars NullEvolve NullNews NullSHRExtract SphericalHarmonicReconGen"
ActiveThorns = "NaNChecker"
ActiveThorns = "IOHDF5 IOHDF5Util"

"""
        #
        timestep = self.timestep  # 0.25
        out += """
#------------------------------------------------------------------------------

driver::global_nx = 22
driver::global_ny = 22
driver::global_nz = 22
driver::ghost_size = 3

grid::xyzmin = -5
grid::xyzmax = +5
grid::type = "byrange"

cactus::terminate         = time
cactus::cctk_initial_time = 0
cactus::cctk_final_time   = 1000000.00
cactus::cctk_itlast       = 1
time::timestep_method     = given
time::timestep            = %f
cactus::highlight_warning_messages = no

""" % timestep
        #
        out += """
#------------------------------------------------------------------------------
NullEvolve::boundary_data      = "SHRE"
NullEvolve::dissip_J           = 0.0
NullEvolve::first_order_scheme = yes
NullEvolve::initial_J_data     = "vanishing_J_scri"

NullSHRExtract::cr    = %d # the non-compactified worldtube radius
NullSHRExtract::mass  = 1.0
NullSHRExtract::l_max = 16


""" % self.ExtractionRadius
        #
        out += """
#Interpolation
#-----------------------------------------
NullInterp::interpolation_order = 4
NullInterp::stereo_patch_type   = "circle"
NullInterp::deriv_accuracy      = 4

"""
        #
        out += """
# Null Grid
#-----------------------------------------
NullGrid::null_rwt     = %d
NullGrid::null_xin     = 0.48
NullGrid::N_radial_pts = 151
NullGrid::N_ang_pts_inside_eq = 61
NullGrid::N_ang_stencil_size  = 2
NullGrid::N_ang_ev_outside_eq = 2
NullGrid::N_ang_ghost_pts     = 3
""" % self.ExtractionRadius
        #
        if self.maxtimelevels is None:
            nullnews_maxtimelevels = 100
        else:
            nullnews_maxtimelevels = self.maxtimelevels
        out += """
#------------------------------------------------------------------------------

NullDecomp::l_max     = 8
NullDecomp::use_rsYlm = no
NullNews::write_spherical_harmonics = yes
Nullnews::interp_to_constant_uBondi = yes
NullNews::max_timelevels       = %d
NullNews::use_linearized_omega = yes
NullNews::first_order_scheme   = no

IO::out_dir = $parfile

IOBasic::outInfo_every        = 1
IOBasic::outInfo_reductions   = "norm_inf"
IOBasic::outScalar_every      = 1
IOBasic::outScalar_reductions = "norm2 norm_inf"
IOASCII::out1D_every          = -1
IOASCII::out2D_every          = -1

IOBasic::outInfo_vars    = "NullNews::NewsB"
IOBasic::outScalar_style = "gnuplot"
IOBasic::outScalar_vars  = "NullVars::jcn[0] NullVars::jcn[20] NullNews::NewsB NullNews::News NullNews::uBondi NullVars::bcn[0] NullVars::bcn[20] NullVars::ucn[0] NullVars::ucn[20] NullVars::wcn[0] NullVars::wcn[20]"

""" % nullnews_maxtimelevels
        #
        out += """
#------------------------------------------------------------------------------

ADMBase::initial_shift = "zero"

"""
        #
        out += """
#------------------------------------------------------------------------------

NaNChecker::action_if_found = "terminate"
NaNChecker::check_every     = 1
NaNChecker::check_vars      = "all"
NaNChecker::report_max      = 1


"""
        #
        out += """
#-----------------------------------------------------------------------------
# ASCII input


SphericalHarmonicReconGen::sphere_number = 0 # sphere number counting from 0
SphericalHarmonicReconGen::time_derivative_in_file = yes
SphericalHarmonicReconGen::time_fd_order    = 4
SphericalHarmonicReconGen::time_interpolate = yes
SphericalHarmonicReconGen::verbose          = yes
SphericalHarmonicReconGen::cached_timesteps = 20
SphericalHarmonicReconGen::lmaxInFile       = 16


SphericalHarmonicReconGen::path   = "/dev/shm/"
SphericalHarmonicReconGen::format = "SpEC-H5-v2"

SphericalHarmonicReconGen::file_lapse[0]  = "Cce.h5"
SphericalHarmonicReconGen::file_shiftx[0] = "Cce.h5"
SphericalHarmonicReconGen::file_shifty[0] = "Cce.h5"
SphericalHarmonicReconGen::file_shiftz[0] = "Cce.h5"
SphericalHarmonicReconGen::file_gxx[0]    = "Cce.h5"
SphericalHarmonicReconGen::file_gxy[0]    = "Cce.h5"
SphericalHarmonicReconGen::file_gxz[0]    = "Cce.h5"
SphericalHarmonicReconGen::file_gyy[0]    = "Cce.h5"
SphericalHarmonicReconGen::file_gyz[0]    = "Cce.h5"
SphericalHarmonicReconGen::file_gzz[0]    = "Cce.h5"

SphericalHarmonicReconGen::file_lapse[1]  = "Cce.h5"
SphericalHarmonicReconGen::file_shiftx[1] = "Cce.h5"
SphericalHarmonicReconGen::file_shifty[1] = "Cce.h5"
SphericalHarmonicReconGen::file_shiftz[1] = "Cce.h5"
SphericalHarmonicReconGen::file_gxx[1]    = "Cce.h5"
SphericalHarmonicReconGen::file_gxy[1]    = "Cce.h5"
SphericalHarmonicReconGen::file_gxz[1]    = "Cce.h5"
SphericalHarmonicReconGen::file_gyy[1]    = "Cce.h5"
SphericalHarmonicReconGen::file_gyz[1]    = "Cce.h5"
SphericalHarmonicReconGen::file_gzz[1]    = "Cce.h5"

SphericalHarmonicReconGen::file_lapse[2]  = "Cce.h5"
SphericalHarmonicReconGen::file_shiftx[2] = "Cce.h5"
SphericalHarmonicReconGen::file_shifty[2] = "Cce.h5"
SphericalHarmonicReconGen::file_shiftz[2] = "Cce.h5"
SphericalHarmonicReconGen::file_gxx[2]    = "Cce.h5"
SphericalHarmonicReconGen::file_gxy[2]    = "Cce.h5"
SphericalHarmonicReconGen::file_gxz[2]    = "Cce.h5"
SphericalHarmonicReconGen::file_gyy[2]    = "Cce.h5"
SphericalHarmonicReconGen::file_gyz[2]    = "Cce.h5"
SphericalHarmonicReconGen::file_gzz[2]    = "Cce.h5"


SphericalHarmonicReconGen::column_lapse[0]  = 0
SphericalHarmonicReconGen::column_shiftx[0] = 1
SphericalHarmonicReconGen::column_shifty[0] = 2
SphericalHarmonicReconGen::column_shiftz[0] = 3
SphericalHarmonicReconGen::column_gxx[0]    = 4
SphericalHarmonicReconGen::column_gxy[0]    = 5
SphericalHarmonicReconGen::column_gxz[0]    = 6
SphericalHarmonicReconGen::column_gyy[0]    = 7
SphericalHarmonicReconGen::column_gyz[0]    = 8
SphericalHarmonicReconGen::column_gzz[0]    = 9

# radial derivatives
SphericalHarmonicReconGen::column_lapse[1]  = 10
SphericalHarmonicReconGen::column_shiftx[1] = 11
SphericalHarmonicReconGen::column_shifty[1] = 12
SphericalHarmonicReconGen::column_shiftz[1] = 13
SphericalHarmonicReconGen::column_gxx[1]    = 14
SphericalHarmonicReconGen::column_gxy[1]    = 15
SphericalHarmonicReconGen::column_gxz[1]    = 16
SphericalHarmonicReconGen::column_gyy[1]    = 17
SphericalHarmonicReconGen::column_gyz[1]    = 18
SphericalHarmonicReconGen::column_gzz[1]    = 19

# time derivatives
SphericalHarmonicReconGen::column_lapse[2]  = 20
SphericalHarmonicReconGen::column_shiftx[2] = 21
SphericalHarmonicReconGen::column_shifty[2] = 22
SphericalHarmonicReconGen::column_shiftz[2] = 23
SphericalHarmonicReconGen::column_gxx[2]    = 24
SphericalHarmonicReconGen::column_gxy[2]    = 25
SphericalHarmonicReconGen::column_gxz[2]    = 26
SphericalHarmonicReconGen::column_gyy[2]    = 27
SphericalHarmonicReconGen::column_gyz[2]    = 28
SphericalHarmonicReconGen::column_gzz[2]    = 29


"""
        #
        if self.nn > 1:
            recover = 'auto'
            recover_dir = '"%s/%s/"' % (self.outdir, recover_from)
        elif self.nn == 1:
            recover = 'autoprobe'
            recover_dir = '$parfile'
        out += """
### Checkpointing

IOHDF5::checkpoint     = yes
IO::checkpoint_ID      = no
IO::recover            = %s
IO::recover_dir        = %s
IO::checkpoint_every   = 256
IO::out_proc_every     = 2
IO::checkpoint_keep    = 3
IO::checkpoint_dir     = $parfile
IO::abort_on_io_errors = yes
""" % (recover, recover_dir)
        #
        fout = open(self.get_parfilename(), "w+")
        fout.write(out)
        fout.close()
        os.chdir(pwd)
        return
        # }}}

    def write_continuationscript(self):
        # {{{
        pwd = cmd.getoutput('pwd')
        os.chdir(self.outdir)
        out = """\
#!/bin/env python

import commands as cmd
import ccerun

# First tar the output and transfer it --tarring ==> cannot restart !
#cmd.getoutput('tar -cvf %s/%s.tar %s')

# Continue the run?
r = ccerun.cce_run( datafile='%s', datadir='%s', outdir='%s', verbose=True )

#r.setup_rerun(submit=True)
if r.is_to_be_continued_2(): r.setup_rerun(submit=True)

""" % (self.get_outdirname(), self.get_parfilename().strip('.par'),
        self.get_parfilename().strip('.par'), self.get_datafile(),
        self.get_datadirname(), self.get_outdirname())
        fout = open(self.get_continuationfilename(), "w+")
        fout.write(out)
        fout.close()
        os.chdir(pwd)
        return
        # }}}

    def setup_run(self, submit=False):
        # {{{
        pwd = cmd.getoutput('pwd')
        cmd.getoutput('mkdir -p %s' % self.outdir)
        os.chdir(self.outdir)
        #
        cmd.getoutput('ln -s %s' % (self.datadir + '/' + self.datafile))
        if not os.path.exists('./cactus_pittnull') or \
                cmd.getoutput('diff ./cactus_pittnull %s' % self.pittnull):
            cmd.getoutput('cp %s .' % self.pittnull)
        cmd.getoutput('mkdir -p logs')
        self.write_submitfile()
        self.write_parfile()
        self.write_continuationscript()
        #
        if submit:
            self.submit_run()
        os.chdir(pwd)
        return
        # }}}

    def submit_run(self):
        # {{{
        pwd = cmd.getoutput('pwd')
        os.chdir(self.outdir)
        #
        comm = '/opt/torque/bin/qsub -d %s -N %s ./%s >> sub.out' % \
            (self.outdir, self.outdir, self.get_submitfilename())
        if self.verbose:
            print("Submitting new job: %s\n\n" % comm)
        cmd.getoutput('echo %s >> sub.out' % comm)
        out = cmd.getoutput(comm)
        #
        if self.verbose:
            print(out, file=sys.stderr)
        if out:
            print("Make sure the job submission succeeded!", file=sys.stderr)
        #
        os.chdir(pwd)
        return
        # }}}

    def get_next_error_file(self):
        # Will only work when at least one run segment has been completed
        # {{{
        return 'logs/' + self.prefix + '-%d.err' % (self.nn)
        # }}}

    #

    def get_recent_error_file(self):
        # Will only work when at least one run segment has been completed
        # {{{
        return 'logs/' + self.prefix + '-%d.err' % (self.nn - 1)
        # }}}

    #

    def get_recent_output_file(self):
        # Will only work when at least one run segment has been completed
        # {{{
        return 'logs/' + self.prefix + '-%d.out' % (self.nn - 1)
        # }}}

    #

    def is_to_be_continued(self, ofile=None, checkforuncopied=False):
        """
        Checks error/output files created for walltime used. If they are not
        created yet, the job is still running. If the walltime exceeds 48 hours,
        the job did not complete. If the walltime is too less (below 120s) the job
        encountered an error while starting up. If the 120s < walltime < 48 hours,
        the job must have finished successfully.
        """
        # {{{
        pwd = cmd.getoutput('pwd')
        os.chdir(self.outdir)
        try:
            if ofile is None and not checkforuncopied:
                ofile = self.get_recent_error_file()
            elif checkforuncopied:
                ofile = self.get_next_error_file()
                if not os.path.exists(ofile):
                    print(
                        "Next segment has not written log files. Reverting to the one that has.\n",
                        file=sys.stderr)
                    ofile = self.get_recent_error_file()
            else:
                ofile = ofile
            asked, used = cmd.getoutput('/bin/cat %s | grep walltime=' %
                                        ofile).split('\n')
        except ValueError:
            odirname = ofile.split('.')[-2]
            odirname = odirname.split('/')[-1]
            print("Odirname = ", odirname)
            if os.path.exists(odirname) and os.path.getsize(odirname) > 1024:
                print("Run probably ongoing, output file not written.",
                      file=sys.stdout)
                if os.path.getmtime('%s/Psi4_scri.L02Mp02.asc' % odirname) > \
                        (time.time() - 600):
                    print("Psi4 files freshly written.", file=sys.stdout)
                else:
                    print(
                        "Psi4 files are old. Run aborted? Needs to be  restarted",
                        file=sys.stdout)
                    os.chdir(pwd)
                    return True
            else:
                print("Odirname = ", odirname)
                if self.verbose:
                    print("No run ongoing", file=sys.stdout)
            os.chdir(pwd)
            return False
        ##
        #
        print(ofile)
        ##
        usedtime = int(getSec(used.split('=')[-1]))
        askedtime = int(getSec(asked.split('=')[-1]))
        if usedtime < askedtime:
            if usedtime < 120:
                if cmd.getoutput(
                        r'/bin/cat %s | grep Requested\ timestep\ not\ in\ worldtube\ data\ file'
                        % ofile):
                    if self.verbose:
                        print("UseTime = ", usedtime, "\n", file=sys.stdout)
                        print(
                            "Warning: %s stopped too soon, but the Run might have completed to begin with."
                            % self.outdir,
                            file=sys.stdout)
                    os.chdir(pwd)
                    return False
                elif self.verbose:
                    print("Warning: %s Run stopped too soon!" % (self.outdir),
                          file=sys.stdout)
                os.chdir(pwd)
                return False
        if 'Requested' in cmd.getoutput(
                r'/bin/cat %s | grep Requested\ timestep\ not\ in\ worldtube\ data\ file'
                % ofile) or 'Done' in cmd.getoutput(
                    '/bin/cat %s | grep Done' % ofile):
            print("PASSED TEST FOR STRING IN ERR FILE", ofile)
            print(cmd.getoutput(
                r'/bin/cat %s | grep Requested\ timestep\ not\ in\ worldtube\ data\ file'
                % ofile) != '')
            print(cmd.getoutput('/bin/cat %s | grep Done' % ofile))  # != ''
            if self.verbose:
                print(
                    "Run %s either Done or requested for timestep not in worldtube data file\n"
                    % (self.outdir),
                    file=sys.stdout)
            if checkforuncopied:
                # Check if the output has been copied over
                if not os.path.exists(self.prefix + '-%d' % (self.nn)):
                    print("..but data has not been copied over.",
                          file=sys.stdout)
                    return True
            os.chdir(pwd)
            return False
        else:
            print("DID NOT PASS TEST FOR STRING IN ERR FILE", ofile)
            print(cmd.getoutput(
                r'/bin/cat %s | grep Requested\ timestep\ not\ in\ worldtube\ data\ file'
                % ofile) != '')
            print(cmd.getoutput('/bin/cat %s | grep Done' % ofile))  # != ''
        if self.verbose:
            print("Run needs to be restarted", file=sys.stdout)
        os.chdir(pwd)
        return True
        # else:
        #  if self.verbose:
        #    print >>sys.stdout, "USed = %d, Asked = %d" % (usedtime, askedtime)
        #  if self.verbose: print >>sys.stdout, "Run might have completed"
        #  if checkforuncopied:
        #    # Check if the output has been copied over
        #    if not os.path.exists(self.prefix + '-%d' % (self.nn)):
        #      print >>sys.stdout, "..but data has not been copied over."
        #      return True
        #  #qq = int(cmd.getoutput('tail %s | grep Killing | wc -l' % ofile))
        #  return False
        # }}}

    #

    def is_to_be_continued_2(self,
                             abs_eps=1.e-10,
                             ofile='Psi4_scri.L02Mp02.asc',
                             verbose=False):
        """
        Checks error/output files created for walltime used. If they are not
        created yet, the job is still running. If the walltime exceeds 48 hours,
        the job did not complete. If the walltime is too less (below 120s) the job
        encountered an error while starting up. If the 120s < walltime < 48 hours,
        the job must have finished successfully.
        """
        # {{{
        pwd = cmd.getoutput('pwd')
        os.chdir(self.outdir)
        #
        prefix = self.outdir + '/' + self.prefix
        end_times = []
        for i in range(1, self.nn):
            if verbose:
                print("checking ", os.path.join(prefix + '-%d' % i, ofile))
            psi_d = np.loadtxt(os.path.join(prefix + '-%d' % i, ofile))
            try:
                end_times.append(psi_d[-1, 0])
            except IndexError:
                end_times.append(psi_d[0])
        #
        if abs(end_times[-1] - end_times[-2]) < abs_eps:
            if verbose:
                print(
                    "Last two segments ended at the same time, must have completed"
                )
            return False
        else:
            return True
        # }}}

    #

    def setup_rerun(self, submit=False):
        # ASSUMPTION, ALL RUNS ARE CONTINUOUSLY NUMBERED, as CceR0100-1,2,..
        # {{{
        if self.nn == 1:
            self.setup_run(submit=submit)
            return
        pwd = cmd.getoutput('pwd')
        os.chdir(self.outdir)
        #
        self.write_submitfile()
        # if self.nn > 1:
        #  cmd.getoutput('ln -s %s %s' % \
        #    (self.prefix + '-%d' % (self.nn-1), self.prefix + '-0'))
        recover_dir = self.get_parfilename(nn=self.nn - 1).split('.par')[0]
        self.write_parfile(recover_from=recover_dir)
        self.write_continuationscript()
        #
        if submit:
            self.submit_run()
        os.chdir(pwd)
        return
        # }}}

    def submit_rerun(self):
        # {{{
        self.submit_run()
        return
        # }}}

    #

    def combine_output(self, subdirs=None, redo=True):
        # {{{
        self.redo = redo
        pwd = cmd.getoutput('pwd')
        os.chdir(self.outdir)
        # Check if the output directory already exists. Assume output does as well
        # if the directory does
        outdir = './' + self.prefix + '.joined'
        if os.path.exists(outdir):
            print("output for %s has been joined already.!" %
                  cmd.getoutput('pwd'),
                  file=sys.stderr)
            if not self.redo:
                os.chdir(pwd)
                return
        else:
            os.mkdir(outdir)
        idir = '.'
        #
        # Begin
        #
        # Get the list of output dirs
        #
        # Override if subdirectories provided
        if subdirs is None:
            subdirs = self.get_all_subdirs()
        if int(subdirs[0][-1]) == 0:
            subdirs = subdirs[1:]
        if subdirs is None or len(subdirs) == 0:
            raise IOError("No directories of the form %s-?. Wrong tag?" %
                          self.prefix)
        # subdirs.sort()
        # Append the most recent portion of the run
        #if os.path.exists(idir+'/highResCce'): subdirs.append(idir+'/highResCce')
        if self.verbose:
            print("directories used: ", subdirs, file=sys.stderr)
        #
        # Join files of each tag
        #
        for ftag in self.filetags:
            #
            # Assume all dirs in subdirs have the same files satisfying the
            # ftag
            tmp_files = glob.glob(subdirs[0] + '/' + ftag)
            file_names = [dd.split('/')[-1] for dd in tmp_files]
            if self.verbose:
                print("files found: ", file_names, file=sys.stderr)
            if self.verbose:
                print("total: %d" % len(file_names), file=sys.stderr)
            for fnam in file_names:
                outfnam = outdir + '/' + fnam
                if os.path.exists(outfnam) and os.path.getsize(
                        outfnam) and not self.redo:
                    print("\nNOT Joining: ",
                          fnam,
                          " in ",
                          subdirs,
                          file=sys.stderr)
                    continue
                if self.verbose:
                    print("\nJoining: ",
                          fnam,
                          " in ",
                          subdirs,
                          file=sys.stderr)
                # Following convoluted way of reading data is for runs which die unsafely
                # and might have partially flushed lines trailing as unprintable characters
                # at the end of the data file.
                # As subsequent segments are substantially overlapping,
                # actual data will not be lost by ignoring these trailing
                # lines.
                data = []
                for sdir in subdirs:
                    try:
                        tmp_data = np.loadtxt(open(sdir + '/' + fnam))
                        # It may happen that the output file has only ONE line,
                        # handle that!
                        if len(np.shape(tmp_data)) < 2:
                            tmp_data = np.array([tmp_data])
                        data.append(tmp_data)
                    except BaseException:
                        fin = open(sdir + '/' + fnam).readlines()
                        tmp_data = []
                        for line in fin:
                            try:
                                tmp_data.append(np.float128(line.split()))
                            except BaseException:
                                continue
                        if not len(tmp_data):
                            data.append(np.array(tmp_data))
                #
                # Assume the data is in chronological order in the files
                tarr, rearr, imarr = [
                    data[0][:, 0], data[0][:, 1], data[0][:, 2]
                ]
                for dd in data[1:]:
                    tmp_t = dd[:, 0]
                    try:
                        tmp_startidx = np.where(tmp_t > tarr[-1])[0][0]
                    except IndexError:
                        continue
                    tarr = np.append(tarr, dd[tmp_startidx:, 0])
                    rearr = np.append(rearr, dd[tmp_startidx:, 1])
                    imarr = np.append(imarr, dd[tmp_startidx:, 2])
                #
                # Write the joined data to disk
                #
                if self.verbose:
                    print("Writing to: %s" % (outdir + '/' + fnam),
                          file=sys.stderr)
                fout = open(outfnam, 'w')
                for i in range(len(tarr)):
                    fout.write('%.16e\t%.16e\t%.16e\n' %
                               (tarr[i], rearr[i], imarr[i]))
                fout.close()
            #
        os.chdir(pwd)
        return
        # }}}

    #

    def uniformly_sample_output(self, joineddir=None):
        # Uniformly sample the PSi4 and NewsB modes
        # ASSUMPTION, ALL RUNS ARE CONTINUOUSLY NUMBERED, as CceR0100-1,2,..
        # {{{
        pwd = cmd.getoutput('pwd')
        os.chdir(self.outdir)
        # Check if the output directory already exists. Assume output does as well
        # if the directory does
        if joineddir is None:
            outdir = './' + self.prefix + '.joined'
        else:
            outdir = joineddir
        if not os.path.exists(outdir):
            print("Outdir is ", outdir)
            raise RuntimeError(
                "The director for combined PSi4 output doesn't exist" % outdir)
        else:
            os.chdir(outdir)
        #
        for ftag in self.filetags:
            fnames = glob.glob(ftag)
            for fnam in fnames:
                fin = open(fnam, 'r')
                pdt = np.loadtxt(fin)
                fin.close()
                #
                dt = pdt[1, 0] - pdt[0, 0]
                pdtreali = interp1d(pdt[:, 0], pdt[:, 1])
                pdtimagi = interp1d(pdt[:, 0], pdt[:, 2])
                #
                timeout = arange(pdt[:, 0].min(), pdt[:, 0].max(), dt)
                prealout = pdtreali(timeout)
                pimagout = pdtimagi(timeout)
                #
                fout = open(fnam.split('.asc')[0] + '_uform.asc', 'w')
                for i in range(len(timeout)):
                    fout.write("%.16e\t%.16e\t%.16e\n" %
                               (timeout[i], prealout[i], pimagout[i]))
                #
                fout.close()
                #
                if self.verbose:
                    print("Uniformly sampled %s" % fnam, file=sys.stderr)
        #
        os.chdir(pwd)
        # }}}

    #

    def integrate_psi4_to_hlm(self,
                              joineddir=None,
                              fstring=None,
                              inputdir=None,
                              datafile=None,
                              datatype='ASCII',
                              outputdir=None,
                              outputtype='ASCII',
                              resample=True,
                              lmax=8,
                              ffifreq=0.005,
                              m0_time_domain=True,
                              align_time_to_amax=False,
                              align_time_to_psi4=True):
        """
    Integrate the PSi4 modes to strain modes
    using the FFI method from http://arxiv.org/abs/1006.1632

      joineddir : path-Where is the output to be stored?
      inputdir  : path-Where is/are the input file/s currently
      fstring   : string-Tag string which is present at the beginning of input psi4 files
      datafile  : string-Name of the data file in case there's a single one (HDF5)
      datatype  : string-'ASCII' or 'HDF5' or 'HDFfp'
      outputtype: string-'HDF5'
      resample  : bool-Whether to re-sample the Psi4 files
      lmax      : int-Maximum value of L to go to
      ffifreq   : float-Value of FFI cut-off frequency
      m0_time_domain : bool-Whether to integrate l=0 modes in time-domain
      ASCII : joineddir = inputdir = CceR0XXX.joined
      HDF : inputdir = ., outdir = .
        """
        # {{{
        try:
            from scipy.integrate import simps
        except BaseException:
            raise ImportError("Could not import simps from scipy.integrate")
        pwd = cmd.getoutput('pwd')
        os.chdir(self.outdir)
        # Check if the output directory already exists. Assume output does as well
        # if the directory does
        if joineddir is None:
            outdir = './' + self.prefix + '.joined'
        else:
            outdir = joineddir
        if self.verbose:
            print("Integrating modes in %s" % outdir)
        if not os.path.exists(outdir):
            raise RuntimeError(
                "The director for combined PSi4 output doesn't exist" % outdir)
        elif resample:
            # Uniformly sample the Psi4 modes
            self.uniformly_sample_output(joineddir=outdir)
        #
        # Where are the input files to be read from?
        if 'ASCII' in datatype:
            if inputdir is not None:
                initial_dir = inputdir
            else:
                initial_dir = outdir
        elif 'HDF' in datatype:
            if inputdir is None:
                inputdir = '.'
            datafin = h5py.File(os.path.join(inputdir, datafile), 'r')
        #
        # FFI cutoff frequency. This must be choosen
        # smaller than any physically expected frequency.
        f0 = ffifreq / (2 * pi)
        #
        # Initialize a mode array for psi4
        WF = InitModeArray(lmax)
        # Initialize a mode array for h
        WFint = InitModeArray(lmax)
        #
        # Load each psi4-mode into a WaveFunction object and store it in mode
        # array
        for l in range(2, lmax + 1):
            for m in range(-l, l + 1):
                if m == 0 and m0_time_domain:
                    continue  # Handle m=0 modes in time-domain
                if self.verbose:
                    print("Load (l,m) = ", l, m)
                WF[l][m] = WaveFunction([], [])
                if 'ASCII' in datatype:
                    if fstring is not None:
                        fnam = initial_dir + '/' + fstring % (l, m)
                    else:
                        if m < 0:
                            mstr = 'm%02d' % abs(m)
                        else:
                            mstr = 'p%02d' % abs(m)
                        fnam = ("%s/Psi4_scri.L%02dM" % (initial_dir, l)) +\
                            mstr + "_uform.asc"
                    if self.verbose:
                        print("reading %s" % fnam)
                    WF[l][m].Load(fnam)
                elif 'HDF5' in datatype:
                    ccegrp = self.prefix + '.dir'
                    WF[l][m].Load(datafin[ccegrp]['Y_l%d_m%d.dat' %
                                                  (l, m)].value,
                                  datatype='HDFfp')
                else:
                    raise IOError("datatype must be either ASCII or HDF5")
                # To be consistent with NumRel, multiply by 2 and conjugate
                for i in range(len(WF[l][m].f)):
                    WF[l][m].f[i] = 2 * conjugate(WF[l][m].f[i])
        #
        # Integrate 2,2 mode using FFI
        WFint[2][2] = GethFromPsi4(WF[2][2], f0)
        # Get time of merger from maximum of 2,2-amplitude
        tmerger = WFint[2][2].Amplitude().FindAbsMaxInterpolated(1e-4)
        #
        # Integrate all remaining modes, shift them according to tmerger, and
        # store them in a file.
        if self.verbose:
            print("\ntmerger = %f\n" % tmerger)
        for l in range(2, lmax + 1):
            for m in range(-l, l + 1):
                if m == 0 and m0_time_domain:
                    continue  # Handle m=0 modes in time-domain
                if self.verbose:
                    print("Integrating (l,m) = ", l, m)
                WFint[l][m] = GethFromPsi4(WF[l][m], f0 * m * 0.5)
                if align_time_to_amax:
                    WFint[l][m].x -= tmerger
                elif align_time_to_psi4:
                    WFint[l][m].x = (WFint[l][m].x -
                                     WFint[l][m].x[0]) + WF[l][m].x[0]
                if m < 0:
                    mstr = 'm%02d' % abs(m)
                else:
                    mstr = 'p%02d' % abs(m)
                fnam = ("%s/h_from_Psi4_scri.L%02dM" %
                        (outdir, l)) + mstr + ".dat"
                fout = open(fnam, 'w')
                fout.write("# [1] = t/M\n")
                fout.write("# [2] = Re{rhOverM(%d,%d)}\n" % (l, m))
                fout.write("# [3] = Im{rhOverM(%d,%d)}\n" % (l, m))
                fout.write("# FFI cut-off: omega = " + str(f0 * pi * m) + "\n")
                for ii in range(0, WFint[l][m].Length()):
                    fout.write("%.12e\t%.12e\t%.12e\n" %
                               (WFint[l][m].x[ii], WFint[l][m].f[ii].real,
                                WFint[l][m].f[ii].imag))
                fout.close()
        #
        # Take care of the (l, m=0) modes in time domain
        # Integrate Psi4 & Write to disk
        m = 0
        for l in range(2, lmax + 1):
            if not m0_time_domain:
                continue
            if self.verbose:
                print("Integrating (l,m) = ", l, m)
            if 'ASCII' in datatype:
                if fstring is not None:
                    fnam = initial_dir + '/' + fstring % (l, m)
                else:
                    fnam = ("%s/Psi4_scri.L%02dMp%02d" %
                            (initial_dir, l, m)) + "_uform.asc"
                    if self.verbose:
                        print("reading %s" % fnam)
                    data = np.loadtxt(fnam)
            elif 'HDF5' in datatype:
                data = datafin[ccegrp]['Y_l%d_m0.dat' % l].value
            tarr = data[:, 0]
            tarr -= (tarr[0] - WFint[2][2].x[0])
            parr = data[:, 1] + data[:, 2] * 1.0j
            # Get News by one time-integration
            narr = np.array([
                simps(parr[:i], x=tarr[:i], dx=tarr[1] - tarr[0])
                for i in range(1, len(parr))
            ])
            narr = narr - narr[len(tarr) / 2]
            # Get strain by second time-integration
            harr = np.array([
                simps(narr[:i], x=tarr[:i], dx=tarr[1] - tarr[0])
                for i in range(1, len(narr))
            ])
            harr = harr - harr[len(harr) / 2]
            #
            fnam = ("%s/h_from_Psi4_scri.L%02dM" % (outdir, l)) + "p00.dat"
            fout = open(fnam, 'w')
            fout.write("# [1] = t/M\n")
            fout.write("# [2] = Re{rhOverM(%d,%d)}\n" % (l, m))
            fout.write("# [3] = Im{rhOverM(%d,%d)}\n" % (l, m))
            fout.write("# FFI cut-off: omega = " + str(f0 * pi * m) + "\n")
            for ii in range(0, len(harr)):
                fout.write("%.12e\t%.12e\t%.12e\n" %
                           (tarr[ii], harr[ii].real, harr[ii].imag))
            fout.close()
        #
        os.chdir(pwd)
        if 'HDF' in outputtype:
            if self.verbose:
                print("Writing Hlm to disk")
            print("outdir (joineddir) = ", outdir)
            print("outputdir (outdir) = ", outputdir)
            self.write_to_hdf5(
                joineddir=outdir,
                outdir=outputdir,
                lmax=lmax,
                prefix='h_from_Psi4_scri',
                postfix='.dat',
                filename='rhOverM_FromPsi4_CcePITT_Asymptotic_GeometricUnits.h5'
            )
        return
        # }}}

    #

    def integrate_news_to_hlm(self,
                              joineddir=None,
                              fstring=None,
                              inputdir=None,
                              datafile=None,
                              datatype='ASCII',
                              outputdir='.',
                              outputtype='HDF',
                              resample=True,
                              lmax=8,
                              ffifreq=0.005,
                              m0_time_domain=True):
        """
    Integrate the News modes to strain modes
    using the FFI method from http://arxiv.org/abs/1006.1632

      joineddir : path-Where is the output to be stored?
      inputdir  : path-Where is/are the input file/s currently
      fstring   : string-Tag string which is present at the beginning of input psi4 files
      datafile  : string-Name of the data file in case there's a single one (HDF5)
      datatype  : string-'ASCII' or 'HDF5' or 'HDFfp'
      outputtype: string-'HDF5'
      resample  : bool-Whether to re-sample the Psi4 files
      lmax      : int-Maximum value of L to go to
      ffifreq   : float-Value of FFI cut-off frequency
      m0_time_domain : bool-Whether to integrate l=0 modes in time-domain
      ASCII : joineddir = inputdir = CceR0XXX.joined
      HDF : inputdir = ., outdir = .
        """
        # {{{
        try:
            from scipy.integrate import simps
        except BaseException:
            raise ImportError("Could not import simps from scipy.integrate")
        pwd = cmd.getoutput('pwd')
        os.chdir(self.outdir)
        # Check if the output directory already exists. Assume output does as well
        # if the directory does
        if joineddir is None:
            outdir = './' + self.prefix + '.joined'
        else:
            outdir = joineddir
        if self.verbose:
            print("Integrating modes in %s" % outdir)
        if not os.path.exists(outdir):
            raise RuntimeError(
                "The director for combined PSi4 output doesn't exist" % outdir)
        elif resample:
            # Uniformly sample the News modes
            self.uniformly_sample_output(joineddir=outdir)
        #
        # Where are the input files to be read from?
        if 'ASCII' in datatype:
            if inputdir is not None:
                initial_dir = inputdir
            else:
                initial_dir = outdir
        elif 'HDF' in datatype:
            if inputdir is None:
                inputdir = '.'
            datafin = h5py.File(os.path.join(inputdir, datafile), 'r')
        #
        # FFI cutoff frequency. This must be choosen
        # smaller than any physically expected frequency.
        f0 = ffifreq / (2 * pi)
        #
        # Initialize a mode array for psi4
        WF = InitModeArray(lmax)
        # Initialize a mode array for h
        WFint = InitModeArray(lmax)
        #
        # Load each psi4-mode into a WaveFunction object and store it in mode
        # array
        for l in range(2, lmax + 1):
            for m in range(-l, l + 1):
                if m == 0 and m0_time_domain:
                    continue  # Handle m=0 modes in time-domain
                if self.verbose:
                    print("Load (l,m) = ", l, m)
                WF[l][m] = WaveFunction([], [])
                if 'ASCII' in datatype:
                    if fstring is not None:
                        fnam = initial_dir + '/' + fstring % (l, m)
                    else:
                        if m < 0:
                            mstr = 'm%02d' % abs(m)
                        else:
                            mstr = 'p%02d' % abs(m)
                        fnam = ("%s/NewsB_scri.L%02dM" % (initial_dir, l)) +\
                            mstr + "_uform.asc"
                    if self.verbose:
                        print("reading %s" % fnam)
                    WF[l][m].Load(fnam)
                elif 'HDF5' in datatype:
                    ccegrp = self.prefix + '.dir'
                    WF[l][m].Load(datafin[ccegrp]['Y_l%d_m%d.dat' %
                                                  (l, m)].value,
                                  datatype='HDFfp')
                else:
                    raise IOError("datatype must be either ASCII or HDF5")
                # To be consistent with NumRel, multiply by 2 and conjugate
                for i in range(len(WF[l][m].f)):
                    WF[l][m].f[i] = 2 * conjugate(WF[l][m].f[i])
        #
        # Integrate 2,2 mode using FFI
        WFint[2][2] = GethFromNews(WF[2][2], f0)
        # Get time of merger from maximum of 2,2-amplitude
        tmerger = WFint[2][2].Amplitude().FindAbsMaxInterpolated(1e-4)
        print(WFint[2][2].x[0], tmerger)
        # Integrate all remaining modes, shift them according to tmerger, and
        # store them in a file.
        if self.verbose:
            print("\ntmerger = %f\n" % tmerger)
        for l in range(2, lmax + 1):
            for m in range(-l, l + 1):
                if m == 0 and m0_time_domain:
                    continue  # Handle m=0 modes in time-domain
                if self.verbose:
                    print("Integrating (l,m) = ", l, m)
                WFint[l][m] = GethFromNews(WF[l][m], f0 * m * 0.5)
                WFint[l][m].x -= tmerger
                if m < 0:
                    mstr = 'm%02d' % abs(m)
                else:
                    mstr = 'p%02d' % abs(m)
                fnam = ("%s/h_from_News_scri.L%02dM" %
                        (outdir, l)) + mstr + ".dat"
                fout = open(fnam, 'w')
                fout.write("# [1] = t/M\n")
                fout.write("# [2] = Re{rhOverM(%d,%d)}\n" % (l, m))
                fout.write("# [3] = Im{rhOverM(%d,%d)}\n" % (l, m))
                fout.write("# FFI cut-off: omega = " + str(f0 * pi * m) + "\n")
                for ii in range(0, WFint[l][m].Length()):
                    fout.write(
                        str(WFint[l][m].x[ii]) + " " +
                        str(WFint[l][m].f[ii].real) + " " +
                        str(WFint[l][m].f[ii].imag) + "\n")
                fout.close()
        #
        # Take care of the (l, m=0) modes in time domain
        # Integrate News & Write to disk
        for l in range(2, lmax + 1):
            if not m0_time_domain:
                continue
            if self.verbose:
                print("Integrating (l,m) = ", l, 0)
            data = datafin[ccegrp]['Y_l%d_m0.dat' % l].value
            tarr = data[:, 0]
            tarr -= (tarr[0] - WFint[2][2].x[0])
            narr = data[:, 1] + data[:, 2] * 1.0j
            # Get News by one time-integration
            harr = np.array([
                simps(narr[:i], x=tarr[:i], dx=tarr[1] - tarr[0])
                for i in range(1, len(narr))
            ])
            harr = harr - harr[len(tarr) / 2]
            #
            fnam = ("%s/h_from_News_scri.L%02dM" % (outdir, l)) + "p00.dat"
            fout = open(fnam, 'w')
            fout.write("# [1] = t/M\n")
            fout.write("# [2] = Re{rhOverM(%d,%d)}\n" % (l, m))
            fout.write("# [3] = Im{rhOverM(%d,%d)}\n" % (l, m))
            fout.write("# FFI cut-off: omega = " + str(f0 * pi * m) + "\n")
            for ii in range(0, len(harr)):
                fout.write("%.12e\t%.12e\t%.12e\n" %
                           (tarr[ii], harr[ii].real, harr[ii].imag))
            fout.close()
        #
        os.chdir(pwd)
        if 'HDF' in outputtype:
            if self.verbose:
                print("Writing Hlm to disk")
            self.write_to_hdf5(
                outdir=outputdir,
                joineddir=outdir,
                lmax=lmax,
                prefix='h_from_News_scri',
                filename=
                'rhOverM_FromNews_CcePITT_Asymptotic_GeometricUnits_m00.h5')
        return
        # }}}

    #
    # Note that the file postfix (not pre) needs a dot (.) in it explicitly.

    def write_to_hdf5(self,
                      prefix='h_from_Psi4_scri',
                      postfix='.dat',
                      outdir=None,
                      joineddir=None,
                      filename='rhOverM_CcePITT_Asymptotic_GeometricUnits.h5',
                      lmax=8,
                      replace=False):
        """
        Arguments:
        outdir     : string- PATH W.R.T. THE LEV DIRECTORY
        joineddir  : string- PATH W.R.T. THE LEV DIRECTORY
        filename   : string- NAME OF OUTPUT FILE
        prefix     : string- PREFIX- OF THE INPUT FILES, THE CONTENTS OF WHICH ARE
                      TO BE WRITTEN TO hdf
        postfix    : string- POSTFIX or EXTENSION OF THE INPUT FILES
        """
        # {{{
        import h5py
        pwd = cmd.getoutput('pwd')
        if outdir is None:
            outdir = self.outdir
        os.chdir(outdir)
        if filename is None:
            raise IOError("Please specify the name of output HDF5 file.")
        #
        fin = h5py.File(filename, 'a')  # Already in outdir
        grpname = self.datafile.split('/')[-1].replace('h5', 'dir')
        print("group name = ", grpname)
        print("datafile name = ", self.datafile)
        if grpname in list(fin.keys()) and not replace:
            fin.close()
            os.chdir(pwd)
            return
        elif grpname in list(fin.keys()) and replace:
            raise IOError("Replacing dataset is not supported yet!")
        #
        fin.create_group(grpname)
        for l in range(2, lmax + 1):
            for m in range(-l, l + 1):
                #
                if m < 0:
                    mstr = 'm%02d' % abs(m)
                else:
                    mstr = 'p%02d' % abs(m)
                if joineddir is not None:
                    fname = os.path.join(joineddir, prefix)
                else:
                    fname = self.datafile.replace('h5',
                                                  'joined') + '/' + prefix
                fname = fname + (".L%02dM" % l) + mstr + postfix
                #
                datasetname = 'Y_l%d_m%d.dat' % (l, m)
                fin[grpname].create_dataset(datasetname,
                                            data=np.loadtxt(fname))
                print("Written dataset ", datasetname)
        #
        fin.flush()
        fin.close()
        os.chdir(pwd)
        return
        # }}}

    #

    def tar_output(self, remove=True):
        # {{{
        pwd = cmd.getoutput('pwd')
        os.chdir(self.outdir)
        # Check if the output directory already exists. Assume output does as well
        # if the directory does
        nfiles = int(cmd.getoutput('/bin/ls | grep %s | wc -l' % self.prefix))
        if nfiles == 0:
            print("No files found!!", file=sys.stderr)
            return
        elif nfiles == 1 and os.path.exists(self.prefix + '.tar.gz'):
            print("only 1 file found. Its the tar file", file=sys.stderr)
            return
        #
        # To have gotten this far means that Output directories exist
        # Before tarring, make sure that any old tar files are renamed
        # (CRITICAL)
        if os.path.exists(self.prefix + '.tar.gz'):
            cmd.getoutput(
                'mv %s %s' %
                (self.prefix + '.tar.gz', self.prefix + '.tar.gz.old'))
        to_run = 'tar -cvzf %s.tar.gz `/bin/ls | grep %s`' % \
            (self.prefix, self.prefix)
        if remove:
            to_run = to_run + ' --remove-files'
        print("Running %s\n" % to_run, file=sys.stderr)
        print(cmd.getoutput(to_run), file=sys.stdout)
        os.chdir(pwd)
        return
        # }}}

    #

    def untar_output(self, to_be_extracted=None):
        # {{{
        if to_be_extracted is None:
            to_be_extracted = self.prefix + '.joined'
        pwd = cmd.getoutput('pwd')
        os.chdir(self.outdir)
        # Check fi the output directory already exists
        if os.path.exists(self.prefix + '.joined'):
            if self.verbose:
                print("Output joined directory exists. Not Untarring")
            os.chdir(pwd)
            return
        to_run = 'tar -xvzf %s.tar.gz %s' % (self.prefix, to_be_extracted)
        if self.verbose:
            print("Running %s\n" % to_run, file=sys.stderr)
        cmdout = cmd.getoutput(to_run)
        if self.verbose:
            print(cmdout, file=sys.stdout)
        #
        os.chdir(pwd)
        return
        # }}}

    # }}}


#
# Class that calculates basic quantities of Psi4 modes that are usually used
# when comparing the output of CCE with different configurations.
# The inputs are:
# indir = EXACT directory where Psi4 fiels are stored
# inprefix = prefix in the names of the Psi4 mode files, before L02Mp02
# inpostfix = postfix in the names of the Psi4 modes, after L02Mp02
# lmax = MAX l value of the modes the user wants to manipulate


class psi4_waveforms():
    # {{{
    def __init__(self,
                 indir='.',
                 filename=None,
                 inprefix='Psi4_scri.',
                 inpostfix='_uform.asc',
                 lmax=8,
                 verbose=True):
        #
        self.indir = indir
        self.infilename = filename
        self.inprefix = inprefix
        self.inpostfix = inpostfix
        #
        self.lmax = lmax
        self.verbose = verbose

    #

    def read_psi4_waveforms(self):
        # {{{
        self.waveforms = {}
        for l in arange(2, self.lmax + 1):
            self.waveforms[l] = {}
            for m in arange(-l, l + 1):
                if m < 0:
                    sgn = 'm'
                else:
                    sgn = 'p'
                fnam = (self.inprefix + 'L%02dM' + sgn + '%02d' +
                        self.inpostfix) % (l, abs(m))
                fnam = self.indir + '/' + fnam
                if self.verbose:
                    print("Reading Psi4 data from ", fnam)
                fin = open(fnam)
                data = loadtxt(fin)
                fin.close()
                hp = TimeSeries(data[:, 1], delta_t=data[1, 0] - data[0, 0])
                hc = TimeSeries(data[:, 2], delta_t=data[1, 0] - data[0, 0])
                self.waveforms[l][m] = [hp, hc]
        return
        # }}}

    #

    def read_psi4_waveforms_hdf5(self, R=100):
        # {{{
        self.waveforms = {}
        fnam = self.indir + '/' + self.infilename
        fin = h5py.File(fnam, 'r')
        if self.verbose:
            print("Reading Psi4 data from ", fnam)
        for l in arange(2, self.lmax + 1):
            self.waveforms[l] = {}
            for m in arange(-l, l + 1):
                data = fin["R%04d.dir" % R]["Y_l%d_m%d.dat" % (l, m)]
                hp = TimeSeries(data[:, 1], delta_t=data[1, 0] - data[0, 0])
                hc = TimeSeries(data[:, 2], delta_t=data[1, 0] - data[0, 0])
                self.waveforms[l][m] = [hp, hc]
        fin.close()
        return
        # }}}

    #

    def get_psi4_phases(self):
        # {{{
        try:
            type(self.waveforms)
        except AttributeError:
            self.read_psi4_waveforms()
        self.wavephases = {}
        for l in arange(2, self.lmax + 1):
            self.wavephases[l] = {}
            for m in arange(-l, l + 1):
                if self.verbose:
                    print("Calculating Phi(t) for (l,m) = ", l, m)
                hp, hc = self.waveforms[l][m]
                self.wavephases[l][m] = TimeSeries(arctan2(hp.data, hc.data),
                                                   delta_t=hp.delta_t,
                                                   copy=True)
        return
        # }}}

    #

    def get_psi4_amplitudes(self):
        # {{{
        try:
            type(self.waveforms)
        except AttributeError:
            self.read_psi4_waveforms()
        self.waveamplitudes = {}
        for l in arange(2, self.lmax + 1):
            self.waveamplitudes[l] = {}
            for m in arange(-l, l + 1):
                if self.verbose:
                    print("Calculating A(t) for (l,m) = ", l, m)
                hp, hc = self.waveforms[l][m]
                self.waveamplitudes[l][m] = amplitude_from_polarizations(
                    hp, hc)
        return
        # }}}

    #

    def get_psi4_peaktimes(self):
        # {{{
        try:
            type(self.waveamplitudes)
        except AttributeError:
            self.get_psi4_amplitudes()
        self.peaktimes = {}
        for l in arange(2, self.lmax + 1):
            self.peaktimes[l] = {}
            for m in arange(-l, l + 1):
                if self.verbose:
                    print("Calculating t(max(A(t))) for (l,m) = ", l, m)
                am = self.waveamplitudes[l][m]
                self.peaktimes[l][m] = am.max_loc()[-1] * am.delta_t
        return
        # }}}

    # }}}


#
# Class containing the information for ONE NR RUN. All Levs and All Extraction
# radii are covered.
# The inputs are:
# datadir = EXACT directory for the run (one level above Ccedata).
# outdir= EXACT directory where the run is to be set up
# time the allowed walltime is exceeded.
# modeldir = directory with some model files to be used TBD


class nr_run_cce():
    # {{{
    def __init__(
            self,
            datadir=None,
            subdatadir='CceData',
            outdir=None,
            pittnull='/home/p/pfeiffer/prayush/src/cactus_cce/Cactus/exe/cactus_pittnull',
            ccemodeldir='/home/p/pfeiffer/prayush/scratch/projects/CCE_modeldir/',
            cceparfile='highResCce700.par',
            increasemaxtimelevels=False,
            sample_rate=16384,
            time_length=16,
            verbose=True):
        # {{{
        if datadir is None:
            print("No CCE data file directory specified")
        if outdir is None:
            raise IOError("Where to set up the CCE run?")
        if not os.path.exists(pittnull):
            print("Cannot located pittnull exe")
        #
        self.datadir = datadir
        self.subdatadir = subdatadir
        self.outdir = outdir
        self.pittnull = pittnull
        self.sample_rate = sample_rate
        self.time_length = time_length
        self.verbose = verbose
        #
        self.ccemodeldir = ccemodeldir
        self.cceparfile = cceparfile
        #
        if not os.path.exists(str(datadir)) and os.path.exists(str(outdir)):
            self.levs = cmd.getoutput('/bin/ls %s/ | grep Lev' %
                                      (self.outdir)).split()
            return
        elif not os.path.exists(str(datadir)) and not os.path.exists(
                str(outdir)):
            raise IOError(
                "Must specify either the data or the output directory")
        #
        self.levs = cmd.getoutput('/bin/ls %s/%s | grep Lev' %
                                  (self.datadir, self.subdatadir)).split()
        self.ccefiles, self.cce_runs = {}, {}
        for ld in self.levs:
            self.cce_runs[ld], tmplist = {}, []
            # List of H5 files in alphabetical order -- which would coincide with
            # ordering in Extraction Radius
            self.ccefiles[ld] = list(
                np.sort(
                    glob.glob('%s/%s/%s/CceR*.h5' %
                              (self.datadir, self.subdatadir, ld))))
            #
            for cc in self.ccefiles[ld]:
                ccfnam = cc.split('/')[-1]
                tmplist.append(ccfnam)
                self.cce_runs[ld][ccfnam] = cce_run(
                    datafile=ccfnam,
                    datadir='%s/%s/%s' % (self.datadir, self.subdatadir, ld),
                    outdir='%s/%s' % (self.outdir, ld),
                    pittnull=pittnull,
                    verbose=self.verbose)
            #
            self.ccefiles[ld] = tmplist
        # }}}

    #

    def get_nrprefix(self):
        # {{{
        self.nrprefix = None
        if self.datadir is not None:
            tmp1 = self.datadir.strip('/')
        elif self.outdir is not None:
            tmp1 = self.outdir.strip('/')
        prefix = tmp1.split('/')[-1]
        if 'SKS' in prefix or 'CF' in prefix:
            self.nrprefix = prefix
            return prefix
        raise RuntimeError("NR prefix %s inconsistent" % prefix)
        # }}}

    def tar_runs(self, ld, ccefiles, remove=False):
        # {{{
        if type(ccefiles) != list and type(ccefiles) == str:
            ccefiles = [ccefiles]
        elif type(ccefiles) != list and type(ccefiles) != str:
            raise RuntimeError("Please pass a single or list of Ccefiles")
        else:
            for cc in ccefiles:
                self.cce_runs[ld][cc].tar_output(remove=remove)
        return
        # }}}

    def tar_all_runs_at_lev(self, ld, remove=False):
        # {{{
        self.tar_runs(ld, self.ccefiles[ld], remove=remove)
        return
        # }}}

    def setup_runs(self, ld, ccefiles, submit=False):
        # {{{
        if type(ccefiles) != list and type(ccefiles) == str:
            ccefiles = [ccefiles]
        elif type(ccefiles) != list and type(ccefiles) != str:
            raise RuntimeError("Please pass a single or list of Ccefiles")
        else:
            for cc in ccefiles:
                self.cce_runs[ld][cc].setup_run(submit=submit)
        return
        # }}}

    def setup_all_runs_at_lev(self, ld, submit=False):
        # {{{
        self.setup_runs(ld, self.ccefiles[ld], submit=submit)
        return
        # }}}

    def setup_highestR_run_at_lev(self, ld, num_runs=1, submit=False):
        # {{{
        num_runs *= -1
        self.setup_runs(ld,
                        list([np.sort(self.ccefiles[ld])[num_runs]]),
                        submit=submit)
        return
        # }}}

    def setup_highestR_runs_at_lev(self, ld, num_runs=1, submit=False):
        # {{{
        num_runs *= -1
        self.setup_runs(ld,
                        list(np.sort(self.ccefiles[ld])[num_runs:]),
                        submit=submit)
        return
        # }}}

    def submit_runs(self, ld, ccefiles):
        # {{{
        if type(ccefiles) != list and type(ccefiles) == str:
            ccefiles = [ccefiles]
        elif type(ccefiles) != list and type(ccefiles) == str:
            raise RuntimeError("Please pass a single or list of Ccefiles")
        else:
            for cc in ccefiles:
                self.cce_runs[ld][cc].submit_run()
        return
        # }}}

    def submit_all_runs_at_lev(self, ld):
        # {{{
        self.submit_runs(ld, self.ccefiles[ld])
        return
        # }}}

    def submit_highestR_runs_at_lev(self, ld, num_runs=1):
        # {{{
        num_runs *= -1
        self.submit_runs(ld, list(np.sort(self.ccefiles[ld])[num_runs:]))
        #self.submit_runs( ld, self.ccefiles[ld] )
        return
        # }}}

    def combine_output_for_runs(self, ld, ccefiles, redo=False):
        # {{{
        if type(ccefiles) != list and type(ccefiles) == str:
            ccefiles = [ccefiles]
        elif type(ccefiles) != list and type(ccefiles) == str:
            raise RuntimeError("Please pass a single or list of Ccefiles")
        else:
            for cc in ccefiles:
                self.cce_runs[ld][cc].combine_output(redo=redo)
        return
        # }}}

    def combine_output_highestR_runs_at_lev(self, ld, num_runs=1, redo=False):
        # {{{
        num_runs *= -1
        self.combine_output_for_runs(
            ld, list(np.sort(self.ccefiles[ld])[num_runs:]), redo=redo)
        return
        # }}}

    #
    # Analysis Functios below
    #

    def get_psd(self, low_frequency_cutoff=12.):
        # {{{
        sample_rate, time_length = self.sample_rate, self.time_length
        N = sample_rate * time_length
        try:
            if len(self.psd) != N / 2 + \
                    1 or abs(self.psd.delta_f - 1. / time_length) > 1.e-12:
                raise RuntimeError(
                    "self.psd not in agreement with specified N / df")
        except BaseException:
            self.psd = from_txt(
                '/home/p/pfeiffer/prayush/advLIGO_PSDs/ZERO_DET_high_P.dat',
                N / 2 + 1,
                1. / time_length,
                low_freq_cutoff=low_frequency_cutoff)
        return self.psd
        # }}}

    #

    def read_waveforms_from_hdf5_files(
            self, wavefilename='rhOverM_CcePITT_Asymptotic_GeometricUnits.h5'):
        # This assumes there is a file for EACH Lev with the same name, located in
        # outdir/LevN
        # {{{
        self.wavefiles, self.hwaveforms = {}, {}
        for ld in self.levs:
            print("Reading from ",
                  self.outdir + '/' + ld + '/' + wavefilename,
                  file=sys.stdout)
            sys.stdout.flush()
            self.wavefiles[ld] = h5py.File(
                self.outdir + '/' + ld + '/' + wavefilename, 'r')
            self.hwaveforms[ld] = {}
            for kk in list(self.wavefiles[ld].keys()):
                self.hwaveforms[ld][kk] = UseNRinDA.nr_waveform(
                    filename=self.wavefiles[ld][kk]['Y_l2_m2.dat'].value,
                    filetype='dataset',
                    sample_rate=self.sample_rate,
                    time_length=self.time_length)
        return
        # }}}

    #

    def read_extrapolated_waveforms_from_hdf5_files(
            self,
            dirname=None,
            wavefilename='rhOverM_CcePITT_Asymptotic_GeometricUnits.h5'):
        # This assumes there is a file for EACH Lev with the same name, located in
        # outdir/LevN
        # {{{
        if dirname is None or not os.path.exists(dirname):
            raise IOError(
                "Directory %s supposed to contain the extrapolated waveforms doesn't exist"
                % dirname)
        self.extrap_wavefiles, self.extrap_hwaveforms = {}, {}
        for ld in self.levs:
            self.extrap_wavefiles[ld] = h5py.File(
                dirname + '/' + ld + '/' + wavefilename, 'r')
            self.extrap_hwaveforms[ld] = {}
            for kk in list(self.extrap_wavefiles[ld].keys()):
                self.extrap_hwaveforms[ld][kk] = UseNRinDA.nr_waveform(
                    filename=self.extrap_wavefiles[ld][kk]
                    ['Y_l2_m2.dat'].value,
                    filetype='dataset',
                    sample_rate=self.sample_rate,
                    time_length=self.time_length)
        return
        # }}}

    #

    def calculate_mismatch_between_levs_hdf5(
            self,
            wavefilename='rhOverM_CcePITT_Asymptotic_GeometricUnits.h5',
            outdir='matches',
            outputfile='OverlapsLevs.h5',
            catalogfile=None,
            m_upper=100.,
            m_delta=5.):
        # {{{
        cmd.getoutput('mkdir -p %s/%s' % (self.outdir, outdir))
        fout = h5py.File(self.outdir + '/' + outdir + '/' + outputfile, "a")
        #
        # Get the waveforms for different levs
        self.read_waveforms_from_hdf5_files(wavefilename=wavefilename)
        # Get PSD
        sample_rate, time_length = self.sample_rate, self.time_length
        N = sample_rate * time_length
        self.psd = self.get_psd()
        #
        ccefiles = list(self.wavefiles[self.levs[0]].keys())
        #ccefiles = list(np.sort( self.ccefiles[self.levs[0]] )[num_runs:])
        # Obtain the waveform files for given CceR, at Lev3,4,5
        # In pairs, compare Lev3,4,5
        self.levs.sort()
        for ccef in ccefiles:
            # choose a pair of levs
            for i1 in range(len(self.levs)):
                ld1 = self.levs[i1]
                for i2 in range(i1, len(self.levs)):  # Include self overlaps
                    ld2 = self.levs[i2]
                    if ccef not in list(self.hwaveforms[ld1].keys()) or \
                            ccef not in list(self.hwaveforms[ld2].keys()):
                        print(ccef, " waveforms not found in both %s and %s" %
                              (ld1, ld2))
                        continue
                    # Create a group in output file for this ccefile
                    if ccef not in list(fout.keys()):
                        fout.create_group(ccef)
                    # Compute matches
                    if self.verbose:
                        print("\n\nOverlaps for %s Between %s and %s" %
                              (ccef, ld1, ld2),
                              file=sys.stderr)
                    overlaps = overlaps_vs_totalmass(
                        self.hwaveforms[ld1][ccef],
                        self.hwaveforms[ld2][ccef],
                        psd=self.psd,
                        m_upper=m_upper,
                        m_delta=m_delta)
                    # Add matches and masses as a dataset to the group
                    dsetname = ld1 + '_' + ld2 + '.dat'
                    fout[ccef].create_dataset(dsetname, data=overlaps)
        #
        fout.flush()
        fout.close()
        return
        # }}}

    #

    def calculate_mismatch_between_extraction_radii_hdf5(
            self,
            wavefilename='rhOverM_CcePITT_Asymptotic_GeometricUnits.h5',
            outdir='matches',
            outputfile='OverlapsExtractionRadii.h5',
            catalogfile=None,
            m_upper=100.,
            m_delta=5.):
        # {{{
        cmd.getoutput('mkdir -p %s/%s' % (self.outdir, outdir))
        fout = h5py.File(self.outdir + '/' + outdir + '/' + outputfile, "a")
        #
        # Get the waveforms for different levs
        self.read_waveforms_from_hdf5_files(wavefilename=wavefilename)
        # Get PSD
        sample_rate, time_length = self.sample_rate, self.time_length
        N = sample_rate * time_length
        self.psd = self.get_psd()
        #
        ccefiles = list(self.wavefiles[self.levs[0]].keys())
        #ccefiles = list(np.sort( self.ccefiles[self.levs[0]] )[num_runs:])
        # Obtain the waveform files for given CceR, at Lev3,4,5
        # In pairs, compare Lev3,4,5
        self.levs.sort()
        for ld in self.levs:
            # choose a pair of levs
            for i1 in range(len(ccefiles)):
                ccef1 = ccefiles[i1]
                for i2 in range(i1, len(ccefiles)):  # Include self overlaps
                    ccef2 = ccefiles[i2]
                    if ccef1 not in list(self.hwaveforms[ld].keys()) or \
                            ccef2 not in list(self.hwaveforms[ld].keys()):
                        print("%s and %s waveforms not found in %s" %
                              (ccef1, ccef2, ld))
                        continue
                    # Create a group in output file for this ccefile
                    if ld + '.dir' not in list(fout.keys()):
                        fout.create_group(ld + '.dir')
                    # Compute matches
                    if self.verbose:
                        print("Overlaps for %s Between %s and %s" %
                              (ld, ccef1, ccef2),
                              file=sys.stderr)
                    overlaps = overlaps_vs_totalmass(
                        self.hwaveforms[ld][ccef1],
                        self.hwaveforms[ld][ccef2],
                        psd=self.psd,
                        m_upper=m_upper,
                        m_delta=m_delta)
                    # Add matches and masses as a dataset to the group
                    dsetname = ccef1 + '_' + ccef2 + '.dat'
                    if self.verbose:
                        print("Creating dataset ", dsetname)
                    if self.verbose:
                        print("keys: ", list(fout.keys()), ld,
                              list(fout[ld + '.dir'].keys()))
                    # if self.verbose: print "overlaps", overlaps,
                    # np.shape(overlaps)
                    fout[ld + '.dir'].create_dataset(dsetname, data=overlaps)
        #
        fout.flush()
        fout.close()
        return
        # }}}

    #

    def plot_mismatches_from_hdf5(self,
                                  matchfilename='matches/OverlapsLevs.h5',
                                  outdir='plots',
                                  prefix=None):
        # {{{
        cmd.getoutput('mkdir -p %s/%s' % (self.outdir, outdir))
        #
        # Reading mismatches
        fin = h5py.File(self.outdir + '/' + matchfilename, 'r')
        constt_keys = list(fin.keys())
        var_keys, overlaps, mismatches, masses = {}, {}, {}, {}
        for kk in constt_keys:
            var_keys[kk] = list(fin[kk].keys())
            overlaps[kk], mismatches[kk], masses[kk] = {}, {}, {}
            for ll in var_keys[kk]:
                overlaps[kk][ll] = fin[kk][ll].value
                masses[kk][ll] = overlaps[kk][ll][:, 0]
                mismatches[kk][ll] = []
                for i in range(1, np.shape(overlaps[kk][ll])[1]):
                    mismatches[kk][ll].append(1. - overlaps[kk][ll][:, i])
        #
        # Plotting mismatches
        markers = ["o", "s", "^", "v", "*", "x"]
        line = ["-", "--", ":", "-."]  # comparisons (levs or N)
        markers.extend(markers)  # cyclic
        line.extend(line)  # cyclic
        colors = [
            "blue", "red", "green", "magenta", "cyan", "gold", "darkorange"
        ]  # blending options
        l_blend = ["A", "B", "C", "D", "E"]
        l_compare = []
        plot_lines = []
        #
        for kk in range(len(constt_keys)):
            const_key = constt_keys[kk]
            if prefix is None:
                plotname = self.outdir + '/' + outdir + '/' + const_key + '.png'
            else:
                plotname = self.outdir + '/' + outdir + '/' + prefix + const_key + '.png'
            fig, ax = plt.subplots(1)
            fig.set_size_inches(16, 12)
            mismatch = mismatches[const_key]
            mass = masses[const_key]
            #
            plot_lines = [[], [], [], [], [], []]
            print(mismatch[list(mismatch.keys())[0]], list(mismatch.keys()))
            l_compare = []
            for i in range(len(mismatch[list(mismatch.keys())[0]])):
                for n in range(len(list(mismatch.keys()))):
                    print("i = %d/%d, n = %d/%d" %
                          (i, len(mismatch[list(mismatch.keys())[0]]), n,
                           len(list(mismatch.keys()))))
                    var_key = var_keys[const_key][n]
                    #strs = var_key.split('_')
                    #strs[-1] = strs[-1].strip('.dat')
                    # print strs
                    # if strs[0] == strs[1]: continue
                    l_compare.append(var_key)
                    pl, = plt.plot(mass[var_key],
                                   mismatch[var_key][i],
                                   linestyle=line[n],
                                   marker=markers[n],
                                   color=colors[i])
                    plot_lines[i].append(pl, )
            #
            #l_compare = var_keys[const_key]
            plt.yscale('log')
            plt.xlabel('mass (solar mass)')
            plt.ylabel('mismatch (1-overlap)')
            plt.grid(b=True, which='major')
            legend_1 = plt.legend(plot_lines[0], l_compare, loc="lower left")
            if 'Lev' in var_keys[const_key][0]:
                legend_2 = plt.legend(list(zip(*plot_lines))[0],
                                      l_blend,
                                      ncol=2,
                                      loc="lower right")
                plt.title("Mismatch with blended waveforms, at Fixed " +
                          const_key + "\n" + self.get_nrprefix())
            if 'Cce' in var_keys[const_key][0]:
                legend_2 = plt.legend(list(zip(*plot_lines))[0],
                                      l_blend,
                                      ncol=2,
                                      loc="lower right")
                plt.title("Mismatch with blended waveforms, at Fixed " +
                          const_key + "\n" + self.get_nrprefix())
            plt.legend()
            plt.gca().add_artist(legend_1)
            plt.gca().add_artist(legend_2)
            plt.ylim([1e-7, 0.1])
            plt.savefig(plotname)
            print("plot saved to %s." % plotname)
        #
        return
        # }}}

    #

    def calculate_mismatch_with_extrapolated_hdf5(
            self,
            wavefilename='rhOverM_CcePITT_Asymptotic_GeometricUnits.h5',
            extrap_dirname='/home/p/pfeiffer/pfeiffer/SimulationAnnex/Incoming/ChuAlignedRuns/',
            extrap_runname=None,
            extrap_filename='rhOverM_Asymptotic_GeometricUnits.h5',
            outdir='matches',
            outputfile='OverlapsExtrapolated.h5',
            catalogfile=None,
            m_upper=100.,
            m_delta=5.):
        # {{{
        cmd.getoutput('mkdir -p %s/%s' % (self.outdir, outdir))
        fout = h5py.File(self.outdir + '/' + outdir + '/' + outputfile, "a")
        #
        # Get the Cce waveforms for different levs
        print("Reading Cce waveforms")
        try:
            tmp_len = len(self.hwaveforms)
        except BaseException:
            self.read_waveforms_from_hdf5_files(wavefilename=wavefilename)
        # Get the Extrapolated waveforms for different Extrapolation ORders
        print("Reading Extrapolated waveforms")
        try:
            tmp_len = len(self.extrap_hwaveforms)
        except BaseException:
            if extrap_runname is None:
                extrap_runname = extrapolated_outdir_from_cce_outdir(
                    self.outdir)
            self.read_extrapolated_waveforms_from_hdf5_files(
                dirname=extrap_dirname + '/' + extrap_runname,
                wavefilename=extrap_filename)
        # Get PSD
        self.get_psd()
        #
        # Select the levs, Cce extraction radii
        self.levs.sort()
        #Levs = ['Lev5']
        Levs = self.levs
        #CceRindices = [-1]
        CceRindices = np.arange(len(list(self.wavefiles[Levs[0]].keys())))
        #ExtrapOrders = ['Extrapolated_N2.dir','Extrapolated_N3.dir','OutermostExtraction.dir']
        ExtrapOrders = list(self.extrap_hwaveforms[Levs[0]].keys())
        all_ccefiles = list(self.wavefiles[self.levs[0]].keys())
        all_ccefiles.sort()
        ccefiles = [all_ccefiles[idx] for idx in CceRindices]
        #
        print(Levs, "\n", ccefiles, "\n", CceRindices, "\n", ExtrapOrders)
        # At each Lev, compare waveforms at all cce radii with extrapolation
        # orders
        for lev in Levs:
            print("At ", lev)
            if lev not in list(self.hwaveforms.keys()) or lev not in list(
                    self.extrap_hwaveforms.keys()):
                print(
                    "No waveforms at %s in either the Cce or Extrapolated set"
                    % lev)
                continue
            if lev + '.dir' not in list(fout.keys()):
                fout.create_group(lev + '.dir')
            for ccef in ccefiles:
                print("For ", ccef)
                if ccef not in list(self.hwaveforms[lev].keys()):
                    print(ccef, " waveform not found for %s" % lev)
                hp0 = self.hwaveforms[lev][ccef]
                for extrap_order in ExtrapOrders:
                    # Safegaurd for N-1 orders. New convention is
                    # OutermostExtrapolation
                    if 'N-1' in extrap_order:
                        print("Skipping ", extrap_order, " at ", lev)
                        continue
                    #
                    print("At ", extrap_order)
                    if extrap_order not in list(
                            self.extrap_hwaveforms[lev].keys()):
                        print("%s waveform not found at %s" %
                              (extrap_order, lev))
                    hp1 = self.extrap_hwaveforms[lev][extrap_order]
                    overlaps = overlaps_vs_totalmass(hp0,
                                                     hp1,
                                                     psd=self.psd,
                                                     m_upper=m_upper,
                                                     m_delta=m_delta)
                    # Add matches and masses as a dataset to the group
                    dsetname = ccef + '_' + extrap_order + '.dat'
                    fout[lev + '.dir'].create_dataset(dsetname, data=overlaps)
        #
        fout.flush()
        fout.close()
        return
        # }}}

    # }}}


def extrapolated_outdir_from_cce_outdir(outdir):
    #
    # Accept SKS_d19.8-q1-sA_0_0_-0.8_sB_0_0_-0.8
    # Return BBH_SKS_d19.8_q1_sA_0_0_-0.800_sB_0_0_-0.800
    #
    # {{{
    outdir = outdir.strip('/').split('/')[-1]
    try:
        idtype, dq, s1x, s1y, s1z, _, s2x, s2y, s2z = outdir.split('_')
    except ValueError:
        if outdir[0] == 'd':
            outdir = 'CF_' + outdir
            idtype, dq, s1x, s1y, s1z, _, s2x, s2y, s2z = outdir.split('_')
        else:
            raise ValueError('Cannot translate dir name to extrapolated dir')
    if idtype == 'CF':
        idtype += 'MS'
    d, q, _ = dq.split('-')
    print(q)
    if '.' in q:
        q = 'q%.2f' % np.float64(q[1:])
    if np.float64(d[1:]) == np.round(np.float64(d[1:])):
        d = 'd' + str(int(np.float64(d[1:])))
    print(s1z, s2z)
    if np.float(s1z) == 0.:
        s1z = '0'
    else:
        s1z = '%.3f' % np.float128(s1z)
    if np.float(s2z) == 0.:
        s2z = '0'
    else:
        s2z = '%.3f' % np.float128(s2z)
    retdir = 'BBH_%s_%s_%s_sA_%s_%s_%s_sB_%s_%s_%s' % (idtype, d, q, s1x, s1y,
                                                       s1z, s2x, s2y, s2z)
    return retdir
    # }}}


def initial_frequency_from_metadata(id_string, lev=None, xml_table=None):
    # {{{
    if xml_table is None:
        raise IOError("Please provide the catalog table")
    if lev is None:
        raise IOError("What Lev is the waveform..?")
    for line in xml_table:
        if id_string in line.waveform and lev in line.waveform:
            return line.f_lower
    raise IOError("Waveform not found in the catalog..! Lev missing?")
    # }}}
