#!/usr/bin/env python
import subprocess as cmd
import h5py
import numpy as np
import sys
import os
import lalsimulation as ls
import lal
from optparse import OptionParser
import romspline
from pycbc.waveform import amplitude_from_polarizations, phase_from_polarizations
from pycbc.types import TimeSeries
from pycbc.pnutils import eta_mass1_to_mass2
from numpy.random import uniform
import time
__itime = time.time()
sys.path.append('/home/prayush/src//')
sys.path.append('/home/prayush/src/romspline/')

__author__ = "Prayush Kumar <prkumar@cita.utoronto.ca>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])

#########################################
#############   OPTIONS #################
#########################################
# {{{
### option parsing ###
parser = OptionParser(usage="""
### INPUTS
# 1 No of waves to generate
# 2 Path of directory to write data to
# 3 SEOBNRv3 version number, 3 = v3, 300 = v3_opt, 304 = v3_opt_rk4
# 4 Tolerance of spline (-1 defaults sensibly)
# 5 Degree of spline polynomial (1 <= k <= 5) (-1 defaults sensibly)
# 6 0/1/2 - 0 for compressing amplitude/phase, 1 for compressing modes, 2 for raw
# 7 Sub-sampling factor "subsamp_n" (defaults to 4)
# 8 Output type: (0) HDF5 or (1) ASCII
""",
                      description="")

parser.add_option('-n',
                  '--num',
                  help='No of waves to generate',
                  type=int,
                  default=4)
parser.add_option('-o',
                  '--output-dir-prefix',
                  help='Path of directory to write data to',
                  type=str,
                  default='.')
parser.add_option(
    '-s',
    '--seobnr-version',
    help='SEOBNRv3 version number, 3 = v3, 300 = v3_opt, 304 = v3_opt_rk4',
    type=int,
    default=3)
parser.add_option('--aligned-spin',
                  action="store_true",
                  help='If specified, spin1x=spin1y=spin2x=spin2y=0',
                  default=False)

parser.add_option('--q-min', help='Min. mass ratio', type=float, default=1.0)
parser.add_option('--q-max', help='Max. mass ratio', type=float, default=10.0)
parser.add_option('--spin1z-min',
                  help='Min. spin1z',
                  type=float,
                  default=-0.99)
parser.add_option('--spin1z-max', help='Max. spin1z', type=float, default=0.99)
parser.add_option('--spin2z-min',
                  help='Min. spin2z',
                  type=float,
                  default=-0.99)
parser.add_option('--spin2z-max', help='Max. spin2z', type=float, default=0.99)
parser.add_option('--spin-mag-min',
                  help='Min. spin magnitude',
                  type=float,
                  default=0.)
parser.add_option('--spin-mag-max',
                  help='Max. spin magnitude',
                  type=float,
                  default=0.99)

parser.add_option('--phi-ref', help='Ref phase [rads]', type=float, default=0)
parser.add_option('--distance',
                  help='Distance to source [Mpc]',
                  type=float,
                  default=1.0e6)

parser.add_option('--sample-rate',
                  help='Sample Rate [Hz]',
                  type=float,
                  default=4096.0)
parser.add_option('--f-lower',
                  help='Low frequency cutoff [Hz]',
                  type=float,
                  default=14.0)

parser.add_option('-t',
                  '--tolerance',
                  help='Tolerance of spline (defaults sensibly)',
                  type=float,
                  default=1.e-4)
parser.add_option('-d',
                  '--degree',
                  help='Tolerance of spline (-1 defaults sensibly)',
                  type=int,
                  default=5)

parser.add_option('-c',
                  '--compression-mode',
                  help="""0/1/2 - 0 for compressing amplitude/phase,
                    1 for compressing modes,
                    2 for raw""",
                  type=int,
                  default=2)
parser.add_option("--write-amp-phase",
                  action="store_true",
                  help="Write amplitude and phase (raw)",
                  default=False)

parser.add_option('-f',
                  '--subsamp-factor',
                  help='Sub-sampling factor "subsamp_n" (defaults to 4)',
                  type=int,
                  default=1)
parser.add_option('--output-type',
                  help='Output type: (0) HDF5 or (1) ASCII',
                  type=int,
                  default=0)

parser.add_option('--num-write-verbose',
                  help='No of waves to generate before prompting the user',
                  type=int,
                  default=1)
parser.add_option("-V",
                  "--verbose",
                  action="store_true",
                  help="print extra debugging information",
                  default=False)

options, argv = parser.parse_args()
print("Restricting to aligned spins..: ", options.aligned_spin)

num_waves = options.num
num_write = 30 + int(uniform(0, 70))  # RANDOM PRESET \in [50, 100]
output_dir = options.output_dir_prefix  # + '_%06d' % int( uniform() * 1e7 )
cmd.getoutput('mkdir -p %s' % output_dir)

precessing_eob_version = options.seobnr_version
sptol = options.tolerance
spdeg = options.degree
if spdeg < 1 or spdeg > 5:
    raise IOError("degree of spline should be [1,5]")

cflag = options.compression_mode
compress_amplitude_phase = False
compress_modes = False
compress_none = False
if cflag == 0:
    compress_amplitude_phase = True
elif cflag == 1:
    compress_modes = True
else:
    compress_none = True

to_amp_phase = to_modes = True
if compress_none:
    to_amp_phase = False
    to_modes = False
elif compress_amplitude_phase:
    to_modes = False
elif compress_modes:
    to_amp_phase = False

write_amp_phase = False
if options.write_amp_phase:
    write_amp_phase = True

subsamp_n = options.subsamp_factor

output_mode = 'BOTH'
oflag = options.output_type
if oflag == 0:
    output_mode = 'HDF'
elif oflag > 0:
    output_mode = 'ASCII'
else:
    pass

num_write_verbose = options.num_write_verbose
# }}}

#########################################
############# FUNCTIONS #################
#########################################


def compress_modes_to_splines(hLM,
                              total_mass,
                              to_amp_phase=True,
                              to_modes=False,
                              tol_one=1e-5,
                              tol_two=1e-5,
                              func_one=np.array,
                              subsamp_fac_one_low=4,
                              subsamp_fac_one_high=2,
                              func_two=np.log,
                              subsamp_fac_two_low=4,
                              subsamp_fac_two_high=1,
                              verbose=False):
    """
Compresses modes provided in hLM according to total mass,
by either first converting to amplitude and phase OR using
the modes directly.
Users can subsample the time-series and apply a function to
them before using romspline compression. These functions can
decrease run time drastically.

[DEFAULTS are sensible for raw modes.]
    """
    # {{{
    #################################
    t_arr = TimeSeries(hLM.tdata.data * total_mass * lal.MTSUN_SI,
                       delta_t=hLM.mode.deltaT)
    h_real = TimeSeries(hLM.mode.data.data.real, delta_t=hLM.mode.deltaT)
    h_imag = TimeSeries(hLM.mode.data.data.imag, delta_t=hLM.mode.deltaT)
    ampLM = amplitude_from_polarizations(h_real, h_imag)
    val, idx = ampLM.abs_max_loc()
    #################################
    if to_amp_phase and not to_modes:
        if verbose:
            print("Converting to amplitude and phase")
        #################################
        if verbose:
            __t0 = time.time()
        phsLM = phase_from_polarizations(h_real, h_imag) + np.pi
        phsLM_min = phsLM.min()
        #################
        t_amp_subsampled = np.append(
            t_arr.data[0:idx - 500:subsamp_fac_one_low],
            t_arr.data[idx - 500:-1:subsamp_fac_one_high])
        amp_subsampled = np.append(
            ampLM.data[0:idx - 500:subsamp_fac_one_low],
            ampLM.data[idx - 500:-1:subsamp_fac_one_high])
        funcamp = func_one(amp_subsampled)
        #
        if verbose:
            print(" took %.2f seconds.." % (time.time() - __t0))
            __t0 = time.time()
            print("RomSpline compressing amplitude")

        funcamp_subsampled_spline = romspline.ReducedOrderSpline(
            t_amp_subsampled, funcamp, tol=tol_one)
        #################
        t_phs_subsampled = np.append(
            t_arr.data[0:idx - 500:subsamp_fac_two_low],
            t_arr.data[idx - 500:-1:subsamp_fac_two_high])
        phs_subsampled = np.append(
            phsLM.data[0:idx - 500:subsamp_fac_two_low],
            phsLM.data[idx - 500:-1:subsamp_fac_two_high])
        funcphs = func_two(phs_subsampled - phsLM_min)
        #
        if verbose:
            print(" took %.2f seconds.." % (time.time() - __t0))
            __t0 = time.time()
            print("RomSpline compressing phase")
        #
        funcphs_subsampled_spline = romspline.ReducedOrderSpline(
            t_phs_subsampled, funcphs, tol=tol_two)
        if verbose:
            print(" took %.2f seconds.." % (time.time() - __t0))
        ################
        return funcamp_subsampled_spline, funcphs_subsampled_spline, phsLM_min,\
            t_arr.min(), t_arr.max()
        #################################
    elif to_modes and not to_amp_phase:
        if verbose:
            print("Using mode real/imaginary parts directly")
        #################################
        if verbose:
            __t0 = time.time()
        t_h_real_subsampled = np.append(
            t_arr.data[0:idx - 500:subsamp_fac_one_low],
            t_arr.data[idx - 500:-1:subsamp_fac_one_high])
        h_real_subsampled = np.append(
            h_real.data[0:idx - 500:subsamp_fac_one_low],
            h_real.data[idx - 500:-1:subsamp_fac_one_high])
        if verbose:
            print(" took %.2f seconds.." % (time.time() - __t0))
            __t0 = time.time()
            print("RomSpline compressing real part of mode")
        func_hreal_subsampled_spline = romspline.ReducedOrderSpline(
            t_h_real_subsampled, h_real_subsampled, tol=tol_one)
        #
        t_h_imag_subsampled = np.append(
            t_arr.data[0:idx - 500:subsamp_fac_two_low],
            t_arr.data[idx - 500:-1:subsamp_fac_two_high])
        h_imag_subsampled = np.append(
            h_imag.data[0:idx - 500:subsamp_fac_two_low],
            h_imag.data[idx - 500:-1:subsamp_fac_two_high])
        if verbose:
            print(" took %.2f seconds.." % (time.time() - __t0))
            __t0 = time.time()
            print("RomSpline compressing imaginary part of mode")
        func_himag_subsampled_spline = romspline.ReducedOrderSpline(
            t_h_imag_subsampled, h_imag_subsampled, tol=tol_two)
        if verbose:
            print(" took %.2f seconds.." % (time.time() - __t0))
        #
        return func_hreal_subsampled_spline, func_himag_subsampled_spline,\
            t_arr.min(), t_arr.max()
        #################################
    elif not to_modes and not to_amp_phase:
        if verbose:
            print("Using mode real/imaginary parts RAW (NOT COMPRESSING)")
        #################################
        if verbose:
            __t0 = time.time()
        if verbose:
            print(" took %.2f seconds.." % (time.time() - __t0))
        #
        return h_real, h_imag, t_arr.min(), t_arr.max()
        #################################
    else:
        IOError(
            "CANNOT compress BOTH of amp/phase OR modes (CAN WRITE RAW though)"
        )
    #################################


# }}}


def write_compressed_modes_to_HDF5(waves):
    """
  Write mode time-series to disk as ASCII. Very specific inputs.
    """
    # {{{
    for file_name, mode_array, splines, tminmax in waves.values():
        # Write data to HDF5 file by passing a group descriptor
        fp = h5py.File(file_name + '.h5', 'w')
        for jdx, mode_lm in enumerate(mode_array):
            el, em = mode_lm
            spline_real, spline_imag = splines[jdx]
            group = fp.create_group('Re_Y_l%d_m%d' % (el, em))
            spline_real.write(group)
            group = fp.create_group('Im_Y_l%d_m%d' % (el, em))
            spline_imag.write(group)
        fp.create_dataset("TimeRangeInSeconds", data=tminmax)
        fp.close()
    return


# }}}


def write_raw_modes_to_HDF5(waves, subsamp_n=1, modes=True):
    """
  Write mode time-series to disk as ASCII. Very specific inputs.
    """
    # {{{
    for file_name, mode_array, splines, tminmax in waves.values():
        # Write data to HDF5 file by passing a group descriptor
        fp = h5py.File(file_name + '.h5', 'w')
        # Header
        header_string = "[1] Time"
        for jdx, mode_lm in enumerate(mode_array):
            el, em = mode_lm
            spline_real, spline_imag = splines[jdx]
            if modes:
                header_string += "\n[%d] Re[h%d%d]" % (2 * jdx + 2, el, em)
                header_string += "\n[%d] Im[h%d%d]" % (2 * jdx + 3, el, em)
            else:
                # If compression was not desired, check if raw amp/phase are to
                # be written instead of the raw modes
                ampLM = amplitude_from_polarizations(spline_real, spline_imag)
                phsLM = phase_from_polarizations(spline_real,
                                                 spline_imag) + np.pi
                spline_real, spline_imag = ampLM, phsLM
                header_string += "\n[%d] Amp[h%d%d]" % (2 * jdx + 2, el, em)
                header_string += "\n[%d] Phase[h%d%d]" % (2 * jdx + 3, el, em)
            #
            if jdx == 0:
                data_array = np.array(
                    list(
                        zip(spline_real.sample_times.data, spline_real.data,
                            spline_imag.data)))
            elif jdx > 0:
                tmp_data_array = np.array(
                    list(zip(spline_real.data, spline_imag.data)))
                data_array = np.append(data_array, tmp_data_array, axis=1)
        # Downsample data
        data_array = data_array[::subsamp_n, :]
        fp.create_dataset("AllModes", data=data_array)
        fp.create_dataset("TimeRangeInSeconds", data=tminmax)
        fp.create_dataset("ModesKey", data=header_string)
        fp.close()
    return


# }}}


def write_raw_modes_to_ASCII(waves, subsamp_n=1, modes=True):
    """
  Write mode time-series to disk as ASCII. Very specific inputs.
    """
    # {{{
    for file_name, mode_array, splines, tminmax in waves.values():
        # Header
        header_string = "[1] Time"
        for jdx, mode_lm in enumerate(mode_array):
            el, em = mode_lm
            spline_real, spline_imag = splines[jdx]
            if modes:
                header_string += "\n[%d] Re[h%d%d]" % (2 * jdx + 2, el, em)
                header_string += "\n[%d] Im[h%d%d]" % (2 * jdx + 3, el, em)
            else:
                # If compression was not desired, check if raw amp/phase are to
                # be written instead of the raw modes
                ampLM = amplitude_from_polarizations(spline_real, spline_imag)
                phsLM = phase_from_polarizations(spline_real,
                                                 spline_imag) + np.pi
                spline_real, spline_imag = ampLM, phsLM
                header_string += "\n[%d] Amp[h%d%d]" % (2 * jdx + 2, el, em)
                header_string += "\n[%d] Phase[h%d%d]" % (2 * jdx + 3, el, em)
            #
            if jdx == 0:
                data_array = np.array(
                    list(
                        zip(spline_real.sample_times.data, spline_real.data,
                            spline_imag.data)))
            elif jdx > 0:
                tmp_data_array = np.array(
                    list(zip(spline_real.data, spline_imag.data)))
                data_array = np.append(data_array, tmp_data_array, axis=1)
        # Downsample data
        data_array = data_array[::subsamp_n, :]
        ##
        # Write to file
        ##
        nrow, ncol = np.shape(data_array)
        file_format = ''
        for j in range(ncol):
            file_format += '%.16e\t'
        # Write
        if not os.path.exists(file_name):
            np.savetxt(file_name + '.txt.gz',
                       data_array,
                       fmt=file_format,
                       header=header_string)
        else:
            print("Warning: FILE NOT WRITTEN FOR ", file_name, header_string)
            continue
    return


# }}}

#########################################
############# MAIN    # #################
#########################################
filter_dt = 1. / options.sample_rate
f_low = options.f_lower

distance = options.distance * lal.PC_SI
phiref = options.phi_ref
inclination = 0 * lal.PI / 6.  # 0 #0.000189136217684

m2_min = 3.0
q_min = options.q_min
q_max = options.q_max
eta_max = q_min / (1. + q_min)**2
eta_min = q_max / (1. + q_max)**2

spin1z_min = options.spin1z_min
spin1z_max = options.spin1z_max
spin2z_min = options.spin2z_min
spin2z_max = options.spin2z_max
spin_mag_min = options.spin_mag_min
spin_mag_max = options.spin_mag_max

##
# GENERATE AND WRITE WAVES
# -> Compress using RomSpline
# -> Either real/imag modes or amplitude/phase
##
m2 = m2_min
waves = {}

for idx in range(num_waves):
    __t1 = time.time()
    # WRITE DATA GENERATED SO FAR
    if idx != 0 and (idx % num_write == 0):
        if compress_none:
            if 'ASCII' in output_mode:
                write_raw_modes_to_ASCII(waves,
                                         subsamp_n=subsamp_n,
                                         modes=write_amp_phase)
            elif 'HDF' in output_mode:
                write_raw_modes_to_HDF5(waves,
                                        subsamp_n=subsamp_n,
                                        modes=write_amp_phase)
            else:
                write_raw_modes_to_ASCII(waves,
                                         subsamp_n=subsamp_n,
                                         modes=write_amp_phase)
                write_raw_modes_to_HDF5(waves,
                                        subsamp_n=subsamp_n,
                                        modes=write_amp_phase)
        else:
            write_compressed_modes_to_HDF5(waves)
        del waves
        waves = {}
    if options.verbose and idx % num_write_verbose == 0:
        print("Generating wave %d" % (idx + 1))
    ##
    # Generate SEOBNRv3 modes
    #
    # Sample masses
    #q   = (uniform() * (q_max - q_min)) + q_min
    eta = (uniform() * (eta_max - eta_min)) + eta_min
    q = m2 / eta_mass1_to_mass2(eta, m2)
    m1 = q * m2
    #
    # Sample spin A
    _t1, _t2, _t3 = uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)
    _abst = (_t1**2 + _t2**2 + _t3**2)**0.5
    _s1m = uniform(spin_mag_min, spin_mag_max)
    s1x = _s1m * (_t1 / _abst)
    s1y = _s1m * (_t2 / _abst)
    s1z = _s1m * (_t3 / _abst)
    if options.aligned_spin:
        s1z = uniform(spin1z_min, spin1z_max)
        s1x = s1y = 0
    # Sample spin B
    _t1, _t2, _t3 = uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)
    _abst = (_t1**2 + _t2**2 + _t3**2)**0.5
    _s2m = uniform(spin_mag_min, spin_mag_max)
    s2x = _s2m * (_t1 / _abst)
    s2y = _s2m * (_t2 / _abst)
    s2z = _s2m * (_t3 / _abst)
    if options.aligned_spin:
        s2z = uniform(spin2z_min, spin2z_max)
        s2x = s2y = 0
    #
    # Generate Waveform
    __t0 = time.time()
    try:
        hplus, hcross, dynHi, hlmPTS, hlmPTSHi, hIMRlmJTSHi, hLM, attachP = \
            ls.SimIMRSpinEOBWaveformAll(phiref, filter_dt,
                                        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, f_low, distance, inclination,
                                        s1x, s1y, s1z, s2x, s2y, s2z, precessing_eob_version)
        f_samp = 1. / filter_dt
    except BaseException:
        hplus, hcross, dynHi, hlmPTS, hlmPTSHi, hIMRlmJTSHi, hLM, attachP = \
            ls.SimIMRSpinEOBWaveformAll(phiref, filter_dt / 2.,
                                        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, f_low, distance, inclination,
                                        s1x, s1y, s1z, s2x, s2y, s2z, precessing_eob_version)
        f_samp = 2. / filter_dt
    if options.verbose:
        print("\t waveform generated in %.2f seconds" % (time.time() - __t0))
    ###
    # Store data to be saved for the modes
    #
    # Name of file
    file_name = 'BBH_f%.1f_f%.1f_m%.6f_m%.6f_sA%.6f_%.6f_%.6f__sB%.6f_%.6f_%.6f' %\
                (f_low, f_samp / subsamp_n, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z)
    file_name = output_dir + '/' + file_name
    orig_hLM = hLM
    #
    # Header
    header_string = "[1] Time"
    time_array = np.array(hLM.tdata.data)
    #
    # COMPRESS ALL modes
    splines = []
    mode_array = []
    jdx = 0
    for j in range(5):
        el, em = hLM.l, hLM.m
        if em <= 0:
            hLM = hLM.__next__
            continue
        if options.verbose and (idx % num_write_verbose) == 0:
            print("\tCompressing (%d,%d) mode.." % (el, em))
        mode_array.append([el, em])
        spline_real, spline_imag, t0, t1 = compress_modes_to_splines(
            hLM,
            m1 + m2,
            to_amp_phase=to_amp_phase,
            to_modes=to_modes,
            func_one=np.array,
            func_two=np.array,
            tol_one=sptol,
            tol_two=sptol,
            subsamp_fac_one_low=subsamp_n,
            subsamp_fac_two_low=subsamp_n,
            subsamp_fac_one_high=1,
            subsamp_fac_two_high=1,
            verbose=True)
        #
        splines.append([spline_real, spline_imag])
        jdx += 1
        hLM = hLM.__next__
    # Collect data
    waves[idx] = [file_name, mode_array, splines, [t0, t1]]
    if options.verbose:
        print("Wave generation+compression done in %.2f seconds" %\
            (time.time() - __t1))

# WRITE THE LAST BATCH
if compress_none:
    if 'ASCII' in output_mode:
        write_raw_modes_to_ASCII(waves,
                                 subsamp_n=subsamp_n,
                                 modes=write_amp_phase)
    elif 'HDF' in output_mode:
        write_raw_modes_to_HDF5(waves,
                                subsamp_n=subsamp_n,
                                modes=write_amp_phase)
    else:
        write_raw_modes_to_HDF5(waves,
                                subsamp_n=subsamp_n,
                                modes=write_amp_phase)
        write_raw_modes_to_ASCII(waves,
                                 subsamp_n=subsamp_n,
                                 modes=write_amp_phase)
else:
    write_compressed_modes_to_HDF5(waves)

print("\n\nAll done in %.3f seconds!" % (time.time() - __itime))
