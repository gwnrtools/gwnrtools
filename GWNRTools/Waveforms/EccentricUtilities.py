#!/usr/bin/env python
# Copyright (C) 2017 Prayush Kumar
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
########################################
## IMPORTS
########################################
import os, sys
import glob, commands as cmd
import h5py
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import re

os.environ['LD_LIBRARY_PATH'] =\
    '/home/prayush/research/Eccentric_IMRGPR/Code/MergerRingdownModel/C_implementation/bin/'

########################################
## Other FUNCTIONS
########################################

def get_q_m_e_pn_o_from_filename(filename):
    ## qstr, Mstr, wstr, pnstr = [s.split('-')[-1] for s in filename.split('/')[-1].split('_')[-5:-1] ]
    #wstr, pnstr, _, m1str, m2str, estr = [s.split('-')[-1] for s in filename.split('/')[-1].split('_')]
    _, m1str, m2str, estr = [s for s in filename.split('/')[-1].split('_')]
    #
    m1 = float(m1str[3:])
    m2 = float(m2str[3:])
    M  = m1 + m2
    q  = m1 / m2
    # q = float(qstr)
    # M = float(Mstr)
    PNO = -1#int(pnstr)
    omega = -1#float(wstr.strip('data'))
    e0 = float(estr[4:])
    return q, M, e0, PNO, omega

def get_q_m_e_from_filename(filename):
    # FIXME
    if 'good' in filename or 'bad' in filename:
      filename = filename.split('/')[-1]
      print "filename received: %s" % filename
      filename = filename.strip('bad_14Hz_').strip('good_14Hz_')
      print "filename trimmed: %s" % filename
      m1str, m2str, e0str = [s.split('-')[-1] for s in filename.split('/')[-1].split('_')]
      m1 = float(m1str)
      m2 = float(m2str)
      e0 = float(e0str)
      q  = m1 / m2
      M  = m1 + m2
    else:
      q, M, e0, PNO, omega = get_q_m_e_pn_o_from_filename(filename)
    return (q, M, e0)


########################################
## USER-FACING FUNCTIONS
########################################
def get_eccentric_waveform_and_dynamics(\
                  m1_min, m1_max, m1_nbins, m2_min, m2_max, m2_nbins,
                  e_min, e_max, e_nbins, f_lower, delta_t,
                  EXE='export LD_LIBRARY_PATH=/home/prayush/src/EccIMR/code/MergerRingdownModel/C_implementation/bin/:${LD_LIBRARY_PATH} && /home/prayush/src/EccIMR/code/map_link_codes/bbhall -d',
                  mean_anomaly=0, inclination=0, init_phase=0, tolerance=1e-12,
                  verbose=False):
    """
This function computes an eccentric inspiral, and returns both the coordinate
trajectory information as well as the GW polarizations.
    """
    #{{{
    if m1_nbins != 1 or m2_nbins != 1 or e_nbins != 1:
        raise IOError("Function does not support generating multiple waveforms at present..")
    cmd_string = "%s -m %.18e -M %.18e -x %d -n %.18e -N %.18e -y %d -e %.18e -E %.18e -z %d" %\
                    (EXE, m1_min, m1_max, m1_nbins, m2_min, m2_max, m2_nbins, e_min, e_max, e_nbins)
    cmd_string += " -a %.18e -i %.18e -b %.18e -t %.18e -f %.18e -s %.18e" %\
                    (mean_anomaly, inclination, init_phase, tolerance, f_lower, 1./delta_t)
    output_file_tag = 'tmp_%06d' % int(np.random.random() * 1e7)
    cmd_string += " -o %s -v" % output_file_tag
    if verbose:
        print >> sys.stdout, "Command being run: %s" % cmd_string
        sys.stdout.flush()
    cmd_output = cmd.getoutput(cmd_string)
    if verbose:
        print >> sys.stdout, cmd_output
        sys.stdout.flush()
    ##
    dynamics_filename = sorted(glob.glob(output_file_tag + "*" + "Dynamics*"))[-1]
    if verbose:
        print >> sys.stdout, "Reading dynamics from: %s" % dynamics_filename
        sys.stdout.flush()
    ##
    dynamics_headers, _= ParseHeaderForSpECTabularOutputASCII(dynamics_filename,
                                    separate_quantities_from_subdomains=False)
    dynamics_data = np.loadtxt(dynamics_filename)
    dynamics = {}
    for idx in dynamics_headers:
        label = dynamics_headers[idx].split()[0]
        if verbose: print "reading %s from col %d" % (label, idx)
        dynamics[label] = dynamics_data[:,idx]
    ##
    wave_filename = sorted(glob.glob(output_file_tag + "*"))[0]
    if verbose:
        print >> sys.stdout, "Reading waveform from: %s" % wave_filename
        sys.stdout.flush()
    ##
    wave_data = np.loadtxt(wave_filename)
    if verbose:
        print >> sys.stdout, "Reading polarization timeseries from OBJ of shape: ", np.shape(wave_data)
    dynamics['h_Time'] = wave_data[:,0]
    dynamics['hp'] = wave_data[:,1]
    dynamics['hc'] = wave_data[:,2]
    ##
    cmd_string = 'rm -f %s %s' % (wave_filename, dynamics_filename)
    cmd.getoutput(cmd_string)
    return dynamics
    #}}}

def optimize_eccentricity(x1, y1, q,
                          use_var="r",
                          EXE=None,
                          x_low_lim=None,
                          x_high_lim=None,
                          ecc_low=0,
                          ecc_high=0.1,
                          ecc_init=0,
                          anom_low=0,
                          anom_high=6.28318,
                          anom_init=0,
                          sample_rate = 4096,
                          f_lower = 30.0,
                          method='Nelder-Mead',
                          objective_scaling_fac=None,
                          num_retries=1,
                          verbose=True, debug=False):
    """
Given a trajectory, this function computes the eccentricity and initial
mean anomaly for a binary (at f_low Hz) that optimizes the agreement
between its radial evolution and the trajectory.

[Goal]
To minimize :
    f(x_offset,
        e0,
        anom0) := \int_{x_low_lim}^{x_high_lim} |y2(x + x_offset; e0, anom0) - y1(x)| dx

over {x_offset, e0, anom0). This is done in two steps:-

1) f(x_offset, e0, anom0) is minimized over x_offset using align_curves,
   to get fOpt(x0, anom0)

2) fOpt(e0, anom0) is minimized over {e0, anom0} using scipy.minimize here.


[Notes]
1) [x_low_lim, x_high_lim] are with respect to the (x1, y1) pair.
   The other pair (x1, y2) is the one effectively shifted.

2) Not specifying [x_low_lim, x_high_lim] is equivalent to integrating
   the mean-square difference over the complete (x2) vector.

    """
    #{{{
    if use_var != "r" and use_var != "omega":
        raise RuntimeError("Which variable to use for eccentricity optimization")
    if objective_scaling_fac is None:
        if use_var == "r": objective_scaling_fac = 1.0
        else: objective_scaling_fac = 1e5
    #
    def objective_function_eccentricity(x, *args):
        anom0, e0 = x
        x1, y1, q = args
        lbl = '%.12f,%.12f' % (e0, anom0)
        if lbl not in objective_function_eccentricity.waves.keys():
            try:
                if EXE is None:
                    retval = get_eccentric_waveform_and_dynamics(\
                                20 * q, 20 * q, 1, 20, 20, 1, e0, e0, 1,\
                                f_lower, 1./sample_rate, mean_anomaly=anom0,
                                verbose=debug)
                else:
                    retval = get_eccentric_waveform_and_dynamics(\
                                20 * q, 20 * q, 1, 20, 20, 1, e0, e0, 1,\
                                f_lower, 1./sample_rate, mean_anomaly=anom0, EXE=EXE,
                                verbose=debug)
                objective_function_eccentricity.waves[lbl] = retval
            except:
                return 1e99
        else:
            retval = objective_function_eccentricity.waves[lbl]
        if use_var == "omega": used_var = retval['phidot']
        elif use_var == "r": used_var = retval['r']
        shift, res = align_curves(x1, y1, retval['Time'], used_var,
                                  x_low_lim=x_low_lim,
                                  x_high_lim=x_high_lim,
                                  offset_low_lim=-400,
                                  offset_high_lim=-100)
        if objective_function_eccentricity.counter % 1 == 0:
            print "trying out: e0 = %.12f, anom0 = %.12f, objective = %.12f" %\
                    (e0, anom0, objective_scaling_fac*res.fun)
        objective_function_eccentricity.counter += 1
        return res.fun * objective_scaling_fac
    objective_function_eccentricity.counter = 0
    objective_function_eccentricity.waves = {}
    ###
    ### CALL THE scipy.optimize.minimize TO COMPUTE OPTIMAL ECCENTRICITY & INIT MEAN ANOMALY
    opt_args = (x1, y1, q)

    if debug:
        print "Testing objective function"
        print "Offset 0: ", objective_function_eccentricity([0,0], *opt_args)
        print "Offset 550: ", objective_function_eccentricity([np.pi,0.01], *opt_args)

    for idx in range(num_retries):
        if verbose:
            print >>sys.stdout, "\nTry %d to compute optimal eccentricity" % idx
            sys.stdout.flush()
        ##
        bnds = ((anom_low, anom_high), (ecc_low, ecc_high))
        retval = minimize(objective_function_eccentricity,
                          [anom_init, ecc_init],
                          bounds=bnds,
                          method=method,
                          args=opt_args)
        anom_init, ecc_init = retval.x
    ##
    if verbose:
        print >>sys.stdout,\
            "optimization took %d objective func evals" % objective_function_eccentricity.counter
        sys.stdout.flush()
    ### 7) RETURN OPTIMIZED PARAMETERS
    return [retval.x, retval]
    ###}}}
