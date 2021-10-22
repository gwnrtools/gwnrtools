# Copyright (C) 2018 Prayush Kumar
#
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
import numpy as np

import scipy.integrate as integrate
import scipy.interpolate as interpolate
from scipy.optimize import newton

import lal

######################################################
# Constants
######################################################
verbose = True
vverbose = True
debug = True

quantiles_68 = [0.16, 0.5, 0.84]  # 1-sigma
quantiles_95 = [0.0228, 0.5, 0.9772]  # 2-sigma ~ 95.4%
percentiles_68 = 100 * np.array(quantiles_68)
percentiles_95 = 100 * np.array(quantiles_95)
perc_int_95 = [2.28, 97.72]
perc_int_99_7 = [0.13, 99.87]  # 3 sigma

CILevels = [90.0, 68.26895, 95.44997, 99.73002]

verbose = True


######################################################################
######################################################################
#
#     UTILITIES FOR COSMOLOGICAL CALCULATIONS
#
######################################################################
######################################################################
def calculate_redshift(distance, h=0.6790, om=0.3065, ol=0.6935, w0=-1.0):
    """
    Calculate the redshift from the luminosity distance measurement using the
    Cosmology Calculator provided in LAL.
    By default assuming cosmological parameters from arXiv:1502.01589 - 'Planck 2015 results. XIII. Cosmological parameters'
    Using parameters from table 4, column 'TT+lowP+lensing+ext'
    This corresponds to Omega_M = 0.3065, Omega_Lambda = 0.6935, H_0 = 67.90 km s^-1 Mpc^-1
    Returns an array of redshifts
    """
    def find_z_root(z, dl, omega):
        return dl - lal.LuminosityDistance(omega, z)

    omega = lal.CreateCosmologicalParameters(h, om, ol, w0, 0.0, 0.0)
    if isinstance(distance, float):
        z = np.array([
            newton(find_z_root,
                   np.random.uniform(0.0, 2.0),
                   args=(distance, omega))
        ])
    else:
        z = np.array([
            newton(find_z_root, np.random.uniform(0.0, 2.0), args=(d, omega))
            for d in distance
        ])
    return z


def source_to_detector_frame(m, z):
    return m * (1 + z)


def detector_to_source_frame(m, z):
    return m / (1. + z)


def make_z_cosmo_inverseCDF(z_max, R0, H0, Omega_m, Omega_Lambda, Omega_k, w0,
                            w1):
    z_arr = np.linspace(1e-8, z_max, int(1e5))
    dz = z_arr[1] - z_arr[0]
    R_0 = R0 * np.power(1e9, -3) / np.power(1e6, -3)
    R_zmax = integrate.quad(
        dR_dz, 0, z_arr[-1],
        (H0, Omega_m, Omega_Lambda, Omega_k, w0, w1, R_0))[0]

    prob_z = np.zeros_like(z_arr)
    for i in xrange(z_arr.shape[0]):
        prob_z[i] = probability_density_Uniform_comoving_volume(
            z_arr[i], H0, Omega_m, Omega_Lambda, Omega_k, w0, w1, R_0, R_zmax)
    z_CDF = np.cumsum(prob_z * dz)
    z_invCDF = interpolate.interp1d(z_CDF, z_arr)
    return z_invCDF


def z_samples_from_iCDF(iCDF, N):
    r = np.random.rand(N)
    return iCDF(r)


def probability_density_Uniform_comoving_volume(z, H0, Omega_m, Omega_Lambda,
                                                Omega_k, w0, w1, r0, R_zmax):
    # eqn 11 in dP_L_M
    # NSBH rate 3210  Gpc^-3 yr^-1, for aligned spin BHs
    # R_zmax = 3210*np.power(1e9,-3)/np.power(1e6,-3) # Mpc^-3 yr^-1
    return dR_dz(z, H0, Omega_m, Omega_Lambda, Omega_k, w0, w1,
                 r0) * (1 / R_zmax)


def dR_dz(z, H0, Omega_m, Omega_Lambda, Omega_k, w0, w1, r0):
    # eqn 12 in dP_L_M
    e_of_z = 1.  # (z, PARAMETERS) #assume this is constant across the

    return dV_dz(z, H0, Omega_m, Omega_Lambda, Omega_k, w0,
                 w1) * r0 * e_of_z / (1 + z)


def dV_dz(z, H0, Omega_m, Omega_Lambda, Omega_k, w0, w1):
    # eqn 13 in dP_L_M

    return 4*np.pi*np.square(DL(z, H0, Omega_m, Omega_Lambda, Omega_k, w0, w1)) /\
        (np.square(1+z)*H(z, H0, Omega_m, Omega_Lambda, Omega_k, w0, w1)/(lal.C_SI*1e-3))


def H(z, H0, Omega_m, Omega_Lambda, Omega_k, w0, w1):
    return H0 * np.sqrt((Omega_m * np.power(1 + z, 3)) +
                        (Omega_k * np.power(1 + z, 2)) +
                        (Omega_Lambda * E(z, w0, w1)))


def OneOverH(z, H0, Omega_m, Omega_Lambda, Omega_k, w0, w1):
    return 1. / (H0 * np.sqrt((Omega_m * np.power(1 + z, 3)) +
                              (Omega_k * np.power(1 + z, 2)) +
                              (Omega_Lambda * E(z, w0, w1))))


def E(z, w0, w1):
    return np.power(1 + z, 3 * (1 + w0 + w1)) * np.exp(-3 * w1 * z / (1 + z))


def DL(z, H0, Omega_m, Omega_Lambda, Omega_k, w0, w1):
    # (lal.C_SI*1e-3) H0 is gievn in km/s, so multiply c with 1e-3 to correct from m/s to km/s
    if Omega_k > 0.:
        return ((lal.C_SI * 1e-3) * (1 + z) / np.sqrt(Omega_k)) * np.sinh(
            np.sqrt(Omega_k) *
            Hubble_integral(z, H0, Omega_m, Omega_Lambda, Omega_k, w0, w1))
    elif Omega_k == 0.:
        # print 'Omega_k == 0.'
        return (lal.C_SI * 1e-3) * (1 + z) * Hubble_integral(
            z, H0, Omega_m, Omega_Lambda, Omega_k, w0, w1)

    else:
        return (
            (lal.C_SI * 1e-3) *
            (1 + z) / np.sqrt(np.absolute(Omega_k))) * np.sin(
                np.sqrt(np.absolute(Omega_k)) *
                Hubble_integral(z, H0, Omega_m, Omega_Lambda, Omega_k, w0, w1))


def DL_vector(z_arr, H0, Omega_m, Omega_Lambda, Omega_k, w0, w1):
    DL_arr = np.array(
        [DL(z, H0, Omega_m, Omega_Lambda, Omega_k, w0, w1) for z in z_arr])
    return DL_arr


def Hubble_integral(z_prime, H0, Omega_m, Omega_Lambda, Omega_k, w0, w1):
    return integrate.quad(OneOverH, 0, z_prime,
                          (H0, Omega_m, Omega_Lambda, Omega_k, w0, w1))[0]
