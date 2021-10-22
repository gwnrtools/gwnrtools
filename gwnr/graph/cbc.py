# Copyright (C) 2020 Prayush Kumar
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
from __future__ import print_function


def ParamLatexLabels():
    return {
        'mass_1': r'$m_1\,(\mathrm{M}_\odot)$',
        'mass_2': r'$m_2\,(\mathrm{M}_\odot)$',
        'mass1': r'$m_1\,(\mathrm{M}_\odot)$',
        'mass2': r'$m_2\,(\mathrm{M}_\odot)$',
        'm1': r'$m_1\,(\mathrm{M}_\odot)$',
        'm2': r'$m_2\,(\mathrm{M}_\odot)$',
        'mchirp': r'$\mathcal{M}\,(\mathrm{M}_\odot)$',
        'chirp_mass': r'$\mathcal{M}\,(\mathrm{M}_\odot)$',
        'mc': r'$\mathcal{M}\,(\mathrm{M}_\odot)$',
        'eta': r'$\eta$',
        'q': r'$q$',
        'mass_ratio': r'$q$',
        'total_mass': r'$M_\mathrm{total}\,(\mathrm{M}_\odot)$',
        'mtotal': r'$M_\mathrm{total}\,(\mathrm{M}_\odot)$',
        'm1_source': r'$m_{1}^\mathrm{source}\,(\mathrm{M}_\odot)$',
        'm2_source': r'$m_{2}^\mathrm{source}\,(\mathrm{M}_\odot)$',
        'mtotal_source':
        r'$M_\mathrm{total}^\mathrm{source}\,(\mathrm{M}_\odot)$',
        'mc_source': r'$\mathcal{M}^\mathrm{source}\,(\mathrm{M}_\odot)$',
        'redshift': r'$z$',
        'mf': r'$M_\mathrm{final}\,(\mathrm{M}_\odot)$',
        'mf_source': r'$M_\mathrm{final}^\mathrm{source}\,(\mathrm{M}_\odot)$',
        'af': r'$a_\mathrm{final}$',
        'e_rad': r'$E_\mathrm{rad}\,(\mathrm{M}_\odot)$',
        'l_peak':
        r'$L_\mathrm{peak}\,(10^{56}\,\mathrm{ergs}\,\mathrm{s}^{-1})$',
        'spin1': r'$S_1$',
        'spin2': r'$S_2$',
        'a1': r'$a_1$',
        'a2': r'$a_2$',
        'a1z': r'$a_{1z}$',
        'a2z': r'$a_{2z}$',
        's1z': r'$\chi_{1z}$',
        's2z': r'$\chi_{2z}$',
        'theta1': r'$\theta_1\,(\mathrm{rad})$',
        'theta2': r'$\theta_2\,(\mathrm{rad})$',
        'phi1': r'$\phi_1\,(\mathrm{rad})$',
        'phi2': r'$\phi_2\,(\mathrm{rad})$',
        'chi_eff': r'$\chi_\mathrm{eff}$',
        'chi_tot': r'$\chi_\mathrm{total}$',
        'chi_p': r'$\chi_\mathrm{P}$',
        'eccentricity': r'$e_0$',
        'e0': r'$e_0$',
        'tilt1': r'$t_1\,(\mathrm{rad})$',
        'tilt2': r'$t_2\,(\mathrm{rad})$',
        'costilt1': r'$\mathrm{cos}(t_1)$',
        'costilt2': r'$\mathrm{cos}(t_2)$',
        'iota': r'$\iota\,(\mathrm{rad})$',
        'cosiota': r'$\mathrm{cos}(\iota)$',
        'time': r'$t_\mathrm{c}\,(\mathrm{s})$',
        'time_mean': r'$<t>\,(\mathrm{s})$',
        'dist': r'$d_\mathrm{L}\,(\mathrm{Mpc})$',
        'ra': r'$\alpha$',
        'dec': r'$\delta$',
        'phase': r'$\phi\,(\mathrm{rad})$',
        'psi': r'$\psi\,(\mathrm{rad})$',
        'theta_jn': r'$\theta_\mathrm{JN}\,(\mathrm{rad})$',
        'costheta_jn': r'$\mathrm{cos}(\theta_\mathrm{JN})$',
        'beta': r'$\beta\,(\mathrm{rad})$',
        'cosbeta': r'$\mathrm{cos}(\beta)$',
        'phi_jl': r'$\phi_\mathrm{JL}\,(\mathrm{rad})$',
        'phi12': r'$\phi_\mathrm{12}\,(\mathrm{rad})$',
        'logl': r'$\mathrm{log}(\mathcal{L})$',
        'h1_end_time': r'$t_\mathrm{H}$',
        'l1_end_time': r'$t_\mathrm{L}$',
        'v1_end_time': r'$t_\mathrm{V}$',
        'h1l1_delay': r'$\Delta t_\mathrm{HL}$',
        'h1v1_delay': r'$\Delta t_\mathrm{HV}$',
        'l1v1_delay': r'$\Delta t_\mathrm{LV}$',
        'lambdat': r'$\tilde{\Lambda}$',
        'dlambdat': r'$\delta \tilde{\Lambda}$',
        'lambda1': r'$\lambda_1$',
        'lambda2': r'$\lambda_2$',
        'lam_tilde': r'$\tilde{\Lambda}$',
        'dlam_tilde': r'$\delta \tilde{\Lambda}$',
        'calamp_h1': r'$\delta A_{H1}$',
        'calamp_l1': r'$\delta A_{L1}$',
        'calpha_h1': r'$\delta \phi_{H1}$',
        'calpha_l1': r'$\delta \phi_{L1}$',
        'polar_eccentricity': r'$\epsilon_{polar}$',
        'polar_angle': r'$\alpha_{polar}$',
        'alpha': r'$\alpha_{polar}$',
        'dchi0': r'$d\chi_0$',
        'dchi1': r'$d\chi_1$',
        'dchi2': r'$d\chi_2$',
        'dchi3': r'$d\chi_3$',
        'dchi4': r'$d\chi_4$',
        'dchi5': r'$d\chi_5$',
        'dchi5l': r'$d\chi_{5}^{(l)}$',
        'dchi6': r'$d\chi_6$',
        'dchi6l': r'$d\chi_{6}^{(l)}$',
        'dchi7': r'$d\chi_7$',
        'dxi1': r'$d\xi_1$',
        'dxi2': r'$d\xi_2$',
        'dxi3': r'$d\xi_3$',
        'dxi4': r'$d\xi_4$',
        'dxi5': r'$d\xi_5$',
        'dxi6': r'$d\xi_6$',
        'dalpha1': r'$d\alpha_1$',
        'dalpha2': r'$d\alpha_2$',
        'dalpha3': r'$d\alpha_3$',
        'dalpha4': r'$d\alpha_4$',
        'dalpha5': r'$d\alpha_5$',
        'dbeta1': r'$d\beta_1$',
        'dbeta2': r'$d\beta_2$',
        'dbeta3': r'$d\beta_3$',
        'dsigma1': r'$d\sigma_1$',
        'dsigma2': r'$d\sigma_2$',
        'dsigma3': r'$d\sigma_3$',
        'dsigma4': r'$d\sigma_4$',
    }
