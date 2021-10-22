# Copyright (C) 2015 Prayush Kumar
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

from gwnr.nr.analysis.types import (Overlaps, SimulationErrors,
                                    EffectualnessAndBias)
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess
import matplotlib as mp

mp.rc('text', usetex=True)
plt.rcParams.update({'text.usetex': True})

#########################################################
# FAITHFULNESS


class plot_mismatches_sim():
    def __init__(self,
                 simdir=None,
                 matchdirs=['matches'],
                 plotdir='plots',
                 verbose=True,
                 debug=False):
        self.verbose = verbose
        self.debug = debug
        self.simdir = simdir
        self.simtag = self.simdir.strip('/').split('/')[-1]
        self.data = SimulationErrors(simdir=simdir,
                                     matchdirs=matchdirs,
                                     verbose=self.verbose,
                                     debug=self.debug)
        for i in range(len(self.data.ccelevs)):
            self.data.ccelevs[i] = str(self.data.ccelevs[i])
        self.lines = ["-", "--", "-.", "-:"]
        self.markers = ["o", "x", "s", "^", "v", "*", '.', '<']
        self.colors = [
            "blue", "red", "green", "magenta", "cyan", "gold", "black",
            "darkorange"
        ]
        self.taperlabels = ["None", "A", "B", "C", "D", "E"]
        self.plotdir = plotdir + matchdirs[0].lstrip('matches')

    #

    def plot_cce_mismatches_all(self,
                                nsubplotrows=2,
                                nsubplotcols=2,
                                savedir=None,
                                savefig=None):
        # {{{
        if savedir is None:
            savedir = self.plotdir
        # panel1,2,3,4
        # For each tapering: x5
        #
        #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,sharex='col',sharey='row')
        fig = plt.figure(int(1e7 * np.random.random()))
        fig.set_size_inches(16, 12)
        fig.suptitle(self.simtag, fontsize=14)
        self.data.ccelevs.sort()
        self.data.cceradii.sort()
        const_list = [
            self.data.ccelevs, self.data.cceradii, self.data.ccelevs,
            self.data.ccelevs
        ]
        const_func = [
            self.data.ccer, self.data.ccelev, self.data.cceextrapolated,
            self.data.cceextrapolated
        ]
        const_keys = [[], [], self.data.extraporders, self.data.cceradii]
        const_addtn = [[], [], [str(self.data.cceradii[0])],
                       [str(self.data.cceradii[1])]]
        for idx in range(4):
            additional_constraint = const_addtn[idx]
            ax = plt.subplot(nsubplotrows, nsubplotcols, idx + 1)  # <<
            l_taper = self.taperlabels
            l_lev = []
            l_ccer = []
            plot_lines_all = []
            plot_lines_levs = []
            # self.data.cceradii): #<<
            for lidx, lev in enumerate(const_list[idx]):
                lev = str(lev)
                l_lev.append(lev)
                const_key = lev
                # For each tapering: x6
                plot_lines = [[], [], [], [], [], []]
                if idx >= 4:
                    overlaps = {}
                    for l2 in const_keys[idx]:
                        overlaps[l2] = const_func[idx](key=[l2] + [lev],
                                                       noduplicate=True)
                else:
                    overlaps = const_func[idx](key=additional_constraint +
                                               [lev],
                                               noduplicate=True)
                # self.data.ccelev(key=lev,noduplicate=True)#.values()[0] <<
                for i in range(overlaps.values()[0].nWindows):
                    for n in range(len(overlaps.keys())):
                        var_key = str(overlaps.keys()[n])
                        olap = overlaps[var_key]
                        mass = olap.X()
                        mismatch = 1. - olap.Y(i)
                        #
                        if i == 0 and lidx == 0:
                            l_ccer.append(var_key.split('/')[-1])
                        pl, = plt.plot(mass,
                                       mismatch,
                                       linestyle=self.lines[lidx],
                                       lw=1,
                                       marker=self.markers[n],
                                       markersize=3,
                                       color=self.colors[i],
                                       alpha=0.65)
                        plot_lines[i].append(pl, )
                plot_lines_all.append(plot_lines)
                plot_lines_levs.append(plot_lines[0])
            #
            ax.set_yscale('log')
            ax.grid(b=True, which='major')
            if idx + 1 >= nsubplotcols * (nsubplotrows - 1):
                ax.set_xlabel('mass (solar mass)')
            if idx % nsubplotcols == 0:
                ax.set_ylabel('Mismatches')
            legend_1 = plt.legend(zip(*plot_lines_levs)[0],
                                  l_lev,
                                  loc="upper left",
                                  prop={'size': 8},
                                  fancybox=True)
            legend_2 = plt.legend(zip(*plot_lines)[0],
                                  l_taper,
                                  ncol=2,
                                  loc="lower right",
                                  prop={'size': 8},
                                  fancybox=True)
            legend_3 = plt.legend(plot_lines[0],
                                  l_ccer,
                                  ncol=2,
                                  loc='lower left',
                                  prop={'size': 8},
                                  fancybox=True)
            ax.legend()
            ax.add_artist(legend_1)
            ax.add_artist(legend_2)
            ax.add_artist(legend_3)
            ax.set_ylim([1.e-5, 0.01])
        #
        subprocess.getoutput('mkdir -p %s' % self.simdir + '/' + savedir)
        savedir = self.simdir + '/' + savedir
        if savefig is not None:
            plt.savefig(savedir + '/' + savefig, dpi=500)
        else:
            plt.savefig(savedir + '/' + self.simtag + '.png', dpi=500)
        return

    # }}}
    #

    def plot_cce_mismatches_only(self,
                                 nsubplotrows=1,
                                 nsubplotcols=2,
                                 savedir=None,
                                 savefig=None):
        # {{{
        if savedir is None:
            savedir = self.plotdir
        # panel1,2,3,4
        # For each tapering: x5
        #
        #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,sharex='col',sharey='row')
        fig = plt.figure(int(1e7 * np.random.random()))
        fig.set_size_inches(12, 9)
        fig.suptitle(self.simtag, fontsize=14)
        self.data.ccelevs.sort()
        self.data.cceradii.sort()
        const_list = [
            self.data.ccelevs, self.data.cceradii, self.data.ccelevs,
            self.data.ccelevs
        ]
        const_func = [
            self.data.ccer, self.data.ccelev, self.data.cceextrapolated,
            self.data.cceextrapolated
        ]
        const_keys = [[], [], self.data.extraporders, self.data.cceradii]
        const_addtn = [[], [], [str(self.data.cceradii[0])],
                       [str(self.data.cceradii[1])]]
        for idx in range(2):
            additional_constraint = const_addtn[idx]
            ax = plt.subplot(nsubplotrows, nsubplotcols, idx + 1)  # <<
            l_taper = self.taperlabels
            l_lev = []
            l_ccer = []
            plot_lines_all = []
            plot_lines_levs = []
            # self.data.cceradii): #<<
            for lidx, lev in enumerate(const_list[idx]):
                lev = str(lev)
                print("lev = ", lev)
                l_lev.append(lev)
                const_key = lev
                # For each tapering: x6
                plot_lines = [[], [], [], [], [], []]
                if idx >= 4:
                    overlaps = {}
                    for l2 in const_keys[idx]:
                        print("l2, lev = ", l2, lev)
                        overlaps[l2] = const_func[idx](key=[l2] + [lev],
                                                       noduplicate=True)
                else:
                    overlaps = const_func[idx](key=additional_constraint +
                                               [lev],
                                               noduplicate=True)
                print(overlaps)
                # self.data.ccelev(key=lev,noduplicate=True)#.values()[0] <<
                for i in range(overlaps.values()[0].nWindows):
                    for n in range(len(overlaps.keys())):
                        print("i = %d/%d, n = %d/%d" %
                              (i, overlaps.values()[0].nWindows, n,
                               len(overlaps.keys())))
                        var_key = str(overlaps.keys()[n])
                        print("var_key = ", var_key)
                        olap = overlaps[var_key]
                        mass = olap.X()
                        mismatch = 1. - olap.Y(i)
                        #
                        if i == 0 and lidx == 0:
                            l_ccer.append(var_key.split('/')[-1])
                        pl, = plt.plot(mass,
                                       mismatch,
                                       linestyle=self.lines[lidx],
                                       lw=1,
                                       marker=self.markers[n],
                                       markersize=3,
                                       color=self.colors[i],
                                       alpha=0.65)
                        plot_lines[i].append(pl, )
                plot_lines_all.append(plot_lines)
                plot_lines_levs.append(plot_lines[0])
            #
            print("l_taper,lev,ccer = ", l_taper, l_lev, l_ccer)
            #
            ax.set_yscale('log')
            ax.grid(b=True, which='major')
            if idx + 1 >= nsubplotcols * (nsubplotrows - 1):
                ax.set_xlabel('mass (solar mass)')
            if idx % nsubplotcols == 0:
                ax.set_ylabel('Mismatches')
            legend_1 = plt.legend(zip(*plot_lines_levs)[0],
                                  l_lev,
                                  loc="upper left",
                                  prop={'size': 8},
                                  fancybox=True)
            legend_2 = plt.legend(zip(*plot_lines)[0],
                                  l_taper,
                                  ncol=2,
                                  loc="lower right",
                                  prop={'size': 8},
                                  fancybox=True)
            legend_3 = plt.legend(plot_lines[0],
                                  l_ccer,
                                  ncol=2,
                                  loc='lower left',
                                  prop={'size': 8},
                                  fancybox=True)
            ax.legend()
            ax.add_artist(legend_1)
            ax.add_artist(legend_2)
            ax.add_artist(legend_3)
            ax.set_ylim([1.e-5, 0.01])
        #
        subprocess.getoutput('mkdir -p %s' % self.simdir + '/' + savedir)
        savedir = self.simdir + '/' + savedir
        if savefig is not None:
            plt.savefig(savedir + '/' + savefig, dpi=400)
        else:
            plt.savefig(savedir + '/' + self.simtag + '_cceOnly' + '.png',
                        dpi=400)
        return

    # }}}
    #

    def plot_cce_extrapolation_mismatches(self,
                                          nsubplotrows=2,
                                          nsubplotcols=3,
                                          savedir=None,
                                          savefig=None):
        # {{{
        if savedir is None:
            savedir = self.plotdir
        # panel1,2,3,4
        # For each tapering: x5
        #
        #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,sharex='col',sharey='row')
        fig = plt.figure(int(1e7 * np.random.random()))
        fig.set_size_inches(16, 12)
        fig.suptitle(self.simtag, fontsize=14)
        self.data.ccelevs.sort()
        self.data.cceradii.sort()
        const_list = []
        for i in range(len(self.data.ccelevs)):
            const_list.append([str(self.data.ccelevs[i])])
        const_list = const_list + const_list
        print(const_list)
        # const_list = self.data.ccelevs + self.data.ccelevs # One plot for each lev
        const_func = [
            self.data.cceextrapolated, self.data.cceextrapolated,
            self.data.cceextrapolated, self.data.cceextrapolated,
            self.data.cceextrapolated, self.data.cceextrapolated
        ]
        #const_keys = [self.data.extraporders,self.data.cceradii]
        const_addtn = [[str(self.data.cceradii[0])],
                       [str(self.data.cceradii[0])],
                       [str(self.data.cceradii[0])],
                       [str(self.data.cceradii[1])],
                       [str(self.data.cceradii[1])],
                       [str(self.data.cceradii[1])]]
        # subplot each for a unique Lev+CCER combination
        for idx in range(6):
            print("idx = ", idx)
            additional_constraint = const_addtn[idx]
            print("additional constraing = ", additional_constraint)
            ax = plt.subplot(nsubplotrows, nsubplotcols, idx + 1)  # <<
            l_taper = self.taperlabels
            l_lev = []
            l_ccer = []
            plot_lines_all = []
            plot_lines_levs = []
            # self.data.cceradii): #<<
            for lidx, lev in enumerate(const_list[idx]):
                lev = str(lev)
                print("lev = ", lev)
                l_lev.append(lev)
                const_key = lev
                # For each tapering: x6
                plot_lines = [[], [], [], [], [], []]
                overlaps = const_func[idx](key=additional_constraint + [lev],
                                           noduplicate=True)
                print(overlaps)
                # self.data.ccelev(key=lev,noduplicate=True)#.values()[0] <<
                for i in range(overlaps.values()[0].nWindows):
                    for n in range(len(overlaps.keys())):
                        print("i = %d/%d, n = %d/%d" %
                              (i, overlaps.values()[0].nWindows, n,
                               len(overlaps.keys())))
                        var_key = str(overlaps.keys()[n])
                        print("var_key = ", var_key)
                        olap = overlaps[var_key]
                        mass = olap.X()
                        mismatch = 1. - olap.Y(i)
                        #
                        if i == 0 and lidx == 0:
                            l_ccer.append(var_key.split('/')[-1])
                        pl, = plt.plot(mass,
                                       mismatch,
                                       linestyle=self.lines[lidx],
                                       lw=1,
                                       marker=self.markers[n],
                                       markersize=3,
                                       color=self.colors[i],
                                       alpha=0.65)
                        plot_lines[i].append(pl, )
                plot_lines_all.append(plot_lines)
                plot_lines_levs.append(plot_lines[0])
            #
            print("l_taper,lev,ccer = ", l_taper, l_lev, l_ccer)
            #
            ax.set_yscale('log')
            ax.grid(b=True, which='major')
            if idx + 1 >= nsubplotcols * (nsubplotrows - 1):
                ax.set_xlabel('mass (solar mass)')
            if idx % nsubplotcols == 0:
                ax.set_ylabel('Mismatches')
            legend_1 = plt.legend(zip(*plot_lines_levs)[0],
                                  l_lev,
                                  loc="upper left",
                                  prop={'size': 8},
                                  fancybox=True)
            legend_2 = plt.legend(zip(*plot_lines)[0],
                                  l_taper,
                                  ncol=3,
                                  loc="lower right",
                                  prop={'size': 8},
                                  fancybox=True)
            legend_3 = plt.legend(plot_lines[0],
                                  l_ccer,
                                  ncol=1,
                                  loc='upper right',
                                  prop={'size': 8},
                                  fancybox=True)
            ax.legend()
            ax.add_artist(legend_1)
            ax.add_artist(legend_2)
            ax.add_artist(legend_3)
            ax.set_ylim([1.e-5, 0.01])
        #
        subprocess.getoutput('mkdir -p %s' % self.simdir + '/' + savedir)
        savedir = self.simdir + '/' + savedir
        if savefig is not None:
            plt.savefig(savedir + '/' + savefig, dpi=500)
        else:
            plt.savefig(savedir + '/' + self.simtag + '_Extraction' + '.png',
                        dpi=500)
        return
        # }}}

    #

    def plot_cce_max_mismatch(self, savedir=None, savefig=None):
        # {{{
        if savedir is None:
            savedir = self.plotdir
        #
        overlaps = self.data.get_max_cce_mismatch()
        X = overlaps.X()
        Y = overlaps.Y()
        fig = plt.figure(int(1e7 * np.random.random()))
        for i in range(overlaps.nWindows):
            plt.semilogy(X, 1. - Y, label=self.taperlabels[i])
        plt.grid()
        plt.xlabel('Total mass (solar masses)')
        plt.ylabel('Max of CCER, Lev, Extraction mismatches')
        plt.title(self.simtag)
        plt.legend(ncol=2, loc='lower left')
        if savefig:
            plt.savefig(savedir + '/' + savefig, dpi=400)
        else:
            plt.savefig(savedir + '/' + self.simtag + '_MAXNR.png', dpi=400)
        return
        # }}}


class plot_mismatches_sims():
    """ 
    This class makes population plots. This is done to find patterns between
    NR errors based on binary parameters."""
    def __init__(self,
                 basedir=None,
                 simdirs=None,
                 matchdirs=['matches'],
                 plotdir='plots',
                 verbose=True,
                 debug=False):
        self.verbose = verbose
        self.debug = debug
        self.basedir = basedir
        self.simdirs = simdirs
        self.simtag = self.simdirs.strip('/').split('/')[-1]
        self.data = {}
        for simdir in self.simdirs:
            self.data[simdir] = plot_mismatches_sim(simdir=self.basedir + '/' +
                                                    simdir,
                                                    matchdirs=matchdirs,
                                                    verbose=self.verbose,
                                                    debug=self.debug)
            for i in range(len(self.data[simdir].ccelevs)):
                self.data.ccelevs[i] = str(self.data.ccelevs[i])
            for i in range(len(self.data[simdir].cceradii)):
                self.data[simdir].cceradii[i] = str(
                    self.data[simdir].cceradii[i])
        self.lines = ["-", "--", "-.", "-:"]
        self.markers = ["o", "x", "s", "^", "v", "*", '.', '<']
        self.colors = [
            "blue", "red", "green", "magenta", "cyan", "gold", "black",
            "darkorange"
        ]
        self.taperlabels = ["None", "A", "B", "C", "D", "E"]
        self.plotdir = plotdir + matchdirs[0].lstrip('matches')

    #

    def hist_cce_mismatch(self):
        """
        This function will make histograms of different sources of error, across
        the catalog. The catalog is given as a list of dir names. The errors considered here are:
        1. NR Res: For R = OUTER: Lev4 vs Lev5, Lev3 vs Lev4
        2. CCE Radius: For Lev = Lev5: R1 vs R2
        3. Extraction: For Lev=Lev5, R = OUTER vs N2, N3, N4
        """
        # {{{
        # Get all required information for each sim, and then combine the info
        overlaps_all = {}
        for d in self.simdirs:
            overlaps_all[d] = []
            spl_lev = 'Lev5'
            spl_extrap = ['N2', 'N3', 'N4']
            spl_ccer = 'CceR%04d' % max(
                [int(x[-4:]) for x in self.data[d].cceradii])
            #
            const_combinations = [[[spl_ccer, 'Lev4', 'Lev5'],
                                   [spl_ccer, 'Lev3', 'Lev4']],
                                  [['Lev3'] + self.data[d].cceradii,
                                   [['Lev4'] + self.data[d].cceradii],
                                   [['Lev5'] + self.data[d].cceradii]],
                                  [['Lev3', spl_ccer, 'N2'],
                                   ['Lev3', spl_ccer, 'N3'],
                                   ['Lev3', spl_ccer, 'N4'],
                                   ['Lev4', spl_ccer, 'N2'],
                                   ['Lev4', spl_ccer, 'N3'],
                                   ['Lev4', spl_ccer, 'N4'],
                                   ['Lev5', spl_ccer, 'N2'],
                                   ['Lev5', spl_ccer, 'N3'],
                                   ['Lev5', spl_ccer, 'N4']]]
            const_funcs = [
                [self.data[d].ccelev, self.data[d].ccelev],
                [self.data[d].ccer, self.data[d].ccer, self.data[d].ccer],
                [
                    self.data[d].cceextrapolated, self.data[d].cceextrapolated,
                    self.data[d].cceextrapolated, self.data[d].cceextrapolated,
                    self.data[d].cceextrapolated, self.data[d].cceextrapolated,
                    self.data[d].cceextrapolated, self.data[d].cceextrapolated,
                    self.data[d].cceextrapolated
                ]
            ]
            for kdx, key_combos in enumerate(const_combinations):
                overlaps_tmp = []
                for idx, key_combo in enumerate(key_combos):
                    overlaps = const_funcs[jdx][idx](key=key_combo,
                                                     noduplicate=True)
                    if len(overlaps.values()) > 1:
                        raise RuntimeError("keys %s gave %d resuls" %
                                           (key_combo, len(overlaps.keys())))
                    # proceed if only 1 dataset
                    overlaps_tmp.append(overlaps.values()[0])
                overlaps_all[d].append(overlaps_tmp)
        #
        # Now combine info for all sims. CONCATENATE, NOT MAXIMIZE!!!
        # TODO
        # }}}

    #

    def plot_cce_mismatches_all(self,
                                nsubplotrows=2,
                                nsubplotcols=2,
                                savedir=None,
                                savefig=None):
        # {{{
        if savedir is None:
            savedir = self.plotdir
        # panel1,2,3,4
        # For each tapering: x5
        #
        #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,sharex='col',sharey='row')
        fig = plt.figure(int(1e7 * np.random.random()))
        fig.set_size_inches(16, 12)
        fig.suptitle(self.simtag, fontsize=14)
        self.data.ccelevs.sort()
        self.data.cceradii.sort()
        const_list = [
            self.data.ccelevs, self.data.cceradii, self.data.ccelevs,
            self.data.ccelevs
        ]
        const_func = [
            self.data.ccer, self.data.ccelev, self.data.cceextrapolated,
            self.data.cceextrapolated
        ]
        const_keys = [[], [], self.data.extraporders, self.data.cceradii]
        const_addtn = [[], [], [str(self.data.cceradii[0])],
                       [str(self.data.cceradii[1])]]
        for idx in range(4):
            print("idx = ", idx)
            additional_constraint = const_addtn[idx]
            print("additional constraing = ", additional_constraint)
            ax = plt.subplot(nsubplotrows, nsubplotcols, idx + 1)  # <<
            l_taper = self.taperlabels
            l_lev = []
            l_ccer = []
            plot_lines_all = []
            plot_lines_levs = []
            # self.data.cceradii): #<<
            for lidx, lev in enumerate(const_list[idx]):
                lev = str(lev)
                print("lev = ", lev)
                l_lev.append(lev)
                const_key = lev
                # For each tapering: x6
                plot_lines = [[], [], [], [], [], []]
                if idx >= 4:
                    overlaps = {}
                    for l2 in const_keys[idx]:
                        print("l2, lev = ", l2, lev)
                        overlaps[l2] = const_func[idx](key=[l2] + [lev],
                                                       noduplicate=True)
                else:
                    overlaps = const_func[idx](key=additional_constraint +
                                               [lev],
                                               noduplicate=True)
                print(overlaps)
                # self.data.ccelev(key=lev,noduplicate=True)#.values()[0] <<
                for i in range(overlaps.values()[0].nWindows):
                    for n in range(len(overlaps.keys())):
                        print("i = %d/%d, n = %d/%d" %
                              (i, overlaps.values()[0].nWindows, n,
                               len(overlaps.keys())))
                        var_key = str(overlaps.keys()[n])
                        print("var_key = ", var_key)
                        olap = overlaps[var_key]
                        mass = olap.X()
                        mismatch = 1. - olap.Y(i)
                        #
                        if i == 0 and lidx == 0:
                            l_ccer.append(var_key.split('/')[-1])
                        pl, = plt.plot(mass,
                                       mismatch,
                                       linestyle=self.lines[lidx],
                                       lw=1,
                                       marker=self.markers[n],
                                       markersize=3,
                                       color=self.colors[i],
                                       alpha=0.65)
                        plot_lines[i].append(pl, )
                plot_lines_all.append(plot_lines)
                plot_lines_levs.append(plot_lines[0])
            #
            print("l_taper,lev,ccer = ", l_taper, l_lev, l_ccer)
            #
            if idx + 1 >= nsubplotcols * (nsubplotrows - 1):
                ax.set_xlabel('mass (solar mass)')
            if idx % nsubplotcols == 0:
                ax.set_ylabel('Mismatches')
            legend_1 = plt.legend(zip(*plot_lines_levs)[0],
                                  l_lev,
                                  loc="upper left",
                                  prop={'size': 8},
                                  fancybox=True)
            legend_2 = plt.legend(zip(*plot_lines)[0],
                                  l_taper,
                                  ncol=2,
                                  loc="lower right",
                                  prop={'size': 8},
                                  fancybox=True)
            legend_3 = plt.legend(plot_lines[0],
                                  l_ccer,
                                  ncol=2,
                                  loc='lower left',
                                  prop={'size': 8},
                                  fancybox=True)
            ax.legend()
            ax.add_artist(legend_1)
            ax.add_artist(legend_2)
            ax.add_artist(legend_3)
            ax.set_ylim([1.e-5, 0.01])
        #
        subprocess.getoutput('mkdir -p %s' % self.simdir + '/' + savedir)
        savedir = self.simdir + '/' + savedir
        if savefig is not None:
            plt.savefig(savedir + '/' + savefig, dpi=500)
        else:
            plt.savefig(savedir + '/' + self.simtag + '.png', dpi=500)
        return
        # }}}


#########################################################
# EFFECTUALNESS


class plot_effectualness_vs_totalmass():
    # {{{
    def __init__(self,
                 outdir='.',
                 infiles=['matches/match1.h5'],
                 plotdir='plots',
                 verbose=True):
        self.verbose = verbose
        self.data = None
        self.outdir = outdir
        self.infiles = infiles
        self.plotdir = outdir + '/' + plotdir
        self.ApproxList = [
            'SEOBNRv1.dat', 'SEOBNRv2.dat', 'IMRPhenomC.dat', 'IMRPhenomD.dat',
            'SpinTaylorT4.dat', 'TaylorF2.dat'
        ]
        self.lines = ["-x", "-o", "-.", "-^", "-v", "--"]
        self.markers = ["o", "x", "s", "^", "v", "*", '.', '<']
        self.colors = [
            "blue", "red", "green", "magenta", "cyan", "gold", "black",
            "darkorange"
        ]

    #

    def read_data_from_combined_file(
            self,
            simtags=[],
            filename="EffectualnessParameterBiases_AllSims.h5"):
        # {{{
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            raise IOError("%s not found" % filename)
        pwd = subprocess.getoutput('pwd')
        os.chdir(self.outdir)
        self.data = EffectualnessAndBias(outdir=self.outdir)
        if self.verbose:
            print("reading from >> ", filename, file=sys.stderr)
        self.data.read_data_from_combined_file(simtags=simtags,
                                               filename=filename)
        os.chdir(pwd)
        return
        # }}}

    #

    def read_data_from_all_files(self, tags=['./matches/*.h5'], simtags=[]):
        # {{{
        pwd = subprocess.getoutput('pwd')
        os.chdir(self.outdir)
        self.data = EffectualnessAndBias(outdir=self.outdir)
        if len(tags) > 0:
            infiles = []
            for tag in tags:
                infiles = infiles + glob.glob(tag)
        elif len(self.infiles) > 0:
            infiles = self.infiles
        else:
            raise IOError(
                "Please specify which files to read, as a list OR tag")
        # print "USING ONLY 10 DATA FILES -- FIXME!"
        # infiles = infiles[:10] # FIXME
        if self.verbose:
            print("reading from >> ", infiles, file=sys.stderr)
        for f in infiles:
            self.data.read_data_from_file(filename=f, simtags=simtags)
        os.chdir(pwd)
        return
        # }}}

    #

    def plot_effectualness_vs_totalmass(self,
                                        inkey=None,
                                        logy=True,
                                        figtype='pdf'):
        # {{{
        try:
            import matplotlib.pyplot as plt
        except:
            return
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        for sim in all_sims:
            plt.figure(int(1e7 * np.random.random()))
            for idx, app in enumerate(self.ApproxList):
                mm, ff = self.data.effectualness_vs_totalmass(inkey=sim,
                                                              approx=app)
                # print "Masses = ", mm
                # print "FF = ", ff
                if not logy:
                    plt.plot(mm,
                             ff,
                             label=app,
                             linestyle=self.lines[-1],
                             lw=3,
                             marker=self.markers[idx],
                             markersize=3,
                             color=self.colors[idx])
                else:
                    plt.semilogy(mm,
                                 1. - ff,
                                 label=app,
                                 linestyle=self.lines[-1],
                                 lw=3,
                                 marker=self.markers[idx],
                                 markersize=3,
                                 color=self.colors[idx])
                plt.hold(True)
            plt.ylim(1.e-4, 1)
            plt.legend(loc='best')
            plt.grid()
            plt.xlabel('Total Mass')  # ($M_\odot$)')
            plt.ylabel('Effectualness')
            plt.title(sim.replace('_', '-'))
            plt.savefig(self.plotdir + '/FF_%s.%s' % (sim[:-4], figtype))
        return
        # }}}

    def plot_effectualness_totalmass_vs_parameters(self,
                                                   inkey=None,
                                                   logy=True,
                                                   figtype='pdf'):
        # {{{
        try:
            import matplotlib.pyplot as plt
        except:
            return
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
            np.array([]), np.array([]), np.array([]), np.array([])
        for idx, app in enumerate(self.ApproxList):
            plt.figure(int(1e7 * np.random.random()))
            for sim in all_sims:
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_ff = np.append(nr_ff, ff)
            #
            # Mass - spin1
            if self.verbose:
                print("Making M-S1 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_masses,
                              nr_spin1z,
                              nr_ff,
                              xlabel='Total Mass',
                              ylabel='Spin of Bigger BH',
                              zlabel='Fitting Factor',
                              title=app[:-4],
                              savefig=self.plotdir +
                              '/FF_TotalMass_Spin1z_%s.%s' %
                              (app[:-4], figtype))
            # Mass - spin2
            if self.verbose:
                print("Making M-S2 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_masses,
                              nr_spin2z,
                              nr_ff,
                              xlabel='Total Mass',
                              ylabel='Spin of Smaller BH',
                              zlabel='Fitting Factor',
                              title=app[:-4],
                              savefig=self.plotdir +
                              '/FF_TotalMass_Spin2z_%s.%s' %
                              (app[:-4], figtype))
            # MassRatio - spin1
            if self.verbose:
                print("Making Q-S1 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_massratios,
                              nr_spin1z,
                              nr_ff,
                              xlabel='Mass Ratio',
                              ylabel='Spin of Bigger BH',
                              zlabel='Fitting Factor',
                              title=app[:-4],
                              savefig=self.plotdir +
                              '/FF_MassRatio_Spin1z_%s.%s' %
                              (app[:-4], figtype))
            # MassRatio - spin2
            if self.verbose:
                print("Making Q-S2 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_massratios,
                              nr_spin2z,
                              nr_ff,
                              xlabel='Mass Ratio',
                              ylabel='Spin of Smaller BH',
                              zlabel='Fitting Factor',
                              title=app[:-4],
                              savefig=self.plotdir +
                              '/FF_MassRatio_Spin2z_%s.%s' %
                              (app[:-4], figtype))
        return
        # }}}

    def plot_mchirperror_vs_totalmass_parameters(self,
                                                 inkey=None,
                                                 logy=True,
                                                 figtype='pdf'):
        # {{{
        try:
            import matplotlib.pyplot as plt
        except:
            return
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff,\
            mchirp_diff, eta_diff, spin1z_diff, spin2z_diff =\
            np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), \
            np.array([]), np.array([]), np.array([]), np.array([])
        for idx, app in enumerate(self.ApproxList):
            plt.figure(int(1e7 * np.random.random()))
            for sim in all_sims:
                masses, nr_q, nr_s1, nr_s2, mc_d, et_d, s1_d, s2_d, ff = \
                    self.data.parameterbiases_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_ff = np.append(nr_ff, ff)
                mchirp_diff = np.append(mchirp_diff, mc_d)
                eta_diff = np.append(eta_diff, et_d)
                spin1z_diff = np.append(spin1z_diff, s1_d)
                spin2z_diff = np.append(spin2z_diff, s2_d)
            #
            # Mass - spin1
            if self.verbose:
                print("Making M-S1 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_masses,
                              nr_spin1z,
                              mchirp_diff,
                              xlabel='Total Mass',
                              ylabel='Spin of Bigger BH',
                              zlabel='Chirp Mass Fractional Bias',
                              title=app[:-4],
                              logz=False,
                              savefig=self.plotdir +
                              '/ChirpMassBias_TotalMass_Spin1z_%s.%s' %
                              (app[:-4], figtype))
            # Mass - spin2
            if self.verbose:
                print("Making M-S2 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_masses,
                              nr_spin2z,
                              mchirp_diff,
                              xlabel='Total Mass',
                              ylabel='Spin of Smaller BH',
                              zlabel='Chirp Mass Fractional Bias',
                              title=app[:-4],
                              logz=False,
                              savefig=self.plotdir +
                              '/ChirpMassBias_TotalMass_Spin2z_%s.%s' %
                              (app[:-4], figtype))
            # MassRatio - spin1
            if self.verbose:
                print("Making Q-S1 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_massratios,
                              nr_spin1z,
                              mchirp_diff,
                              xlabel='Mass Ratio',
                              ylabel='Spin of Bigger BH',
                              zlabel='Chirp Mass Fractional Bias',
                              title=app[:-4],
                              logz=False,
                              savefig=self.plotdir +
                              '/ChirpMassBias_MassRatio_Spin1z_%s.%s' %
                              (app[:-4], figtype))
            # MassRatio - spin2
            if self.verbose:
                print("Making Q-S2 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_massratios,
                              nr_spin2z,
                              mchirp_diff,
                              xlabel='Mass Ratio',
                              ylabel='Spin of Smaller BH',
                              zlabel='Chirp Mass Fractional Bias',
                              title=app[:-4],
                              logz=False,
                              savefig=self.plotdir +
                              '/ChirpMassBias_MassRatio_Spin2z_%s.%s' %
                              (app[:-4], figtype))
        return
        # }}}

    def plot_etaerror_vs_totalmass_parameters(self,
                                              inkey=None,
                                              logy=True,
                                              figtype='pdf'):
        # {{{
        try:
            import matplotlib.pyplot as plt
        except:
            return
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff,\
            mchirp_diff, eta_diff, spin1z_diff, spin2z_diff =\
            np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), \
            np.array([]), np.array([]), np.array([]), np.array([])
        for idx, app in enumerate(self.ApproxList):
            plt.figure(int(1e7 * np.random.random()))
            for sim in all_sims:
                masses, nr_q, nr_s1, nr_s2, mc_d, et_d, s1_d, s2_d, ff = \
                    self.data.parameterbiases_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_ff = np.append(nr_ff, ff)
                mchirp_diff = np.append(mchirp_diff, mc_d)
                eta_diff = np.append(eta_diff, et_d)
                spin1z_diff = np.append(spin1z_diff, s1_d)
                spin2z_diff = np.append(spin2z_diff, s2_d)
            #
            # Mass - spin1
            if self.verbose:
                print("Making M-S1 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_masses,
                              nr_spin1z,
                              eta_diff,
                              xlabel='Total Mass',
                              ylabel='Spin of Bigger BH',
                              zlabel='Eta Fractional Bias',
                              title=app[:-4],
                              logz=False,
                              savefig=self.plotdir +
                              '/EtaBias_TotalMass_Spin1z_%s.%s' %
                              (app[:-4], figtype))
            # Mass - spin2
            if self.verbose:
                print("Making M-S2 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_masses,
                              nr_spin2z,
                              eta_diff,
                              xlabel='Total Mass',
                              ylabel='Spin of Smaller BH',
                              zlabel='Eta Fractional Bias',
                              title=app[:-4],
                              logz=False,
                              savefig=self.plotdir +
                              '/EtaBias_TotalMass_Spin2z_%s.%s' %
                              (app[:-4], figtype))
            # MassRatio - spin1
            if self.verbose:
                print("Making Q-S1 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_massratios,
                              nr_spin1z,
                              eta_diff,
                              xlabel='Mass Ratio',
                              ylabel='Spin of Bigger BH',
                              zlabel='Eta Fractional Bias',
                              title=app[:-4],
                              logz=False,
                              savefig=self.plotdir +
                              '/EtaBias_MassRatio_Spin1z_%s.%s' %
                              (app[:-4], figtype))
            # MassRatio - spin2
            if self.verbose:
                print("Making Q-S2 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_massratios,
                              nr_spin2z,
                              eta_diff,
                              xlabel='Mass Ratio',
                              ylabel='Spin of Smaller BH',
                              zlabel='Eta Fractional Bias',
                              title=app[:-4],
                              logz=False,
                              savefig=self.plotdir +
                              '/EtaBias_MassRatio_Spin2z_%s.%s' %
                              (app[:-4], figtype))
        return
        # }}}

    def plot_spin1error_vs_totalmass_parameters(self,
                                                inkey=None,
                                                logy=True,
                                                figtype='pdf'):
        # {{{
        try:
            import matplotlib.pyplot as plt
        except:
            return
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff,\
            mchirp_diff, eta_diff, spin1z_diff, spin2z_diff =\
            np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), \
            np.array([]), np.array([]), np.array([]), np.array([])
        for idx, app in enumerate(self.ApproxList):
            plt.figure(int(1e7 * np.random.random()))
            for sim in all_sims:
                masses, nr_q, nr_s1, nr_s2, mc_d, et_d, s1_d, s2_d, ff = \
                    self.data.parameterbiases_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_ff = np.append(nr_ff, ff)
                mchirp_diff = np.append(mchirp_diff, mc_d)
                eta_diff = np.append(eta_diff, et_d)
                spin1z_diff = np.append(spin1z_diff, s1_d)
                spin2z_diff = np.append(spin2z_diff, s2_d)
            #
            # Mass - spin1
            if self.verbose:
                print("Making M-S1 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_masses,
                              nr_spin1z,
                              spin1z_diff,
                              xlabel='Total Mass',
                              ylabel='Spin of Bigger BH',
                              zlabel='Spin1 Fractional Bias',
                              title=app[:-4],
                              logz=False,
                              savefig=self.plotdir +
                              '/Spin1Bias_TotalMass_Spin1z_%s.%s' %
                              (app[:-4], figtype))
            # Mass - spin2
            if self.verbose:
                print("Making M-S2 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_masses,
                              nr_spin2z,
                              spin1z_diff,
                              xlabel='Total Mass',
                              ylabel='Spin of Smaller BH',
                              zlabel='Spin1 Fractional Bias',
                              title=app[:-4],
                              logz=False,
                              savefig=self.plotdir +
                              '/Spin1Bias_TotalMass_Spin2z_%s.%s' %
                              (app[:-4], figtype))
            # MassRatio - spin1
            if self.verbose:
                print("Making Q-S1 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_massratios,
                              nr_spin1z,
                              spin1z_diff,
                              xlabel='Mass Ratio',
                              ylabel='Spin of Bigger BH',
                              zlabel='Spin1 Fractional Bias',
                              title=app[:-4],
                              logz=False,
                              savefig=self.plotdir +
                              '/Spin1Bias_MassRatio_Spin1z_%s.%s' %
                              (app[:-4], figtype))
            # MassRatio - spin2
            if self.verbose:
                print("Making Q-S2 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_massratios,
                              nr_spin2z,
                              spin1z_diff,
                              xlabel='Mass Ratio',
                              ylabel='Spin of Smaller BH',
                              zlabel='Spin1 Fractional Bias',
                              title=app[:-4],
                              logz=False,
                              savefig=self.plotdir +
                              '/Spin1Bias_MassRatio_Spin2z_%s.%s' %
                              (app[:-4], figtype))
        return
        # }}}

    def plot_spin2error_vs_totalmass_parameters(self,
                                                inkey=None,
                                                logy=True,
                                                figtype='pdf'):
        # {{{
        try:
            import matplotlib.pyplot as plt
        except:
            return
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff,\
            mchirp_diff, eta_diff, spin1z_diff, spin2z_diff =\
            np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), \
            np.array([]), np.array([]), np.array([]), np.array([])
        for idx, app in enumerate(self.ApproxList):
            plt.figure(int(1e7 * np.random.random()))
            for sim in all_sims:
                masses, nr_q, nr_s1, nr_s2, mc_d, et_d, s1_d, s2_d, ff = \
                    self.data.parameterbiases_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_ff = np.append(nr_ff, ff)
                mchirp_diff = np.append(mchirp_diff, mc_d)
                eta_diff = np.append(eta_diff, et_d)
                spin1z_diff = np.append(spin1z_diff, s1_d)
                spin2z_diff = np.append(spin2z_diff, s2_d)
            #
            # Mass - spin1
            if self.verbose:
                print("Making M-S1 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_masses,
                              nr_spin1z,
                              spin2z_diff,
                              xlabel='Total Mass',
                              ylabel='Spin of Bigger BH',
                              zlabel='Spin2 Fractional Bias',
                              title=app[:-4],
                              logz=False,
                              savefig=self.plotdir +
                              '/Spin2Bias_TotalMass_Spin1z_%s.%s' %
                              (app[:-4], figtype))
            # Mass - spin2
            if self.verbose:
                print("Making M-S2 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_masses,
                              nr_spin2z,
                              spin2z_diff,
                              xlabel='Total Mass',
                              ylabel='Spin of Smaller BH',
                              zlabel='Spin2 Fractional Bias',
                              title=app[:-4],
                              logz=False,
                              savefig=self.plotdir +
                              '/Spin2Bias_TotalMass_Spin2z_%s.%s' %
                              (app[:-4], figtype))
            # MassRatio - spin1
            if self.verbose:
                print("Making Q-S1 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_massratios,
                              nr_spin1z,
                              spin2z_diff,
                              xlabel='Mass Ratio',
                              ylabel='Spin of Bigger BH',
                              zlabel='Spin2 Fractional Bias',
                              title=app[:-4],
                              logz=False,
                              savefig=self.plotdir +
                              '/Spin2Bias_MassRatio_Spin1z_%s.%s' %
                              (app[:-4], figtype))
            # MassRatio - spin2
            if self.verbose:
                print("Making Q-S2 plot for ", app, file=sys.stderr)
            make_scatter_plot(nr_massratios,
                              nr_spin2z,
                              spin2z_diff,
                              xlabel='Mass Ratio',
                              ylabel='Spin of Smaller BH',
                              zlabel='Spin2 Fractional Bias',
                              title=app[:-4],
                              logz=False,
                              savefig=self.plotdir +
                              '/Spin2Bias_MassRatio_Spin2z_%s.%s' %
                              (app[:-4], figtype))
        return
        # }}}

    def plot_effectualness_vs_parameters(self,
                                         inkey=None,
                                         logy=True,
                                         elevation=30,
                                         azimuthal=30,
                                         alpha=0.8,
                                         figtype='pdf'):
        # {{{
        try:
            pass
        except:
            return
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
            np.array([]), np.array([]), np.array([]), np.array([])
        for idx, app in enumerate(self.ApproxList):
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_ff = np.append(nr_ff, ff)
            #
            # Mass - spin1
            if self.verbose:
                print("Making M-S1 plot for ", app, file=sys.stderr)
            print("With ", nr_spin1z, nr_spin2z, nr_masses, nr_ff)
            make_scatter_plot3D(nr_spin1z,
                                nr_spin2z,
                                nr_masses,
                                nr_ff,
                                elevation=elevation,
                                azimuthal=azimuthal,
                                alpha=alpha,
                                label=inkey,
                                xlabel='Spin of Bigger BH',
                                ylabel='Spin of Smaller BH',
                                zlabel='Total Mass',
                                clabel='Fitting Factor',
                                title=app[:-4],
                                savefig=self.plotdir +
                                '/FF_TotalMass_Spin1z_Spin2z_%s.%s' %
                                (app[:-4], figtype))
        return
        # }}}

    def write_effectualness_vs_parameters_mult(
            self, outfile='effectualness_parameters.h5'):
        # {{{
        if self.data == None:
            self.read_data_from_all_files()
        try:
            fout = h5py.File(outfile, 'w')
        except:
            raise IOError("Error opening %s for writing" % outfile)
        all_sims = self.data.data.keys()
        for sim in all_sims:
            fout.create_group(sim)
            for idx, app in enumerate(self.ApproxList):
                masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                    self.data.parameterbiases_vs_parameters(
                        inkey=sim, approx=app)
                dout = np.array([[
                    masses[i], nr_q[i], nr_s1[i], nr_s2[i], ff[i], mc_diff[i],
                    eta_diff[i], s1_diff[i], s2_diff[i]
                ] for i in range(len(ff))])
                fout[sim].create_dataset(app, data=dout)
        fout.close()
        return
        # }}}

    def plot_effectualness_vs_parameters_mult(self,
                                              logy=True,
                                              elevation=30,
                                              azimuthal=30,
                                              alpha=0.8,
                                              figtype='pdf'):
        # {{{
        try:
            pass
        except:
            return
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
        #                    np.array([]), np.array([]), np.array([]), np.array([])
        for idx, app in enumerate(self.ApproxList):
            # q = 1
            inkey = 'q1'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_ff = np.append(nr_ff, ff)
            nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            # q = 2
            inkey = 'q2'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_ff = np.append(nr_ff, ff)
            nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            # q = 3
            inkey = 'q3'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_ff = np.append(nr_ff, ff)
            nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            #
            if self.verbose:
                print("Making M-S1 plot for ", app, file=sys.stderr)
            print("With ", nr_spin1z, nr_spin2z, nr_masses, nr_ff)
            if 'SEOBNRv2' in app:
                bounds = np.log10([0.0001, 0.005, 0.01, 0.02, 0.03, 0.05])
            elif 'SEOBNRv1' in app:
                bounds = np.log10(
                    [0.0001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2])
            elif 'PhenomC' in app:
                bounds = np.log10([0.0001, 0.005, 0.01, 0.02, 0.03, 0.05])
            elif 'Taylor' in app:
                bounds = np.log10([0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.])
            else:
                raise IOError("Approximant %s bounds not known" % app)
            make_scatter_plot3D_mult(nr_spin1zq1,
                                     nr_spin2zq1,
                                     nr_massesq1,
                                     1 - nr_ffq1,
                                     nr_spin1zq2,
                                     nr_spin2zq2,
                                     nr_massesq2,
                                     1 - nr_ffq2,
                                     nr_spin1zq3,
                                     nr_spin2zq3,
                                     nr_massesq3,
                                     1 - nr_ffq3,
                                     elevation=elevation,
                                     azimuthal=azimuthal,
                                     alpha=alpha,
                                     xlabel='$\chi_1$',
                                     ylabel='$\chi_2$',
                                     zlabel='Total Mass$\,(M_{\odot})$',
                                     clabel='$\mathcal{M}$',
                                     bounds=bounds,
                                     title=app[:-4],
                                     savefig=self.plotdir +
                                     '/FF_TotalMass_Spin1z_Spin2z_%s_1.%s' %
                                     (app[:-4], figtype))
            if 'SEOBNRv2' in app:
                bounds = np.log10([0.0001, 0.005, 0.01, 0.05, 0.1])
            elif 'SEOBNRv1' in app:
                bounds = np.log10([0.0001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2])
            elif 'PhenomC' in app:
                bounds = np.log10([0.0001, 0.005, 0.01, 0.03, 0.05])
            elif 'Taylor' in app:
                bounds = np.log10([0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.])
            else:
                raise IOError("Approximant %s bounds not known" % app)
            make_scatter_plot3D_mult(nr_spin1zq1,
                                     nr_spin2zq1,
                                     nr_massesq1,
                                     1 - nr_ffq1,
                                     nr_spin1zq2,
                                     nr_spin2zq2,
                                     nr_massesq2,
                                     1 - nr_ffq2,
                                     nr_spin1zq3,
                                     nr_spin2zq3,
                                     nr_massesq3,
                                     1 - nr_ffq3,
                                     elevation=elevation,
                                     azimuthal=azimuthal,
                                     alpha=alpha,
                                     xlabel='$\chi_1$',
                                     ylabel='$\chi_2$',
                                     zlabel='Total Mass$\,(M_{\odot})$',
                                     clabel='$\mathcal{M}$',
                                     bounds=bounds,
                                     title=app[:-4],
                                     savefig=self.plotdir +
                                     '/FF_TotalMass_Spin1z_Spin2z_%s_3.%s' %
                                     (app[:-4], figtype))
            if 'SEOBNRv2' in app:
                bounds = np.log10([0.0001, 0.005, 0.01, 0.02, 0.03, 0.1])
            elif 'SEOBNRv1' in app:
                bounds = np.log10(
                    [0.0001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 1.])
            elif 'PhenomC' in app:
                bounds = np.log10([0.0001, 0.005, 0.01, 0.02, 0.03, 0.05])
            elif 'Taylor' in app:
                bounds = np.log10([0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.])
            else:
                raise IOError("Approximant %s bounds not known" % app)
            make_scatter_plot3D_mult(nr_spin1zq1,
                                     nr_spin2zq1,
                                     nr_massesq1,
                                     1 - nr_ffq1,
                                     nr_spin1zq2,
                                     nr_spin2zq2,
                                     nr_massesq2,
                                     1 - nr_ffq2,
                                     nr_spin1zq3,
                                     nr_spin2zq3,
                                     nr_massesq3,
                                     1 - nr_ffq3,
                                     elevation=elevation,
                                     azimuthal=azimuthal,
                                     alpha=alpha,
                                     xlabel='$\chi_1$',
                                     ylabel='$\chi_2$',
                                     zlabel='Total Mass$\,(M_{\odot})$',
                                     clabel='$\mathcal{M}$',
                                     bounds=bounds,
                                     title=app[:-4],
                                     savefig=self.plotdir +
                                     '/FF_TotalMass_Spin1z_Spin2z_%s_2.%s' %
                                     (app[:-4], figtype))
        return
        # }}}

    def plot_effectualness_vs_parameters_multrow(self,
                                                 logy=True,
                                                 elevation=30,
                                                 ApproxList=[],
                                                 onlyimr=True,
                                                 bounds=None,
                                                 azimuthal=30,
                                                 alpha=0.8,
                                                 figtype='pdf'):
        # {{{
        try:
            pass
        except:
            return
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        # which approximants to plot ?
        if type(ApproxList) == list and len(ApproxList) != 0:
            approx_present = True
            for app in ApproxList:
                if app not in self.ApproxList:
                    approx_present = False
            if not approx_present:
                ApproxList = self.ApproxList
        else:
            ApproxList = self.ApproxList
        #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
        #                    np.array([]), np.array([]), np.array([]), np.array([])
        Xs, Ys, Zs, Cs = [], [], [], []
        titles = []
        for idx, app in enumerate(ApproxList):
            if onlyimr:
                if 'Taylor' in app:
                    continue
            else:
                if 'Taylor' not in app:
                    continue
            print("\n\n Adding %s for plotting" % app)
            # q = 1
            inkey = 'q1'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_ff = np.append(nr_ff, ff)
            nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            # q = 2
            inkey = 'q2'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_ff = np.append(nr_ff, ff)
            nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            # q = 3
            inkey = 'q3'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_ff = np.append(nr_ff, ff)
            nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            #
            Xrow = [nr_spin1zq1, nr_spin1zq2, nr_spin1zq3]
            Yrow = [nr_spin2zq1, nr_spin2zq2, nr_spin2zq3]
            Zrow = [nr_massesq1, nr_massesq2, nr_massesq3]
            Crow = [1. - nr_ffq1, 1. - nr_ffq2, 1. - nr_ffq3]
            #
            Xs.append(Xrow)
            Ys.append(Yrow)
            Zs.append(Zrow)
            Cs.append(Crow)
            #
            titles.append(app[:-4])
        ###################################
        # Now make the plots
        # print "With ", nr_spin1z, nr_spin2z, nr_masses, nr_ff
        if type(bounds) != np.ndarray and type(bounds) != list:
            if bounds:
                print("SHOULD NOT HAPPEN")
                #bounds = np.array([-0.05,-0.03,-0.01,-0.005,0.005,0.01,0.03,0.05])
        make_scatter_plot3D_multrow(Xs, Ys, Zs, Cs,
                                    elevation=elevation, azimuthal=azimuthal, alpha=alpha,
                                    xlabel='$\chi_1$', ylabel='$\chi_2$',
                                    zlabel='M $(M_\odot)$',\
                                    # clabel='$\mathcal{M}$',\
                                    clabel='$1- \mathrm{Fitting~Factor}$',\
                                    title=titles,\
                                    logC=False,\
                                    bounds=bounds,\
                                    savefig=self.plotdir+'/FF_TotalMass_Spin1z_Spin2z.%s'\
                                    % (figtype))
        #
        return
        # }}}

    def plot_effectualness_contours_vs_parameters(self,
                                                  logy=True,
                                                  elevation=30,
                                                  ApproxList=[],
                                                  selectZ='minC',
                                                  FForMM='FF',
                                                  onlyimr=True,
                                                  bounds=None,
                                                  colors=[],
                                                  xlabel='',
                                                  ylabel='',
                                                  titles=[],
                                                  azimuthal=30,
                                                  alpha=0.8,
                                                  figtype='pdf'):
        # {{{
        try:
            pass
        except:
            return
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        # which approximants to plot ?
        if type(ApproxList) == list and len(ApproxList) != 0:
            approx_present = True
            for app in ApproxList:
                if app not in self.ApproxList:
                    approx_present = False
            if not approx_present:
                ApproxList = self.ApproxList
        else:
            ApproxList = self.ApproxList
        #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
        #                    np.array([]), np.array([]), np.array([]), np.array([])
        Xs, Ys, Zs, Cs = [], [], [], []
        title = []
        for idx, app in enumerate(ApproxList):
            if onlyimr:
                if 'Taylor' in app:
                    continue
            else:
                if 'Taylor' not in app:
                    continue
            print("\n\n Adding %s for plotting" % app)
            # q = 1
            inkey = 'q1'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                # Select which Z value to use for plotting
                if selectZ == 'maxC':
                    idxmaxC = np.where(ff == np.max(ff))[0][0]
                elif selectZ == 'minC':
                    idxmaxC = np.where(ff == np.min(ff))[0][0]
                nr_masses = np.append(nr_masses, masses[idxmaxC])
                nr_massratios = np.append(nr_massratios, nr_q[idxmaxC])
                nr_spin1z = np.append(nr_spin1z, nr_s1[idxmaxC])
                nr_spin2z = np.append(nr_spin2z, nr_s2[idxmaxC])
                nr_ff = np.append(nr_ff, ff[idxmaxC])
            nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            # q = 2
            inkey = 'q2'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                # Select which Z value to use for plotting
                if selectZ == 'maxC':
                    idxmaxC = np.where(ff == np.max(ff))[0][0]
                elif selectZ == 'minC':
                    idxmaxC = np.where(ff == np.min(ff))[0][0]
                nr_masses = np.append(nr_masses, masses[idxmaxC])
                nr_massratios = np.append(nr_massratios, nr_q[idxmaxC])
                nr_spin1z = np.append(nr_spin1z, nr_s1[idxmaxC])
                nr_spin2z = np.append(nr_spin2z, nr_s2[idxmaxC])
                nr_ff = np.append(nr_ff, ff[idxmaxC])
            nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            # q = 3
            inkey = 'q3'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                # Select which Z value to use for plotting
                if selectZ == 'maxC':
                    idxmaxC = np.where(ff == np.max(ff))[0][0]
                elif selectZ == 'minC':
                    idxmaxC = np.where(ff == np.min(ff))[0][0]
                nr_masses = np.append(nr_masses, masses[idxmaxC])
                nr_massratios = np.append(nr_massratios, nr_q[idxmaxC])
                nr_spin1z = np.append(nr_spin1z, nr_s1[idxmaxC])
                nr_spin2z = np.append(nr_spin2z, nr_s2[idxmaxC])
                nr_ff = np.append(nr_ff, ff[idxmaxC])
            nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            #
            Xrow = [nr_spin1zq1, nr_spin1zq2, nr_spin1zq3]
            Yrow = [nr_spin2zq1, nr_spin2zq2, nr_spin2zq3]
            Zrow = [nr_massesq1, nr_massesq2, nr_massesq3]
            if 'FF' in FForMM:
                Crow = [nr_ffq1, nr_ffq2, nr_ffq3]
            elif 'MM' in FForMM:
                Crow = [1. - nr_ffq1, 1. - nr_ffq2, 1. - nr_ffq3]
            #
            Xs.append(Xrow)
            Ys.append(Yrow)
            Zs.append(Zrow)
            Cs.append(Crow)
            #
            title.append(app[:-4])
        ###################################
        # Now make the plots
        if 'FF' in FForMM:
            plotprefix = 'FF'
        elif 'MM' in FForMM:
            plotprefix = 'MM'
        #
        if FForMM == 'FF':
            clabel = '$\mathrm{Fitting~Factor}$'
        elif FForMM == 'MM':
            clabel = '$1- \mathrm{Fitting~Factor}$'
        elif FForMM == 'VolFF':
            clabel = 'Detection Fraction'
            nrows, ncols = np.shape(Cs)
            for rowi in range(nrows):
                for coli in range(ncols):
                    Cs[rowi][coli] = Cs[rowi][coli]**3.
        elif FForMM == 'VolMM':
            clabel = 'Detection Loss'
            nrows, ncols = np.shape(Cs)
            for rowi in range(nrows):
                for coli in range(ncols):
                    Cs[rowi][coli] = 1. - (1. - Cs[rowi][coli])**3.
        print("With ", nr_spin1z, nr_spin2z, nr_masses, nr_ff)
        if type(bounds) != np.ndarray and type(bounds) != list:
            if bounds:
                print("SHOULD NOT HAPPEN")
                #bounds = np.array([-0.05,-0.03,-0.01,-0.005,0.005,0.01,0.03,0.05])
        #
        if xlabel == '':
            xlabel = '$\chi_1$'
        if ylabel == '':
            ylabel = '$\chi_2$'
        #
        make_contour_plot_multrow(Xs,
                                  Ys,
                                  Cs,
                                  alpha=alpha,
                                  logC=logy,
                                  xlabel=xlabel,
                                  ylabel=ylabel,
                                  titles=titles,
                                  clabel=clabel,
                                  title=title,
                                  bounds=bounds,
                                  colors=colors,
                                  savefig=self.plotdir +
                                  '/%s_Spin1z_Spin2z.%s' %
                                  (plotprefix, figtype))
        #
        return
        # }}}

    def plot_parameterbiases_vs_parameters_mult(self,
                                                logy=True,
                                                elevation=30,
                                                bounds=True,
                                                azimuthal=30,
                                                alpha=0.8,
                                                figtype='pdf'):
        # {{{
        try:
            pass
        except:
            return
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
        #                    np.array([]), np.array([]), np.array([]), np.array([])
        for idx, app in enumerate(self.ApproxList):
            # q = 1
            inkey = 'q1'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
                nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                    self.data.parameterbiases_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_mcdiff = np.append(nr_mcdiff, mc_diff)
                nr_etdiff = np.append(nr_etdiff, eta_diff)
                nr_s1diff = np.append(nr_s1diff, s1_diff)
                nr_s2diff = np.append(nr_s2diff, s2_diff)
                nr_ff = np.append(nr_ff, ff)
            nr_mcdiffq1, nr_etdiffq1, nr_s1diffq1, nr_s2diffq1 = nr_mcdiff,\
                nr_etdiff, nr_s1diff, nr_s2diff
            nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            # q = 2
            inkey = 'q2'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
                nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                    self.data.parameterbiases_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_mcdiff = np.append(nr_mcdiff, mc_diff)
                nr_etdiff = np.append(nr_etdiff, eta_diff)
                nr_s1diff = np.append(nr_s1diff, s1_diff)
                nr_s2diff = np.append(nr_s2diff, s2_diff)
                nr_ff = np.append(nr_ff, ff)
            nr_mcdiffq2, nr_etdiffq2, nr_s1diffq2, nr_s2diffq2 = nr_mcdiff,\
                nr_etdiff, nr_s1diff, nr_s2diff
            nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            # q = 3
            inkey = 'q3'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
                nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                    self.data.parameterbiases_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_mcdiff = np.append(nr_mcdiff, mc_diff)
                nr_etdiff = np.append(nr_etdiff, eta_diff)
                nr_s1diff = np.append(nr_s1diff, s1_diff)
                nr_s2diff = np.append(nr_s2diff, s2_diff)
                nr_ff = np.append(nr_ff, ff)
            nr_mcdiffq3, nr_etdiffq3, nr_s1diffq3, nr_s2diffq3 = nr_mcdiff,\
                nr_etdiff, nr_s1diff, nr_s2diff
            nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            #
            # Mass - spin1
            if self.verbose:
                print("Making M-S1 plot for ", app, file=sys.stderr)
            print("With ", nr_spin1z, nr_spin2z, nr_masses, nr_ff)
            if type(bounds) != np.ndarray and type(bounds) != list:
                if bounds:
                    print("SHOULD NOT HAPPEN")
                    bounds = np.array(
                        [-0.05, -0.03, -0.01, -0.005, 0.005, 0.01, 0.03, 0.05])
            make_scatter_plot3D_mult(
                nr_spin1zq1,
                nr_spin2zq1,
                nr_massesq1,
                nr_mcdiffq1,
                nr_spin1zq2,
                nr_spin2zq2,
                nr_massesq2,
                nr_mcdiffq2,
                nr_spin1zq3,
                nr_spin2zq3,
                nr_massesq3,
                nr_mcdiffq3,
                elevation=elevation,
                azimuthal=azimuthal,
                alpha=alpha,
                xlabel='$\chi_1$',
                ylabel='$\chi_2$',
                zlabel='Total Mass $(M_\odot)$',
                clabel='$\Delta\mathcal{M}_c/\mathcal{M}_c$',
                title=app[:-4],
                logC=False,
                bounds=bounds,
                savefig=self.plotdir +
                '/ChirpMassError_TotalMass_Spin1z_Spin2z_%s.%s' %
                (app[:-4], figtype))
            # return
            #
            if type(bounds) != np.ndarray and type(bounds) != list:
                if bounds:
                    bounds = np.array([
                        -0.5, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2,
                        0.5
                    ])
            make_scatter_plot3D_mult(
                nr_spin1zq1,
                nr_spin2zq1,
                nr_massesq1,
                nr_etdiffq1,
                nr_spin1zq2,
                nr_spin2zq2,
                nr_massesq2,
                nr_etdiffq2,
                nr_spin1zq3,
                nr_spin2zq3,
                nr_massesq3,
                nr_etdiffq3,
                elevation=elevation,
                azimuthal=azimuthal,
                alpha=alpha,
                xlabel='$\chi_1$',
                ylabel='$\chi_2$',
                zlabel='Total Mass $(M_\odot)$',
                clabel='$\Delta\eta/\eta$',
                title=app[:-4],
                logC=False,
                bounds=bounds,
                savefig=self.plotdir +
                '/EtaError_TotalMass_Spin1z_Spin2z_%s.%s' %
                (app[:-4], figtype))
            #
            if type(bounds) != np.ndarray and type(bounds) != list:
                if bounds:
                    bounds = np.array([
                        -0.5, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2,
                        0.5
                    ])
            make_scatter_plot3D_mult(
                nr_spin1zq1,
                nr_spin2zq1,
                nr_massesq1,
                nr_s1diffq1,
                nr_spin1zq2,
                nr_spin2zq2,
                nr_massesq2,
                nr_s1diffq2,
                nr_spin1zq3,
                nr_spin2zq3,
                nr_massesq3,
                nr_s1diffq3,
                elevation=elevation,
                azimuthal=azimuthal,
                alpha=alpha,
                xlabel='$\chi_1$',
                ylabel='$\chi_2$',
                zlabel='Total Mass $(M_\odot)$',
                clabel='$\Delta\chi_1$',
                title=app[:-4],
                logC=False,
                bounds=bounds,
                savefig=self.plotdir +
                '/Chi1Error_TotalMass_Spin1z_Spin2z_%s.%s' %
                (app[:-4], figtype))
            #
            if type(bounds) != np.ndarray and type(bounds) != list:
                if bounds:
                    bounds = np.array([
                        -0.5, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2,
                        0.5
                    ])
            make_scatter_plot3D_mult(
                nr_spin1zq1,
                nr_spin2zq1,
                nr_massesq1,
                nr_s2diffq1,
                nr_spin1zq2,
                nr_spin2zq2,
                nr_massesq2,
                nr_s2diffq2,
                nr_spin1zq3,
                nr_spin2zq3,
                nr_massesq3,
                nr_s2diffq3,
                elevation=elevation,
                azimuthal=azimuthal,
                alpha=alpha,
                xlabel='$\chi_1$',
                ylabel='$\chi_2$',
                zlabel='Total Mass $(M_\odot)$',
                clabel='$\Delta\chi_2$',
                title=app[:-4],
                logC=False,
                bounds=bounds,
                savefig=self.plotdir +
                '/Chi2Error_TotalMass_Spin1z_Spin2z_%s.%s' %
                (app[:-4], figtype))
            #
        return
        # }}}

    #

    def plot_parameterbiases_vs_parameters_multrow(self,
                                                   logy=True,
                                                   elevation=30,
                                                   ApproxList=[],
                                                   onlyimr=True,
                                                   chieff=False,
                                                   total_mass=False,
                                                   colormin=None,
                                                   colormax=None,
                                                   bounds=None,
                                                   azimuthal=30,
                                                   alpha=0.8,
                                                   figtype='pdf'):
        # {{{
        try:
            pass
        except:
            return
        #
        # which approximants to plot ?
        if type(ApproxList) == list and len(ApproxList) != 0:
            approx_present = True
            for app in ApproxList:
                if app not in self.ApproxList:
                    approx_present = False
            if not approx_present:
                ApproxList = self.ApproxList
        else:
            ApproxList = self.ApproxList
        ###############################
        # Read data
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
        #                    np.array([]), np.array([]), np.array([]), np.array([])
        Xs, Ys, Zs, mcCs, etCs, s1Cs, s2Cs = [], [], [], [], [], [], []
        titles = []
        for idx, app in enumerate(ApproxList):
            if onlyimr:
                if 'Taylor' in app:
                    continue
            # q = 1
            inkey = 'q1'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
                nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                    self.data.parameterbiases_vs_parameters(inkey=sim,
                                                            chieff=chieff, approx=app, total_mass=total_mass)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_mcdiff = np.append(nr_mcdiff, mc_diff)
                nr_etdiff = np.append(nr_etdiff, eta_diff)
                nr_s1diff = np.append(nr_s1diff, s1_diff)
                nr_s2diff = np.append(nr_s2diff, s2_diff)
                nr_ff = np.append(nr_ff, ff)
            nr_mcdiffq1, nr_etdiffq1, nr_s1diffq1, nr_s2diffq1 = nr_mcdiff,\
                nr_etdiff, nr_s1diff, nr_s2diff
            nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            # q = 2
            inkey = 'q2'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
                nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                    self.data.parameterbiases_vs_parameters(inkey=sim,
                                                            chieff=chieff, approx=app, total_mass=total_mass)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_mcdiff = np.append(nr_mcdiff, mc_diff)
                nr_etdiff = np.append(nr_etdiff, eta_diff)
                nr_s1diff = np.append(nr_s1diff, s1_diff)
                nr_s2diff = np.append(nr_s2diff, s2_diff)
                nr_ff = np.append(nr_ff, ff)
            nr_mcdiffq2, nr_etdiffq2, nr_s1diffq2, nr_s2diffq2 = nr_mcdiff,\
                nr_etdiff, nr_s1diff, nr_s2diff
            nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            # q = 3
            inkey = 'q3'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
                nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                    self.data.parameterbiases_vs_parameters(inkey=sim,
                                                            chieff=chieff, approx=app, total_mass=total_mass)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_mcdiff = np.append(nr_mcdiff, mc_diff)
                nr_etdiff = np.append(nr_etdiff, eta_diff)
                nr_s1diff = np.append(nr_s1diff, s1_diff)
                nr_s2diff = np.append(nr_s2diff, s2_diff)
                nr_ff = np.append(nr_ff, ff)
            nr_mcdiffq3, nr_etdiffq3, nr_s1diffq3, nr_s2diffq3 = nr_mcdiff,\
                nr_etdiff, nr_s1diff, nr_s2diff
            nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            #
            Xrow = [nr_spin1zq1, nr_spin1zq2, nr_spin1zq3]
            Yrow = [nr_spin2zq1, nr_spin2zq2, nr_spin2zq3]
            Zrow = [nr_massesq1, nr_massesq2, nr_massesq3]
            mcCrow = [nr_mcdiffq1, nr_mcdiffq2, nr_mcdiffq3]
            etCrow = [nr_etdiffq1, nr_etdiffq2, nr_etdiffq3]
            s1Crow = [nr_s1diffq1, nr_s1diffq2, nr_s1diffq3]
            s2Crow = [nr_s2diffq1, nr_s2diffq2, nr_s2diffq3]
            #
            Xs.append(Xrow)
            Ys.append(Yrow)
            Zs.append(Zrow)
            mcCs.append(mcCrow)
            etCs.append(etCrow)
            s1Cs.append(s1Crow)
            s2Cs.append(s2Crow)
            #
            titles.append(app[:-4])
        ###################################
        # Now make the plots
        # print "With ", nr_spin1z, nr_spin2z, nr_masses, nr_ff
        if type(bounds) != np.ndarray and type(bounds) != list:
            if bounds:
                print("SHOULD NOT HAPPEN")
                bounds = np.array(
                    [-0.11, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.11])
        print("bounds before calling plotting function  = ", bounds)
        if total_mass:
            make_scatter_plot3D_multrow(
                Xs,
                Ys,
                Zs,
                mcCs,
                elevation=elevation,
                azimuthal=azimuthal,
                alpha=alpha,
                xlabel='$\chi_1$',
                ylabel='$\chi_2$',
                zlabel='M $(M_\odot)$',
                clabel='(Recovered $M$ - Injected $M$) / Injected $M$',
                title=titles,
                logC=False,
                bounds=bounds,
                colormin=colormin,
                colormax=colormax,
                savefig=self.plotdir +
                '/TotalMassError_TotalMass_Spin1z_Spin2z.%s' % (figtype))
        else:
            make_scatter_plot3D_multrow(
                Xs,
                Ys,
                Zs,
                mcCs,
                elevation=elevation,
                azimuthal=azimuthal,
                alpha=alpha,
                xlabel='$\chi_1$',
                ylabel='$\chi_2$',
                zlabel='M $(M_\odot)$',
                clabel=
                '(Recovered $\mathcal{M}_c$ - Injected $\mathcal{M}_c$) / Injected $\mathcal{M}_c$',
                title=titles,
                logC=False,
                bounds=bounds,
                colormin=colormin,
                colormax=colormax,
                savefig=self.plotdir +
                '/ChirpMassError_TotalMass_Spin1z_Spin2z.%s' % (figtype))
        #
        if type(bounds) != np.ndarray and type(bounds) != list:
            if bounds:
                print("SHOULD NOT HAPPEN")
                bounds = np.array([
                    -0.23, -0.15, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.15,
                    0.25, 0.33
                ])
        make_scatter_plot3D_multrow(
            Xs,
            Ys,
            Zs,
            etCs,
            elevation=elevation,
            azimuthal=azimuthal,
            alpha=alpha,
            xlabel='$\chi_1$',
            ylabel='$\chi_2$',
            zlabel='M $(M_\odot)$',
            clabel='(Recovered $\eta$ - Injected $\eta$) / Injected $\eta$',
            title=titles,
            logC=False,
            bounds=bounds,
            colormin=colormin,
            colormax=colormax,
            savefig=self.plotdir + '/EtaError_TotalMass_Spin1z_Spin2z.%s' %
            (figtype))
        #
        if type(bounds) != np.ndarray and type(bounds) != list:
            if bounds:
                print("SHOULD NOT HAPPEN")
                bounds = np.array([
                    -0.5, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, 0.5
                ])
                if chieff:
                    bounds = np.array(
                        [-0.15, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.15])
        if chieff:
            #clabel='Recovered $\chi_\mathrm{eff}$ - Injected $\chi_\mathrm{eff}$'
            #clabel='Recovered $\chi_\mathrm{eff2PN}$ - Injected $\chi_\mathrm{eff2PN}$'
            clabel = 'Recovered $\chi_\mathrm{mw}$ - Injected $\chi_\mathrm{mw}$'
            # figtag='ChiEffPN'
            figtag = 'ChiMW'
        else:
            clabel = 'Recovered $\chi_1$ - Injected $\chi_1$'
            figtag = 'Chi1'
        make_scatter_plot3D_multrow(Xs, Ys, Zs, s1Cs,
                                    elevation=elevation, azimuthal=azimuthal, alpha=alpha,
                                    xlabel='$\chi_1$', ylabel='$\chi_2$',
                                    zlabel='M $(M_\odot)$',
                                    clabel=clabel, title=titles,\
                                    # clabel='$\Delta\chi_\mathrm{eff2PN}$', title=titles,\
                                    # clabel='$\Delta\chi_\mathrm{eff}$', title=titles,\
                                    # clabel='$\Delta\chi_\mathrm{mw}$', title=titles,\
                                    # clabel='$\Delta\chi_\mathrm{effD}$', title=titles,\
                                    logC=False,\
                                    bounds=bounds,\
                                    colormin=colormin, colormax=colormax,\
                                    savefig=self.plotdir+'/'+figtag+'Error_TotalMass_Spin1z_Spin2z.%s'\
                                    % (figtype))
        #
        if type(bounds) != np.ndarray and type(bounds) != list:
            if bounds:
                print("SHOULD NOT HAPPEN")
                bounds = np.array([
                    -0.5, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, 0.5
                ])
        make_scatter_plot3D_multrow(
            Xs,
            Ys,
            Zs,
            s2Cs,
            elevation=elevation,
            azimuthal=azimuthal,
            alpha=alpha,
            xlabel='$\chi_1$',
            ylabel='$\chi_2$',
            zlabel='M $(M_\odot)$',
            clabel='Recovered $\chi_2$ - Injected $\chi_2$',
            title=titles,
            logC=False,
            bounds=bounds,
            colormin=colormin,
            colormax=colormax,
            savefig=self.plotdir + '/Chi2Error_TotalMass_Spin1z_Spin2z.%s' %
            (figtype))
        #
        return
        # }}}

    #

    def plot_effectualness_contours_vs_spins(self,
                                             inkey=None,
                                             logy=True,
                                             alpha=0.8,
                                             figtype='pdf'):
        # {{{
        try:
            pass
        except:
            return
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        #
        for idx, app in enumerate(self.ApproxList):
            # q = 1
            inkey = 'q1'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_ff = np.append(nr_ff, ff)
            nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            # q = 2
            inkey = 'q2'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_ff = np.append(nr_ff, ff)
            nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            # q = 3
            inkey = 'q3'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_ff = np.append(nr_ff, ff)
            nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            #
            # spin1 - spin2
            if self.verbose:
                print("Making M-S1 plot for ", app, file=sys.stderr)
            print("With ", nr_spin1z, nr_spin2z, nr_masses, nr_ff)
            make_contourf_mult(nr_spin1zq1,
                               nr_spin2zq1,
                               nr_ffq1,
                               nr_spin1zq2,
                               nr_spin2zq2,
                               nr_ffq2,
                               nr_spin1zq3,
                               nr_spin2zq3,
                               nr_ffq3,
                               alpha=alpha,
                               xlabel='chi1',
                               ylabel='chi2',
                               clabel='Fitting Factor',
                               title=app[:-4],
                               savefig=self.plotdir +
                               '/FF_TotalMass_Spin1z_Spin2z_%s.%s' %
                               (app[:-4], figtype))
        return
        # }}}

    def plot_effectualness_vs_single_parameter(self,
                                               logy=True,
                                               ApproxList=[],
                                               parameter='chieff',
                                               massmid=70.,
                                               OneMinus=True,
                                               ChiNormalized=True,
                                               onlyimr=True,
                                               bounds=None,
                                               azimuthal=30,
                                               alpha=0.8,
                                               figtype='pdf'):
        # {{{
        try:
            from pycbc.pnutils import mtotal_eta_to_mass1_mass2
        except:
            print(
                "error import PyCBC modules / matplotlib. Cant make this plot."
            )
            return
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        #
        # which approximants to plot ?
        if type(ApproxList) == list and len(ApproxList) != 0:
            approx_present = True
            for app in ApproxList:
                if app not in self.ApproxList:
                    approx_present = False
            if not approx_present:
                ApproxList = self.ApproxList
        else:
            ApproxList = self.ApproxList
        #
        #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
        #                    np.array([]), np.array([]), np.array([]), np.array([])
        Xs, Ys, Zs, Cs = [], [], [], []
        titles = []
        for idx, app in enumerate(ApproxList):
            if onlyimr:
                if 'Taylor' in app:
                    continue
            else:
                if 'Taylor' not in app:
                    continue
            # q = 1
            inkey = 'q1'
            inq = 1.
            ineta = inq / (1. + inq)**2
            nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {}, {}, {}, {}, {}
            nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                np.array([])
            #
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                idx['high'] = np.where(ff == ff.max())[0][0]
                idx['low'] = np.where(ff == ff.min())[0][0]
                if massmid > 0:
                    idx['mid'] = np.where(masses == massmid)[0][0]
                else:
                    idx['mid'] = np.where(masses == masses.min())[0][0]
                for kk in ['low', 'mid', 'high']:
                    nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
                    nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
                    nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
                    nr_ff[kk] = np.append(nr_ff[kk], ff[idx[kk]])
            nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            nr_mass1q1, nr_mass2q1, nr_chieffq1 = {}, {}, {}
            for kk in ['low', 'mid', 'high']:
                nr_mass1q1[kk], nr_mass2q1[kk] = \
                    mtotal_eta_to_mass1_mass2(nr_masses[kk],
                                              ineta * np.ones(len(nr_masses[kk])))
                nr_chieffq1[kk] = spins_to_PNeffective_spin(
                    nr_mass1q1[kk], nr_mass2q1[kk], nr_spin1zq1[kk],
                    nr_spin2zq1[kk])
                if OneMinus:
                    nr_ffq1[kk] = 1. - nr_ffq1[kk]  # Convert FF to 1 - FF
            # q = 2
            inkey = 'q2'
            inq = 2.
            ineta = inq / (1. + inq)**2
            nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {}, {}, {}, {}, {}
            nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                np.array([])
            #
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                idx['high'] = np.where(ff == ff.max())[0][0]
                idx['low'] = np.where(ff == ff.min())[0][0]
                if massmid > 0:
                    idx['mid'] = np.where(masses == massmid)[0][0]
                else:
                    idx['mid'] = np.where(masses == masses.min())[0][0]
                for kk in ['low', 'mid', 'high']:
                    nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
                    nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
                    nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
                    nr_ff[kk] = np.append(nr_ff[kk], ff[idx[kk]])
            nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            nr_mass1q2, nr_mass2q2, nr_chieffq2 = {}, {}, {}
            for kk in ['low', 'mid', 'high']:
                nr_mass1q2[kk], nr_mass2q2[kk] = \
                    mtotal_eta_to_mass1_mass2(nr_masses[kk],
                                              ineta * np.ones(len(nr_masses[kk])))
                nr_chieffq2[kk] = spins_to_PNeffective_spin(
                    nr_mass1q2[kk], nr_mass2q2[kk], nr_spin1zq2[kk],
                    nr_spin2zq2[kk])
                if OneMinus:
                    nr_ffq2[kk] = 1. - nr_ffq2[kk]  # Convert FF to 1 - FF
            # q = 3
            inkey = 'q3'
            inq = 3.
            ineta = inq / (1. + inq)**2
            nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {}, {}, {}, {}, {}
            nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                np.array([])
            #
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, ff = \
                    self.data.effectualness_vs_parameters(
                        inkey=sim, approx=app)
                idx['high'] = np.where(ff == ff.max())[0][0]
                idx['low'] = np.where(ff == ff.min())[0][0]
                if massmid > 0:
                    idx['mid'] = np.where(masses == massmid)[0][0]
                else:
                    idx['mid'] = np.where(masses == masses.min())[0][0]
                for kk in ['low', 'mid', 'high']:
                    nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
                    nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
                    nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
                    nr_ff[kk] = np.append(nr_ff[kk], ff[idx[kk]])
            nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            nr_mass1q3, nr_mass2q3, nr_chieffq3 = {}, {}, {}
            for kk in ['low', 'mid', 'high']:
                nr_mass1q3[kk], nr_mass2q3[kk] = \
                    mtotal_eta_to_mass1_mass2(nr_masses[kk],
                                              ineta * np.ones(len(nr_masses[kk])))
                nr_chieffq3[kk] = spins_to_PNeffective_spin(
                    nr_mass1q3[kk], nr_mass2q3[kk], nr_spin1zq3[kk],
                    nr_spin2zq3[kk])
                if OneMinus:
                    nr_ffq3[kk] = 1. - nr_ffq3[kk]  # Convert FF to 1 - FF
            #
            if ChiNormalized:
                etaq1, etaq2, etaq3 = 1. / 4., 2. / 9., 3. / 16.
                Xrow = [
                    nr_chieffq1['mid'] / (1. - 76. * etaq1 / 113.),
                    nr_chieffq2['mid'] / (1. - 76. * etaq2 / 113.),
                    nr_chieffq3['mid'] / (1. - 76. * etaq3 / 113.)
                ]
            else:
                Xrow = [
                    nr_chieffq1['mid'], nr_chieffq2['mid'], nr_chieffq3['mid']
                ]
            Yrow = [nr_ffq1['mid'], nr_ffq2['mid'], nr_ffq3['mid']]
            Yerror = [
                np.append([nr_ffq1['mid'] - nr_ffq1['low']],
                          [nr_ffq1['high'] - nr_ffq1['mid']],
                          axis=0),
                np.append([nr_ffq2['mid'] - nr_ffq2['low']],
                          [nr_ffq2['high'] - nr_ffq2['mid']],
                          axis=0),
                np.append([nr_ffq3['mid'] - nr_ffq3['low']],
                          [nr_ffq3['high'] - nr_ffq3['mid']],
                          axis=0)
            ]
            Xerror = None
            #
            ###################################
            # Now make the plots
            if OneMinus:
                ymin, ymax = 1.e-3, 1.e-0
                ylabel = '1 - Fitting Factor'
                logy = True
                legendplacement = 'best'  # 'upper left'
                nameprefix = 'MM'
            else:
                ymin, ymax = 0.92, 1.
                ylabel = 'Fitting Factor'
                logy = False
                legendplacement = 'lower left'
                nameprefix = 'FF'
            if ChiNormalized:
                xlabel = '$\chi_\mathrm{eff}/(1-\\frac{76}{113}\eta)$'
            else:
                xlabel = '$\chi_\mathrm{eff}$'
            print(ymin, ymax, ylabel, logy, legendplacement, nameprefix)
            make_2Dplot_errorbars(Xrow,
                                  Yrow,
                                  Xerrs=Xerror,
                                  Yerrs=Yerror,
                                  xlabel=xlabel,
                                  ylabel=ylabel,
                                  title=app[:-4],
                                  logy=logy,
                                  ymin=ymin,
                                  ymax=ymax,
                                  labels=['q=1', 'q=2', 'q=3'],
                                  legendplacement=legendplacement,
                                  savefig=self.plotdir + '/' + nameprefix +
                                  'vsChiEff_' + app[:-4] + '.' + figtype)
        return
        # }}}

    def plot_parameterbias_vs_single_parameter(self,
                                               logy=False,
                                               parameter='chieff',
                                               massmid=70.,
                                               OneMinus=False,
                                               ChiNormalized=True,
                                               ApproxList=[],
                                               biasparameter='ChirpMass',
                                               ylabel='',
                                               ylims=[],
                                               onlyimr=True,
                                               bounds=None,
                                               azimuthal=30,
                                               alpha=0.8,
                                               figtype='pdf'):
        # {{{
        print("trying to make pb vs p plot")
        try:
            from pycbc.pnutils import mtotal_eta_to_mass1_mass2
        except:
            print("error import PyCBC modules / matplotlib")
            return
        #
        # which approximants to plot ?
        if type(ApproxList) == list and len(ApproxList) != 0:
            approx_present = True
            for app in ApproxList:
                if app not in self.ApproxList:
                    approx_present = False
            if not approx_present:
                ApproxList = self.ApproxList
        else:
            ApproxList = self.ApproxList
        #
        # Set the chi-effective flag
        if 'ChiEff' in biasparameter:
            chieffflag = True
        else:
            chieffflag = False
        #
        # Set the total-mass flag
        if 'TotalMass' in biasparameter:
            mtotalflag = True
        else:
            mtotalflag = False
        #
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
        #                    np.array([]), np.array([]), np.array([]), np.array([])
        titles = []
        for idx, app in enumerate(ApproxList):
            if onlyimr:
                if 'Taylor' in app:
                    continue
            else:
                if 'Taylor' not in app:
                    continue
            print("making pb vs p plot for %s" % app)
            #################################################
            # q = 1
            inkey = 'q1'
            inq = 1.
            ineta = inq / (1. + inq)**2
            nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {}, {}, {}, {}, {}
            nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                np.array([])
            #
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff = \
                    self.data.parameterbiases_vs_parameters(inkey=sim,
                                                            approx=app, chieff=chieffflag, total_mass=mtotalflag)
                # Here replace ff with the pb (parameter bias)
                if 'ChirpMass' in biasparameter or 'TotalMass' in biasparameter:
                    ff = mc_diff
                elif 'ChiEff' in biasparameter:
                    ff = s1_diff
                elif 'Eta' in biasparameter:
                    ff = eta_diff
                #
                idx['high'] = np.where(ff == ff.max())[0][0]
                idx['low'] = np.where(ff == ff.min())[0][0]
                if massmid > 0:
                    idx['mid'] = np.where(masses == massmid)[0][0]
                else:
                    idx['mid'] = np.where(masses == masses.min())[0][0]
                for kk in ['low', 'mid', 'high']:
                    nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
                    nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
                    nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
                    nr_ff[kk] = np.append(nr_ff[kk], ff[idx[kk]])
            nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            nr_mass1q1, nr_mass2q1, nr_chieffq1 = {}, {}, {}
            for kk in ['low', 'mid', 'high']:
                nr_mass1q1[kk], nr_mass2q1[kk] = \
                    mtotal_eta_to_mass1_mass2(nr_masses[kk],
                                              ineta * np.ones(len(nr_masses[kk])))
                nr_chieffq1[kk] = spins_to_PNeffective_spin(
                    nr_mass1q1[kk], nr_mass2q1[kk], nr_spin1zq1[kk],
                    nr_spin2zq1[kk])
            #################################################
            # q = 2
            inkey = 'q2'
            inq = 2.
            ineta = inq / (1. + inq)**2
            nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {}, {}, {}, {}, {}
            nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                np.array([])
            #
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff = \
                    self.data.parameterbiases_vs_parameters(inkey=sim,
                                                            approx=app, chieff=chieffflag, total_mass=mtotalflag)
                # Here replace ff with the pb (parameter bias)
                if 'ChirpMass' in biasparameter or 'TotalMass' in biasparameter:
                    ff = mc_diff
                elif 'ChiEff' in biasparameter:
                    ff = s1_diff
                elif 'Eta' in biasparameter:
                    ff = eta_diff
                #
                idx['high'] = np.where(ff == ff.max())[0][0]
                idx['low'] = np.where(ff == ff.min())[0][0]
                print("PK: masses = ", masses)
                if massmid > 0:
                    idx['mid'] = np.where(masses == massmid)[0][0]
                else:
                    idx['mid'] = np.where(masses == masses.min())[0][0]
                for kk in ['low', 'mid', 'high']:
                    nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
                    nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
                    nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
                    nr_ff[kk] = np.append(nr_ff[kk], ff[idx[kk]])
            nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            nr_mass1q2, nr_mass2q2, nr_chieffq2 = {}, {}, {}
            for kk in ['low', 'mid', 'high']:
                nr_mass1q2[kk], nr_mass2q2[kk] = \
                    mtotal_eta_to_mass1_mass2(nr_masses[kk],
                                              ineta * np.ones(len(nr_masses[kk])))
                nr_chieffq2[kk] = spins_to_PNeffective_spin(
                    nr_mass1q2[kk], nr_mass2q2[kk], nr_spin1zq2[kk],
                    nr_spin2zq2[kk])
            #################################################
            # q = 3
            inkey = 'q3'
            inq = 3.
            ineta = inq / (1. + inq)**2
            nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {}, {}, {}, {}, {}
            nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                np.array([])
            #
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff = \
                    self.data.parameterbiases_vs_parameters(inkey=sim,
                                                            approx=app, chieff=chieffflag, total_mass=mtotalflag)
                # Here replace ff with the pb (parameter bias)
                if 'ChirpMass' in biasparameter or 'TotalMass' in biasparameter:
                    ff = mc_diff
                elif 'ChiEff' in biasparameter:
                    ff = s1_diff
                elif 'Eta' in biasparameter:
                    ff = eta_diff
                #
                idx['high'] = np.where(ff == ff.max())[0][0]
                idx['low'] = np.where(ff == ff.min())[0][0]
                if massmid > 0:
                    idx['mid'] = np.where(masses == massmid)[0][0]
                else:
                    idx['mid'] = np.where(masses == masses.min())[0][0]
                for kk in ['low', 'mid', 'high']:
                    nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
                    nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
                    nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
                    nr_ff[kk] = np.append(nr_ff[kk], ff[idx[kk]])
            nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            nr_mass1q3, nr_mass2q3, nr_chieffq3 = {}, {}, {}
            for kk in ['low', 'mid', 'high']:
                nr_mass1q3[kk], nr_mass2q3[kk] = \
                    mtotal_eta_to_mass1_mass2(nr_masses[kk],
                                              ineta * np.ones(len(nr_masses[kk])))
                nr_chieffq3[kk] = spins_to_PNeffective_spin(
                    nr_mass1q3[kk], nr_mass2q3[kk], nr_spin1zq3[kk],
                    nr_spin2zq3[kk])
            #################################################
            #
            Yrow = [nr_ffq1['mid'], nr_ffq2['mid'], nr_ffq3['mid']]
            Yerror = [
                np.append([nr_ffq1['mid'] - nr_ffq1['low']],
                          [nr_ffq1['high'] - nr_ffq1['mid']],
                          axis=0),
                np.append([nr_ffq2['mid'] - nr_ffq2['low']],
                          [nr_ffq2['high'] - nr_ffq2['mid']],
                          axis=0),
                np.append([nr_ffq3['mid'] - nr_ffq3['low']],
                          [nr_ffq3['high'] - nr_ffq3['mid']],
                          axis=0)
            ]
            if ChiNormalized:
                etaq1, etaq2, etaq3 = 1. / 4., 2. / 9., 3. / 16.
                Xrow = [
                    nr_chieffq1['mid'] / (1. - 76. * etaq1 / 113.),
                    nr_chieffq2['mid'] / (1. - 76. * etaq2 / 113.),
                    nr_chieffq3['mid'] / (1. - 76. * etaq3 / 113.)
                ]
                if chieffflag:
                    Yrow = [
                        Yrow[0] / (1. - 76. * etaq1 / 113.),
                        Yrow[1] / (1. - 76. * etaq2 / 113.),
                        Yrow[2] / (1. - 76. * etaq3 / 113.)
                    ]
                    Yerror = [
                        Yerror[0] / (1. - 76. * etaq1 / 113.),
                        Yerror[1] / (1. - 76. * etaq2 / 113.),
                        Yerror[2] / (1. - 76. * etaq3 / 113.)
                    ]
            else:
                Xrow = [
                    nr_chieffq1['mid'], nr_chieffq2['mid'], nr_chieffq3['mid']
                ]
            Xerror = None
            #
            ###################################
            # Now make the plots
            if OneMinus:
                nameprefix = ''
            else:
                ymin, ymax = None, None
                if len(ylims) == 2:
                    ymin, ymax = ylims
                #ymin, ymax = np.array(Yrow).min(), np.array(Yrow).max()
                #ylabel = 'Fitting Factor'
                logy = False
                legendplacement = 'best'
                nameprefix = biasparameter + 'Bias_'
            if ChiNormalized:
                xlabel = '$\chi_\mathrm{eff}/(1-\\frac{76}{113}\eta)$'
            else:
                xlabel = '$\chi_\mathrm{eff}$'
            print(ymin, ymax, ylabel, logy, legendplacement, nameprefix)
            make_2Dplot_errorbars(Xrow,
                                  Yrow,
                                  Xerrs=Xerror,
                                  Yerrs=Yerror,
                                  xlabel=xlabel,
                                  ylabel=ylabel,
                                  title=app[:-4],
                                  logy=logy,
                                  ymin=ymin,
                                  ymax=ymax,
                                  labels=['q=1', 'q=2', 'q=3'],
                                  legendplacement=legendplacement,
                                  savefig=self.plotdir + '/' + nameprefix +
                                  'vsChiEff_' + app[:-4] + '.' + figtype)
        return
        # }}}

    def plot_recoveredparameter_vs_single_parameter(self,
                                                    logy=False,
                                                    parameter='chieff',
                                                    massmid=70.,
                                                    OneMinus=False,
                                                    ChiNormalized=True,
                                                    ApproxList=[],
                                                    biasparameter='ChirpMass',
                                                    ylabel='',
                                                    ylims=[],
                                                    onlyimr=True,
                                                    bounds=None,
                                                    azimuthal=30,
                                                    alpha=0.8,
                                                    figtype='pdf'):
        # {{{
        try:
            from pycbc.pnutils import mtotal_eta_to_mass1_mass2
        except:
            print("error import PyCBC modules / matplotlib")
            return
        #
        # which approximants to plot ?
        if type(ApproxList) == list and len(ApproxList) != 0:
            approx_present = True
            for app in ApproxList:
                if app not in self.ApproxList:
                    approx_present = False
            if not approx_present:
                ApproxList = self.ApproxList
        else:
            ApproxList = self.ApproxList
        #
        # Set the chi-effective flag
        if 'ChiEff' in biasparameter:
            chieffflag = True
        else:
            chieffflag = False
        #
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
        #                    np.array([]), np.array([]), np.array([]), np.array([])
        titles = []
        for idx, app in enumerate(ApproxList):
            if onlyimr:
                if 'Taylor' in app:
                    continue
            else:
                if 'Taylor' not in app:
                    continue
            #################################################
            # q = 1
            inkey = 'q1'
            inq = 1.
            ineta = inq / (1. + inq)**2
            nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {}, {}, {}, {}, {}
            nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                np.array([])
            #
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff = \
                    self.data.parameterbiases_vs_parameters(inkey=sim,
                                                            approx=app, chieff=chieffflag)
                # Here replace ff with the pb (parameter bias)
                if 'ChirpMass' in biasparameter:
                    inj_mc = masses * (nr_q / (1. + nr_q)**2)
                    ff = inj_mc * (1. + mc_diff)
                elif 'ChiEff' in biasparameter:
                    ff = nr_s1 + s1_diff
                elif 'Eta' in biasparameter or 'Q' in biasparameter:
                    inj_et = nr_q / (1. + nr_q)**2
                    ff = inj_et * (1. + eta_diff)
                    ff = eta_to_q(ff)
                #
                idx['high'] = np.where(ff == ff.max())[0][0]
                idx['low'] = np.where(ff == ff.min())[0][0]
                if massmid > 0:
                    idx['mid'] = np.where(masses == massmid)[0][0]
                else:
                    idx['mid'] = np.where(masses == masses.min())[0][0]
                for kk in ['low', 'mid', 'high']:
                    nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
                    nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
                    nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
                    nr_ff[kk] = np.append(nr_ff[kk], ff[idx[kk]])
            nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            nr_mass1q1, nr_mass2q1, nr_chieffq1 = {}, {}, {}
            for kk in ['low', 'mid', 'high']:
                nr_mass1q1[kk], nr_mass2q1[kk] = \
                    mtotal_eta_to_mass1_mass2(nr_masses[kk],
                                              ineta * np.ones(len(nr_masses[kk])))
                nr_chieffq1[kk] = spins_to_PNeffective_spin(
                    nr_mass1q1[kk], nr_mass2q1[kk], nr_spin1zq1[kk],
                    nr_spin2zq1[kk])
            #################################################
            # q = 2
            inkey = 'q2'
            inq = 2.
            ineta = inq / (1. + inq)**2
            nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {}, {}, {}, {}, {}
            nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                np.array([])
            #
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff = \
                    self.data.parameterbiases_vs_parameters(inkey=sim,
                                                            approx=app, chieff=chieffflag)
                # Here replace ff with the pb (parameter bias)
                if 'ChirpMass' in biasparameter:
                    inj_mc = masses * (nr_q / (1. + nr_q)**2)
                    ff = inj_mc * (1. + mc_diff)
                elif 'ChiEff' in biasparameter:
                    ff = nr_s1 + s1_diff
                elif 'Eta' in biasparameter or 'Q' in biasparameter:
                    inj_et = nr_q / (1. + nr_q)**2
                    ff = inj_et * (1. + eta_diff)
                    ff = eta_to_q(ff)
                #
                idx['high'] = np.where(ff == ff.max())[0][0]
                idx['low'] = np.where(ff == ff.min())[0][0]
                print("PK: masses = ", masses)
                if massmid > 0:
                    idx['mid'] = np.where(masses == massmid)[0][0]
                else:
                    idx['mid'] = np.where(masses == masses.min())[0][0]
                for kk in ['low', 'mid', 'high']:
                    nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
                    nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
                    nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
                    nr_ff[kk] = np.append(nr_ff[kk], ff[idx[kk]])
            nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            nr_mass1q2, nr_mass2q2, nr_chieffq2 = {}, {}, {}
            for kk in ['low', 'mid', 'high']:
                nr_mass1q2[kk], nr_mass2q2[kk] = \
                    mtotal_eta_to_mass1_mass2(nr_masses[kk],
                                              ineta * np.ones(len(nr_masses[kk])))
                nr_chieffq2[kk] = spins_to_PNeffective_spin(
                    nr_mass1q2[kk], nr_mass2q2[kk], nr_spin1zq2[kk],
                    nr_spin2zq2[kk])
            #################################################
            # q = 3
            inkey = 'q3'
            inq = 3.
            ineta = inq / (1. + inq)**2
            nr_masses, nr_spin1z, nr_spin2z, nr_ff, idx = {}, {}, {}, {}, {}
            nr_masses['low'], nr_masses['mid'], nr_masses['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin1z['low'], nr_spin1z['mid'], nr_spin1z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_spin2z['low'], nr_spin2z['mid'], nr_spin2z['high'] = np.array([]),\
                np.array([]), np.array([])
            nr_ff['low'], nr_ff['mid'], nr_ff['high'] = np.array([]), np.array([]),\
                np.array([])
            #
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff = \
                    self.data.parameterbiases_vs_parameters(inkey=sim,
                                                            approx=app, chieff=chieffflag)
                # Here replace ff with the pb (parameter bias)
                if 'ChirpMass' in biasparameter:
                    inj_mc = masses * (nr_q / (1. + nr_q)**2)
                    ff = inj_mc * (1. + mc_diff)
                elif 'ChiEff' in biasparameter:
                    ff = nr_s1 + s1_diff
                elif 'Eta' in biasparameter or 'Q' in biasparameter:
                    inj_et = nr_q / (1. + nr_q)**2
                    ff = inj_et * (1. + eta_diff)
                    ff = eta_to_q(ff)
                #
                idx['high'] = np.where(ff == ff.max())[0][0]
                idx['low'] = np.where(ff == ff.min())[0][0]
                if massmid > 0:
                    idx['mid'] = np.where(masses == massmid)[0][0]
                else:
                    idx['mid'] = np.where(masses == masses.min())[0][0]
                for kk in ['low', 'mid', 'high']:
                    nr_masses[kk] = np.append(nr_masses[kk], masses[idx[kk]])
                    nr_spin1z[kk] = np.append(nr_spin1z[kk], nr_s1[idx[kk]])
                    nr_spin2z[kk] = np.append(nr_spin2z[kk], nr_s2[idx[kk]])
                    nr_ff[kk] = np.append(nr_ff[kk], ff[idx[kk]])
            nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            nr_mass1q3, nr_mass2q3, nr_chieffq3 = {}, {}, {}
            for kk in ['low', 'mid', 'high']:
                nr_mass1q3[kk], nr_mass2q3[kk] = \
                    mtotal_eta_to_mass1_mass2(nr_masses[kk],
                                              ineta * np.ones(len(nr_masses[kk])))
                nr_chieffq3[kk] = spins_to_PNeffective_spin(
                    nr_mass1q3[kk], nr_mass2q3[kk], nr_spin1zq3[kk],
                    nr_spin2zq3[kk])
            #################################################
            #
            Yrow = [nr_ffq1['mid'], nr_ffq2['mid'], nr_ffq3['mid']]
            Yerror = [
                np.append([nr_ffq1['mid'] - nr_ffq1['low']],
                          [nr_ffq1['high'] - nr_ffq1['mid']],
                          axis=0),
                np.append([nr_ffq2['mid'] - nr_ffq2['low']],
                          [nr_ffq2['high'] - nr_ffq2['mid']],
                          axis=0),
                np.append([nr_ffq3['mid'] - nr_ffq3['low']],
                          [nr_ffq3['high'] - nr_ffq3['mid']],
                          axis=0)
            ]
            if ChiNormalized:
                etaq1, etaq2, etaq3 = 1. / 4., 2. / 9., 3. / 16.
                Xrow = [
                    nr_chieffq1['mid'] / (1. - 76. * etaq1 / 113.),
                    nr_chieffq2['mid'] / (1. - 76. * etaq2 / 113.),
                    nr_chieffq3['mid'] / (1. - 76. * etaq3 / 113.)
                ]
                if chieffflag:
                    Yrow = [
                        Yrow[0] / (1. - 76. * etaq1 / 113.),
                        Yrow[1] / (1. - 76. * etaq2 / 113.),
                        Yrow[2] / (1. - 76. * etaq3 / 113.)
                    ]
                    Yerror = [
                        Yerror[0] / (1. - 76. * etaq1 / 113.),
                        Yerror[1] / (1. - 76. * etaq2 / 113.),
                        Yerror[2] / (1. - 76. * etaq3 / 113.)
                    ]
            else:
                Xrow = [
                    nr_chieffq1['mid'], nr_chieffq2['mid'], nr_chieffq3['mid']
                ]
            Xerror = None
            #
            ###################################
            # Now make the plots
            if OneMinus:
                pass
            else:
                ymin, ymax = None, None
                if len(ylims) == 2:
                    ymin, ymax = ylims
                #ymin, ymax = np.array(Yrow).min(), np.array(Yrow).max()
                #ylabel = 'Fitting Factor'
                logy = False
                legendplacement = 'best'
                nameprefix = biasparameter + 'Rec_'
            if ChiNormalized:
                xlabel = '$\chi_\mathrm{eff}/(1-\\frac{76}{113}\eta)$'
            else:
                xlabel = '$\chi_\mathrm{eff}$'
            print(ymin, ymax, ylabel, logy, legendplacement, nameprefix)
            make_2Dplot_errorbars(Xrow,
                                  Yrow,
                                  Xerrs=Xerror,
                                  Yerrs=Yerror,
                                  xlabel=xlabel,
                                  ylabel=ylabel,
                                  title=app[:-4],
                                  logy=logy,
                                  ymin=ymin,
                                  ymax=ymax,
                                  labels=['q=1', 'q=2', 'q=3'],
                                  legendplacement=legendplacement,
                                  savefig=self.plotdir + '/' + nameprefix +
                                  'vsChiEff_' + app[:-4] + '.' + figtype)
        return
        # }}}

    def plot_recoveredparameters_vs_parameters_multrow(self,
                                                       logy=True,
                                                       elevation=30,
                                                       ApproxList=[],
                                                       onlyimr=True,
                                                       chieff=False,
                                                       bounds=True,
                                                       azimuthal=30,
                                                       alpha=0.8,
                                                       figtype='pdf'):
        # {{{
        try:
            pass
        except:
            return
        #
        # which approximants to plot ?
        if type(ApproxList) == list and len(ApproxList) != 0:
            approx_present = True
            for app in ApproxList:
                if app not in self.ApproxList:
                    approx_present = False
            if not approx_present:
                ApproxList = self.ApproxList
        else:
            ApproxList = self.ApproxList
        ###############################
        # Read data
        if self.data == None:
            self.read_data_from_all_files()
        all_sims = self.data.data.keys()
        #nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_ff = np.array([]),\
        #                    np.array([]), np.array([]), np.array([]), np.array([])
        Xs, Ys, Zs, mcCs, etCs, qCs, s1Cs, s2Cs = [], [], [], [], [], [], [], []
        titles = []
        for idx, app in enumerate(ApproxList):
            if onlyimr:
                if 'Taylor' in app:
                    continue
            # q = 1
            inkey = 'q1'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
                nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                    self.data.parameterbiases_vs_parameters(inkey=sim,
                                                            chieff=chieff, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_mcdiff = np.append(nr_mcdiff, mc_diff)
                nr_etdiff = np.append(nr_etdiff, eta_diff)
                nr_s1diff = np.append(nr_s1diff, s1_diff)
                nr_s2diff = np.append(nr_s2diff, s2_diff)
                nr_ff = np.append(nr_ff, ff)
            nr_mcdiffq1, nr_etdiffq1, nr_s1diffq1, nr_s2diffq1 = nr_mcdiff,\
                nr_etdiff, nr_s1diff, nr_s2diff
            nr_spin1zq1, nr_spin2zq1, nr_massesq1, nr_ffq1 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            # q = 2
            inkey = 'q2'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
                nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                    self.data.parameterbiases_vs_parameters(inkey=sim,
                                                            chieff=chieff, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_mcdiff = np.append(nr_mcdiff, mc_diff)
                nr_etdiff = np.append(nr_etdiff, eta_diff)
                nr_s1diff = np.append(nr_s1diff, s1_diff)
                nr_s2diff = np.append(nr_s2diff, s2_diff)
                nr_ff = np.append(nr_ff, ff)
            nr_mcdiffq2, nr_etdiffq2, nr_s1diffq2, nr_s2diffq2 = nr_mcdiff,\
                nr_etdiff, nr_s1diff, nr_s2diff
            nr_spin1zq2, nr_spin2zq2, nr_massesq2, nr_ffq2 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            # q = 3
            inkey = 'q3'
            nr_masses, nr_massratios, nr_spin1z, nr_spin2z, nr_mcdiff, nr_etdiff,\
                nr_s1diff, nr_s2diff, nr_ff = np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([]),\
                np.array([]), np.array([]), np.array([]), np.array([])
            for sim in all_sims:
                if inkey is not None and inkey not in str(sim):
                    continue
                masses, nr_q, nr_s1, nr_s2, mc_diff, eta_diff, s1_diff, s2_diff, ff =\
                    self.data.parameterbiases_vs_parameters(inkey=sim,
                                                            chieff=chieff, approx=app)
                nr_masses = np.append(nr_masses, masses)
                nr_massratios = np.append(nr_massratios, nr_q)
                nr_spin1z = np.append(nr_spin1z, nr_s1)
                nr_spin2z = np.append(nr_spin2z, nr_s2)
                nr_mcdiff = np.append(nr_mcdiff, mc_diff)
                nr_etdiff = np.append(nr_etdiff, eta_diff)
                nr_s1diff = np.append(nr_s1diff, s1_diff)
                nr_s2diff = np.append(nr_s2diff, s2_diff)
                nr_ff = np.append(nr_ff, ff)
            nr_mcdiffq3, nr_etdiffq3, nr_s1diffq3, nr_s2diffq3 = nr_mcdiff,\
                nr_etdiff, nr_s1diff, nr_s2diff
            nr_spin1zq3, nr_spin2zq3, nr_massesq3, nr_ffq3 = nr_spin1z, nr_spin2z,\
                nr_masses, nr_ff
            #
            Xrow = [nr_spin1zq1, nr_spin1zq2, nr_spin1zq3]
            Yrow = [nr_spin2zq1, nr_spin2zq2, nr_spin2zq3]
            Zrow = [nr_massesq1, nr_massesq2, nr_massesq3]
            mcCrow = [nr_mcdiffq1, nr_mcdiffq2, nr_mcdiffq3]
            etCrow = [nr_etdiffq1, nr_etdiffq2, nr_etdiffq3]
            s1Crow = [nr_s1diffq1, nr_s1diffq2, nr_s1diffq3]
            s2Crow = [nr_s2diffq1, nr_s2diffq2, nr_s2diffq3]

            etaq1, etaq2, etaq3 = 1. / 4., 2. / 9., 3. / 16.
            inj_mc = [
                nr_massesq1 * etaq1**0.6, nr_massesq2 * etaq2**0.6,
                nr_massesq3 * etaq3**0.6
            ]
            rec_mc = []
            for i in range(3):
                rec_mc.append(inj_mc[i] * (1. + mcCrow[i]))

            rec_et = []
            inj_et = [etaq1, etaq2, etaq3]
            for i in range(3):
                rec_et.append(inj_et[i] * (1. + etCrow[i]))

            rec_q = eta_to_q(rec_et)

            rec_s1 = []
            for i in range(3):
                rec_s1.append(Xrow[i] + s1Crow[i])

            rec_s2 = []
            for i in range(3):
                rec_s2.append(Yrow[i] + s2Crow[i])

            newmcCrow = rec_mc
            newetCrow = rec_et
            news1Crow = rec_s1
            news2Crow = rec_s2
            newqCrow = rec_q

            #
            Xs.append(Xrow)
            Ys.append(Yrow)
            Zs.append(Zrow)
            mcCs.append(newmcCrow)
            etCs.append(newetCrow)
            qCs.append(newqCrow)
            s1Cs.append(news1Crow)
            s2Cs.append(news2Crow)
            #
            titles.append(app[:-4])
        ###################################
        # Now make the plots
        # print "With ", nr_spin1z, nr_spin2z, nr_masses, nr_ff
        if type(bounds) != np.ndarray and type(bounds) != list:
            if bounds:
                print("SHOULD NOT HAPPEN")
                bounds = np.array(
                    [-0.05, -0.03, -0.01, -0.005, 0.005, 0.01, 0.03, 0.05])
        print("bounds before calling plotting function  = ", bounds)
        make_scatter_plot3D_multrow(
            Xs,
            Ys,
            Zs,
            mcCs,
            elevation=elevation,
            azimuthal=azimuthal,
            alpha=alpha,
            xlabel='$\chi_1$',
            ylabel='$\chi_2$',
            zlabel='Total Mass $(M_\odot)$',
            clabel='Recovered $\mathcal{M}_c$',
            title=titles,
            logC=False,
            bounds=bounds,
            savefig=self.plotdir + '/ChirpMassRec_TotalMass_Spin1z_Spin2z.%s' %
            (figtype))
        #
        if type(bounds) != np.ndarray and type(bounds) != list:
            if bounds:
                print("SHOULD NOT HAPPEN")
                bounds = np.array([
                    -0.5, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, 0.5
                ])
        make_scatter_plot3D_multrow(Xs,
                                    Ys,
                                    Zs,
                                    etCs,
                                    elevation=elevation,
                                    azimuthal=azimuthal,
                                    alpha=alpha,
                                    xlabel='$\chi_1$',
                                    ylabel='$\chi_2$',
                                    zlabel='Total Mass $(M_\odot)$',
                                    clabel='Recovered $\eta$',
                                    title=titles,
                                    logC=False,
                                    bounds=bounds,
                                    savefig=self.plotdir +
                                    '/EtaRec_TotalMass_Spin1z_Spin2z.%s' %
                                    (figtype))
        #
        if type(bounds) != np.ndarray and type(bounds) != list:
            if bounds:
                print("SHOULD NOT HAPPEN")
                bounds = np.array([
                    -0.5, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, 0.5
                ])
        make_scatter_plot3D_multrow(Xs,
                                    Ys,
                                    Zs,
                                    qCs,
                                    elevation=elevation,
                                    azimuthal=azimuthal,
                                    alpha=alpha,
                                    xlabel='$\chi_1$',
                                    ylabel='$\chi_2$',
                                    zlabel='Total Mass $(M_\odot)$',
                                    clabel='Recovered $q$',
                                    title=titles,
                                    logC=False,
                                    bounds=bounds,
                                    savefig=self.plotdir +
                                    '/QRec_TotalMass_Spin1z_Spin2z.%s' %
                                    (figtype))
        #
        if type(bounds) != np.ndarray and type(bounds) != list:
            if bounds:
                print("SHOULD NOT HAPPEN")
                bounds = np.array([
                    -0.5, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, 0.5
                ])
        if chieff:
            clabel = 'Recovered $\chi_\mathrm{eff}$'
            figtag = 'ChiEff'
        else:
            clabel = 'Recovered $\chi_1$'
            figtag = 'Chi1'
        make_scatter_plot3D_multrow(Xs, Ys, Zs, s1Cs,
                                    elevation=elevation, azimuthal=azimuthal, alpha=alpha,
                                    xlabel='$\chi_1$', ylabel='$\chi_2$',
                                    zlabel='Total Mass $(M_\odot)$',
                                    clabel=clabel, title=titles,\
                                    # clabel='$\Delta\chi_\mathrm{eff2PN}$', title=titles,\
                                    # clabel='$\Delta\chi_\mathrm{eff}$', title=titles,\
                                    # clabel='$\Delta\chi_\mathrm{mw}$', title=titles,\
                                    # clabel='$\Delta\chi_\mathrm{effD}$', title=titles,\
                                    logC=False,\
                                    bounds=bounds,\
                                    savefig=self.plotdir+'/'+figtag+'Rec_TotalMass_Spin1z_Spin2z.%s'\
                                    % (figtype))
        #
        if type(bounds) != np.ndarray and type(bounds) != list:
            if bounds:
                print("SHOULD NOT HAPPEN")
                bounds = np.array([
                    -0.5, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, 0.5
                ])
        make_scatter_plot3D_multrow(Xs,
                                    Ys,
                                    Zs,
                                    s2Cs,
                                    elevation=elevation,
                                    azimuthal=azimuthal,
                                    alpha=alpha,
                                    xlabel='$\chi_1$',
                                    ylabel='$\chi_2$',
                                    zlabel='Total Mass $(M_\odot)$',
                                    clabel='Recovered $\chi_2$',
                                    title=titles,
                                    logC=False,
                                    bounds=bounds,
                                    savefig=self.plotdir +
                                    '/Chi2Rec_TotalMass_Spin1z_Spin2z.%s' %
                                    (figtype))
        #
        return
        # }}}

    # }}}
