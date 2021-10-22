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

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import h5py

try:
    pass
except ImportError:
    pass


class HandleSpectreReductionDatum(object):
    def __init__(self,
                 reduction_data_file='',
                 name='',
                 hdf_path='element_data.dat'):
        assert os.path.exists(reduction_data_file),\
            "Cannot find data file: {0:s}".format(reduction_data_file)
        self.name = name
        self.reduction_data_file = reduction_data_file
        self.hdf_path = hdf_path
        self.read_data()
        self.plotting_funcs = {
            'linlin': 'plot',
            'loglin': 'semilogx',
            'loglog': 'loglog',
            'linlog': 'semilogy'
        }
        self.linestyles = ['-', '--', '--', '-.', ':']
        self.linecolors = ['r', 'g', 'b', 'k', 'm', 'y']

    def read_data(self):
        logging.info("Reading in: {0:s}".format(self.reduction_data_file))
        self.data = h5py.File(self.reduction_data_file, 'r')[self.hdf_path][()]
        logging.info(".. read in a dataset with shape: {}".format(
            np.shape(self.data)))

    def get_data(self):
        return self.data

    def plot(self,
             column_of_vars={
                 2: '$\Pi$',
                 3: '$\Phi_i$',
                 4: '$\Psi$'
             },
             ax=None,
             xy_scales='linlog',
             lw_a=2.5,
             lw_delta=-0.2,
             linewidths=None,
             linestyles=None,
             linecolors=None,
             plot_kwargs={},
             xlabel='Code time',
             ylabel='L${}^2$ error norms vs analytic soln',
             title=None,
             grid=False,
             legend=True,
             legend_kwargs={
                 'ncol': 3,
                 'fontsize': 14,
                 'frameon': False
             }):
        '''
        With `idx` being the index_of_data_column, this function uses:

        - linewidth = lw_a + idx * lw_delta
        - linecolor = list[idx] (if list of colors provided)
                    = str (if single color is provided)
        - linestyle = list[idx] (if list of colors provided)
                    = str (if single color is provided)
        - label     = column_of_vars[idx]
        '''
        assert len(column_of_vars) > 0, "Nothing asked to plot!"

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        plotting_func = getattr(ax, self.plotting_funcs[xy_scales])

        d = self.get_data()
        logging.info("Plotting {0} of {1} columns".format(
            len(column_of_vars),
            np.shape(d)[-1]))
        for i, idx in enumerate(column_of_vars):
            # formatting
            try:
                linewidth = linewidths[i]
            except:
                if type(linewidths) is float:
                    linewidth = linewidths
                else:
                    linewidth = lw_a + i * lw_delta

            if type(linestyles) is str:
                linestyle = linestyles
            else:
                try:
                    linestyle = linestyles[i]
                except:
                    linestyle = self.linestyles[i % len(self.linestyles)]

            if type(linecolors) is str:
                linecolor = linecolors
            else:
                try:
                    linecolor = linecolors[i]
                except:
                    linecolor = self.linecolors[i % len(self.linecolors)]

            label = column_of_vars[idx]
            if self.name:
                label = label + " ({0:s})".format(self.name)

            logging.info("Plotting {0}".format(column_of_vars[idx]))
            plotting_func(d[:, 0],
                          d[:, idx],
                          label=label,
                          lw=linewidth,
                          ls=linestyle,
                          c=linecolor,
                          **plot_kwargs)

        if grid:
            plt.grid(True)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if legend:
            ax.legend(**legend_kwargs)
        return ax


class HandleSpectreReductionData(object):
    def __init__(self, reduction_data_files=[], **args):
        assert len(reduction_data_files) > 0,\
            "Please provide at least one reduction data file!"
        for f in reduction_data_files:
            assert (os.path.exists(f) and os.path.getsize(f) > 0),\
                "Data file: {0:s} not found / is empty!".format(f)
        self.handler = {}
        for f in reduction_data_files:
            self.handler[f] = HandleSpectreReductionDatum(f, **args)

    def handlers(self):
        return self.handler

    def plot(self, **kwargs):
        for f in self.handler:
            self.handler[f].plot(**kwargs)
