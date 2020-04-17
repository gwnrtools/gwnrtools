#!/usr/bin/env python
#
# Copyright (C) 2020 Prayush Kumar
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


class HandleSpectreReductionDatum(object):
    def __init__(self, reduction_data_file='', name=''):
        assert(os.path.exists(reduction_data_file),
               "Cannot find data file: {0:s}".format(reduction_data_file))
        self.name = name
        self.reduction_data_file = reduction_data_file
        self.read_data()
        self.plotting_funcs = {'linlin': 'plot',
                               'loglin': 'semilogx',
                               'loglog': 'loglog',
                               'linlog': 'semilogy'}
        self.linestyles = ['-', '--', '--', '-.', ':']
        self.linecolors = ['r', 'g', 'b', 'k', 'm', 'y']

    def read_data(self):
        logging.info("Reading in: {0:s}".format(self.reduction_data_file))
        self.data = h5py.File(self.reduction_data_file, 'r')[
            'element_data.dat'][()]
        logging.info(".. read in a dataset with shape: {}".format(
            np.shape(self.data)))

    def get_data(self):
        return self.data

    def plot(self,
             column_of_vars={2: '$\Pi$', 3: '$\Phi_i$', 4: '$\Psi$'},
             ax=None,
             xy_scales='linlog',
             xlabel='Code time',
             ylabel='L${}^2$ error norms vs analytic soln',
             title=None,
             grid=False,
             legend=True,
             legend_ncol=2):
        assert(len(column_of_vars) > 0, "Nothing asked to plot!")

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        plotting_func = getattr(ax, self.plotting_funcs[xy_scales])

        d = self.get_data()
        logging.info(
            "Plotting {0} of {1} columns".format(len(column_of_vars),
                                                 np.shape(d)[-1]))
        for i, idx in enumerate(column_of_vars):
            logging.info("Plotting {0}".format(column_of_vars[idx]))
            plotting_func(d[:, 0], d[:, idx], lw=2.5 - i * 0.2,
                          ls=self.linestyles[i % len(self.linestyles)],
                          c=self.linecolors[i % len(self.linecolors)],
                          label=column_of_vars[idx] + " ({0:s})".format(self.name))
        if grid:
            plt.grid(True)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if legend:
            ax.legend(loc='best', ncol=legend_ncol)
        return ax


class HandleSpectreReductionData(object):
    def __init__(self, reduction_data_files=[]):
        assert(len(reduction_data_file_name) > 0,
               "Please provide at least one reduction data file!")
        for f in reduction_data_files:
            assert(os.path.exists(f) and os.path.getsize(f) > 0,
                   "Data file: {0:s} not found / is empty!".format(f))
        self.handler = {}
        for f in reduction_data_files:
            self.handler[f] = HandleSpectreReductionDatum(f)

    def handlers(self):
        return self.handler

    def plot(self, **kwargs):
        for f in self.handler:
            self.handler[f].plot(**kwargs)
