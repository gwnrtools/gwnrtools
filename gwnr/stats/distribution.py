# Copyright (C) 2017 Prayush Kumar
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
import logging
from gwnr.graph.cbc import ParamLatexLabels
from scipy.stats.kde import gaussian_kde
from scipy.interpolate import UnivariateSpline

import scipy.integrate as si
import scipy.optimize as so

try:
    from statsmodels.nonparametric.kde import KDEUnivariate
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
except:
    pass

import os
import copy as cp
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt

plt.rcParams.update({'text.usetex': True})

logging.getLogger().setLevel(logging.INFO)

######################################################
# Constants

quantiles_68 = [0.16, 0.5, 0.84]  # 1-sigma
quantiles_95 = [0.0228, 0.5, 0.9772]  # 2-sigma ~ 95.4%
percentiles_68 = 100 * np.array(quantiles_68)
percentiles_95 = 100 * np.array(quantiles_95)
perc_int_95 = [2.28, 97.72]
perc_int_99_7 = [0.13, 99.87]  # 3 sigma

######################################################
# Function to make parameter bias plots

linestyles = ['-', '--', '-.', '-x', '--o']
linecolors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
plotdir = 'plots/'
gmean = (5**0.5 + 1) / 2.

# Figure settings
ppi = 72.0
aspect = (5.**0.5 - 1) * 0.5
size = 4.0 * 2  # was 6
figsize = (size, aspect * size)

######################################################################
#     UTILITIES FOR 1D / ND DISTRIBUTIONS


def KDEMeanOverRange(kde_func, x_range):
    total_area = si.quad(kde_func, low, high)
    low = np.min(x_range)
    high = np.max(x_range)

    def give_integrand(xval):
        return kde_func(xval) * xval

    return si.quad(give_integrand, low, high) / total_area


def KDEMedianOverRange(kde_func, x_range):
    low = np.min(x_range)
    high = np.max(x_range)

    def give_pdf(xval):
        return -1.0 * kde_func(xval)

    return so.minimize(give_pdf, H0).x[0]


#######################################################
# CONTAINER and CALCULATION CLASSES


class OneDDistribution:
    '''
DESCRIPTION:
Here we operate on 1-D distributions:

-- compute mean, median of multi-d distributions
-- compute mean, median of 1-d marginalized distributions
-- compute KDE
-- compute normalization

INPUTS:
data - 1-D iterable type (list, array)
data_kde - if provided, must be a KDE estimator class with a member
            function called "evaluate". E.g. use KDEUnivariate class from
            [] from statsmodels.nonparametric.kde import KDEUnivariate
    '''
    def __init__(self,
                 data,
                 data_kde=None,
                 kernel='gau',
                 bw_method='scott',
                 kernel_cut=3.0,
                 xlimits=None,
                 verbose=False,
                 debug=False):
        if debug:
            logging.info("Initializing OneDDistribution object..")
        self.input_data = np.array(data)
        self.bw_method = bw_method
        self.kernel = kernel
        self.kernel_cut = kernel_cut
        if xlimits:
            self.xllimit, self.xulimit = xlimits
        else:
            self.xllimit, self.xulimit = min(self.input_data), max(
                self.input_data)
        if data_kde != None:
            self.kde = data_kde
            logging.warn(
                """WARNING: Be careful when providing a kernel density estimator directly.
                              We assume, but not check, that it matches the input sample set.."""
            )
        self.verbose = verbose
        self.debug = debug
        return

    def data(self):
        return self.input_data

    def xlimits(self):
        return [self.xllimit, self.xulimit]

    def normalization(self, xllimit=None, xulimit=None):
        if hasattr(self, "norm"):
            return self.norm
        if self.verbose:
            logging.info("NORMALIZING 1D KDE")
        input_kde_func = self.kde()
        if xllimit == None:
            xllimit, _ = self.xlimits()
        if xulimit == None:
            _, xulimit = self.xlimits()
        input_data_norm = si.quad(input_kde_func,
                                  xllimit,
                                  xulimit,
                                  epsabs=1.e-16,
                                  epsrel=1.e-16)[0]
        self.norm = input_data_norm
        return input_data_norm

    def mean(self, xllimit=None, xulimit=None):
        tmp_data = self.input_data
        if xllimit is not None:
            tmp_data = tmp_data[tmp_data >= xllimit]
        if xulimit is not None:
            tmp_data = tmp_data[tmp_data >= xllimit]
        return np.mean(tmp_data)

    def median(self, xllimit=None, xulimit=None):
        tmp_data = self.input_data
        if xllimit is not None:
            tmp_data = tmp_data[tmp_data >= xllimit]
        if xulimit is not None:
            tmp_data = tmp_data[tmp_data >= xllimit]
        return np.median(tmp_data)

    def percentile(self, perc, xllimit=None, xulimit=None):
        tmp_data = self.input_data
        if xllimit is not None:
            tmp_data = tmp_data[tmp_data >= xllimit]
        if xulimit is not None:
            tmp_data = tmp_data[tmp_data >= xllimit]
        return np.percentile(tmp_data, perc)

    def kde(self):
        if hasattr(self, "evaluate_kde"):
            return self.evaluate_kde
        if self.verbose:
            logging.info("INITALIZING 1D KDE")
        kde = KDEUnivariate(self.input_data)
        try:
            kde.fit(kernel=self.kernel,
                    bw=self.bw_method,
                    fft=True,
                    cut=self.kernel_cut)
        except:
            kde.fit(kernel=self.kernel,
                    bw=self.bw_method,
                    fft=False,
                    cut=self.kernel_cut)
        self.evaluate_kde = kde.evaluate
        self.kde_object = kde
        return self.evaluate_kde

    def sliced(self, i):
        if i != 0:
            raise RuntimeError(
                "One-D data has only one slice, indexed by 0 (asked for %d)" %
                i)
        return self

    # oneD_

    def pdf_over_range(self, x_range):
        self.kde()
        return self.evaluate_kde(x_range)

    def mean_in_range(self, x_range):
        kde_func = self.kde()
        return KDEMeanOverRange(kde_func, x_range)

    def median_in_range(self, x_range):
        kde_func = self.kde()
        return KDEMeanOverRange(kde_func, x_range)


class MultipleOneDDistributions:
    '''
DESCRIPTION:
Here we operate on multiple n-D distributions:

-- marginalize over 1 parameter
-- marginalize over lots of parameters
-- compute mean, median of multi-d distributions
-- compute mean, median of 1-d marginalized distributions
-- compute KDE
-- ??

INPUTS:
data : >=2 dimensional iterable type (list, array)

var_type: str
    The type of the variables:

        - c : continuous
        - u : unordered (discrete)
        - o : ordered (discrete)

    The string should contain a type specifier for each variable, so for
    example ``var_type='ccuo'``.

data_kde: KDE estimator class object with a member function called "evaluate".
    E.g. use KDEUnivariate class from
            [] from statsmodels.nonparametric.kde import KDEUnivariate

xlimits: iterable of two arrays, one for lower limit and one for uppe
    '''
    def __init__(self,
                 datadir,
                 result_tag,
                 event_ids,
                 var_type,
                 verbose=False):
        ### =======         INPUT CHECKING/HANDLING              ======= ###
        if not os.path.exists(datadir):
            raise IOError("DATA Directory %s does not exist.." % datadir)
        self.datadir = datadir
        self.result_tag = result_tag
        # Event IDs (int) labeling events and their order
        self.event_ids = event_ids
        self.var_type = var_type
        self.data = self.read_distributions()
        self.verbose = verbose

        ### =======         INITLIAZE              ======= ###
        self.event = {}
        self.pdf_event = {}
        self.pdf_norm_event = {}
        self.pdf_cum_events = {}
        ##
        return

    def read_distributions(self):
        datadir, result_tag, event_ids =\
            self.datadir, self.result_tag, self.event_ids
        data = {}
        id_cnt = 0
        for c in result_tag:
            if c == '%':
                id_cnt += 1
        for event_id in event_ids:
            if verbose:
                logging.info("Reading posterior for event ", event_id)
            event_id_pattern = tuple(np.ones(id_cnt) * event_id)
            res_file = os.path.join(datadir, result_tag % event_id_pattern)
            data[event_id] = np.loadtxt(res_file)
        return data

    def process_oned_slices(self, kernel_cut=3.0, reprocess=False):
        for id_cnt, id in enumerate(self.event_ids):
            if id in self.event and not reprocess:
                continue
            if self.verbose:
                logging.info("\n READING EVENT %d" % (id))

            self.event[id] = MultiDDistribution(self.data[id],
                                                self.var_type,
                                                oneD_kernel_cut=kernel_cut)
            self.pdf_norm_event[id] = self.event[id].sliced(0).normalization()
        return self.event

    def combine_oned_slices(self, x_range, prior_func, event_ids=None):
        if len(self.event) == 0:
            logging.info("Please process oneD slices first")
            return
        #
        if event_ids == None:
            event_ids = self.event_ids
        x_range = np.array(x_range)
        self.XRANGE = x_range
        self.PRIORDATA = prior_func(x_range)
        for id_cnt, id in enumerate(event_ids):
            self.pdf_norm_event[id] = self.event[id].sliced(0).normalization(
                xllimit=x_range.min(), xulimit=x_range.max())
            self.pdf_event[id] = self.event[id].sliced(0).kde()(
                x_range) / self.pdf_norm_event[id]
            if id_cnt != 0:
                self.pdf_cum_events[id] = self.pdf_cum_events[event_ids[
                    id_cnt - 1]] * self.pdf_event[id]
                self.pdf_cum_events[id] /= self.PRIORDATA**2
                AREA = np.sum(self.pdf_cum_events[id]) * (
                    x_range.max() - x_range.min()) / len(x_range)
                self.pdf_cum_events[id] /= AREA
            else:
                self.pdf_cum_events[id] = self.pdf_event[id] / \
                    self.pdf_norm_event[id]
        # for id_cnt, id in enumerate(event_ids):
        #    self.pdf_cum_events[id] /= self.PRIORDATA**id_cnt
        #    AREA = np.sum(self.pdf_cum_events[id]) * (x_range.max() - x_range.min()) / len(x_range)
        #    self.pdf_cum_events[id] /= AREA
        return self.pdf_cum_events[id]

    def plot_combined_oned_slices(self,
                                  x_range,
                                  prior_func,
                                  event_ids=None,
                                  labels_per_column=15,
                                  label_every_nth_curve=1,
                                  cum_pdf_ylim=[0, 1.2],
                                  pdf_ylim=[0, 0.22],
                                  cum_pdf_title='CUMULATIVE PDFS',
                                  pdf_title='INDIVIDUAL EVENT PDFS AND KDES',
                                  xlabel='$H_0$'):
        if event_ids == None:
            event_ids = self.event_ids
        self.combine_oned_slices(x_range, prior_func, event_ids=event_ids)
        # MAKE FIGURE
        ax0 = plt.figure(figsize=(8, 6)).add_subplot(111)
        ax2 = plt.figure(figsize=(8, 6)).add_subplot(111)
        #
        label_cnt = 0
        for id_cnt, id in enumerate(event_ids):
            label_txt = None
            if label_every_nth_curve > 1 and ((id_cnt + 1) %
                                              label_every_nth_curve) == 0:
                label_txt = '%d' % (id_cnt + 1)
                label_cnt += 1
            ax0.plot(x_range,
                     self.pdf_cum_events[id],
                     'k',
                     lw=2,
                     alpha=0.1 + id_cnt * 0.7 / len(event_ids),
                     label=label_txt)

            label_txt = None
            if label_every_nth_curve > 1 and ((id_cnt + 1) %
                                              label_every_nth_curve) == 0:
                label_txt = '%d' % (id_cnt + 1)
            ax2.plot(x_range,
                     self.pdf_event[id] / self.pdf_norm_event[id],
                     label=label_txt,
                     color='k',
                     lw=2,
                     alpha=0.1 + id_cnt * 0.7 / len(event_ids))
            _ = ax2.hist(self.event[id].sliced(0).input_data,
                         bins=50,
                         normed=True,
                         alpha=0.03)

            ax0.axvline(self.event[id].sliced(0).mean(),
                        color='g',
                        alpha=0.5,
                        ls='--')
            ax0.axvline([self.event[id].sliced(0).median()],
                        color='r',
                        alpha=0.6,
                        ls='--')
            ax0.axvline(KDEMedianOverRange(
                UnivariateSpline(x_range, self.pdf_cum_events[id]), x_range),
                        color='k',
                        alpha=0.1 + id_cnt * 0.7 / len(event_ids),
                        ls='--')

            ax2.axvline(self.event[id].sliced(0).mean(),
                        color='g',
                        alpha=0.5,
                        ls='--')
            ax2.axvline([self.event[id].sliced(0).median()],
                        color='r',
                        alpha=0.6,
                        ls='--')
        #
        ax0.set_title(cum_pdf_title)
        ax2.set_title(pdf_title)

        ax0.legend(loc="upper left",
                   bbox_to_anchor=(1, 1),
                   ncol=label_cnt / labels_per_column + 1)
        ax2.legend(loc="upper left",
                   bbox_to_anchor=(1, 1),
                   ncol=label_cnt / labels_per_column + 1)

        ax0.set_ylim(cum_pdf_ylim)
        ax2.set_ylim(pdf_ylim)

        # COMMON FORMATTING
        for ax in [ax0, ax2]:
            ax.axvline(H0, color='y', lw=2)
            ax.grid()
            ax.set_xlabel(xlabel)
            ax.set_xlim(x_range.min(), x_range.max())
        return


class MultiDDistribution:
    '''
DESCRIPTION:
Here we operate on n(>=2)-D distributions:

-- marginalize over 1 parameter
-- marginalize over lots of parameters
-- compute mean, median of multi-d distributions
-- compute mean, median of 1-d marginalized distributions
-- compute KDE
-- ??

INPUTS:
data : >=2 dimensional iterable type (list, array). Different dimensions are
in different columns, while a single column should contain samples for that
dimension

var_type: str
    The type of the variables:

        - c : continuous
        - u : unordered (discrete)
        - o : ordered (discrete)

    The string should contain a type specifier for each variable, so for
    example ``var_type='ccuo'``.

data_kde: KDE estimator class object with a member function called "evaluate".
    E.g. use KDEUnivariate class from
            [] from statsmodels.nonparametric.kde import KDEUnivariate

xlimits: iterable of two arrays, one for lower limit and one for upper
    '''
    def __init__(self,
                 data,
                 var_type,
                 var_names=[],
                 data_kde=None,
                 oneD_kernel='gau',
                 oneD_kernel_cut=3.0,
                 oneD_bw_method='normal_reference',
                 mulD_kernel='gau',
                 mulD_bw_method='normal_reference',
                 xlimits=None,
                 verbose=False,
                 debug=False):
        # CHECK INPUTS
        self.verbose = verbose
        self.debug = debug
        if len(np.shape(np.array(data))) == 1:
            logging.info(
                "WARNING: Returning OneDDistribution for 1-D distributions")
            return OneDDistribution(data,
                                    data_kde=data_kde,
                                    kernel=oneD_kernel,
                                    kernel_cut=oneD_kernel_cut,
                                    bw_method=oneD_bw_method,
                                    verbose=verbose,
                                    debug=debug)

        # ENSURE DATA SHAPE, EXTRACT DIMENSION
        if verbose:
            logging.info("..initializing MultiDDistribution object")
        data = np.array(data)

        if len(np.shape(data)) == 1:
            data = data.reshape(np.shape(data)[0], 1)
        elif len(np.shape(data)) == 2:
            if np.shape(data)[0] < np.shape(data)[1]:
                if np.shape(data)[1] != len(var_names):
                    logging.info(
                        "Warning: Found fewer rows than columns in input data. We assume data is in a row-major form."
                    )
                    data = np.transpose(data)
        else:
            raise IOError(
                "Input data has {} dimensions. We only support 1D and 2D data."
                .format(len(np.shape(data))))

        self.input_data = np.array(data)
        self.dim = np.shape(data)[-1]

        if len(var_names) != self.dim:
            var_names = [str(_i) for _i in range(self.dim)]
        self.var_names = var_names

        # PROCESS 1-D SLICES (assuming independently sampled variables)
        if verbose:
            logging.info("...processing 1-D slices")
        self.slices = [
            OneDDistribution(self.sliced(i), kernel_cut=oneD_kernel_cut)
            for i in range(self.dim)
        ]

        # UPPER AND LOWER LIMITS IN 1-D SLICES
        if verbose:
            logging.info("...computing limits in each dimension")
        if xlimits:
            self.xllimit, self.xulimit = xlimits
        else:
            self.xllimit = np.array([min(sl.input_data) for sl in self.slices])
            self.xulimit = np.array([max(sl.input_data) for sl in self.slices])

        # STORE
        self.var_type = var_type
        self.bw_method = mulD_bw_method
        self.kernel = mulD_kernel

        if data_kde:
            self.kde = data_kde
            logging.info(
                """WARNING: Be careful when providing a kernel density estimator directly.
                              We assume, but not check, that it matches the input sample set.."""
            )
        return

    def structured_data(self):
        self.structured_data = cp.deepcopy(self.input_data)

        if self.debug:
            logging.info("Shape of structured_data: ",
                         np.shape(self.structured_data))

        if len(var_names) == np.shape(self.input_data)[-1]:
            self.structured_data = np.array(self.structured_data,
                                            dtype=[(h, '<f8')
                                                   for h in var_names])
        return self.structured_data

    def index_of_name(self, name):
        if name in self.var_names:
            return np.where([n == name for n in self.var_names])[0][0]
        raise IOError("Could not find column variable named: {}".format(name))

    def sliced(self, i):
        """
        This function returns a numpy.array if passed an int,
        or a OneDDistribution if that is available.
        """
        if type(i) == int:
            if hasattr(self, "slices"):
                return self.slices[i]
            else:
                return self.input_data[:, i]
        elif type(i) == str and i in self.var_names:
            return self.slices[self.index_of_name(i)]
        else:
            raise IOError("Could not find slice labelled " + str(i))

    def mean(self, name=None):
        if name in self.var_names:
            return self.sliced(self.index_of_name(name)).mean()
        return np.array([self.sliced(i).mean() for i in range(self.dim)])

    def median(self, name=None):
        if name in self.var_names:
            return self.sliced(self.index_of_name(name)).median()
        return np.array([self.sliced(i).median() for i in range(self.dim)])

    def percentile(self, perc, name=None):
        if name in self.var_names:
            return self.sliced(self.index_of_name(name)).percentile(perc)
        return np.array(
            [self.sliced(i).percentile(perc) for i in range(self.dim)])

    def credible_interval(self, credible_level, name=None):
        perc_int_low = (100.0 - credible_level) * 0.5
        perc_int_high = 100.0 - perc_int_low

        def _get_credible_interval(i):
            return [
                self.sliced(i).percentile(perc_int_low),
                self.sliced(i).percentile(perc_int_high)
            ]

        if name in self.var_names:
            return np.array(_get_credible_interval(self.index_of_name(name)))
        return np.array([_get_credible_interval(i) for i in range(self.dim)])

    def kde(self):
        if hasattr(self, "kde"):
            return self.kde
        kde = KDEMultivariate(self.input_data,
                              var_type=self.var_type,
                              bw=self.bw_method)
        self.kde = kde
        self.evaluate_kde = kde.pdf
        return kde

    def xlimits(self):
        return [self.xllimit, self.xulimit]

    def normalization(self):
        if hasattr(self, "norm"):
            return self.norm
        # ELSE COMPUTE NORM OF MULTI-D DISTRIBUTION
        self.norm = 1.0
        for i in range(self.dim):
            self.norm *= self.sliced(i).normalization()
        return self.norm

    def plot_twod_slice(self, *args):
        raise RuntimeError(
            "This function has been removed. Use gwnr.graph.CornerPlot.")

    def corner_plot(self, *args):
        raise RuntimeError(
            "This function has been removed. Use gwnr.graph.CornerPlot.")

    # }}}
