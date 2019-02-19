#!/usr/bin/env python
# Copyright (C) 2017 Prayush Kumar
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
from matplotlib import use
#try: use('Agg')
#except: pass
import os, sys, time

import copy as cp
import commands as cmd
import numpy as np
import matplotlib
from matplotlib import mlab, cm, use
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex' : True})
from mpl_toolkits.axes_grid1 import ImageGrid

from pydoc import help
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
import scipy.integrate as si
import scipy.optimize as so
from sklearn.neighbors import KernelDensity

from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from GWNRTools.Stats.LALInferenceUtilities import ParamLatexLabels
import h5py

######################################################
# Constants
######################################################
verbose  = True
vverbose = True
debug    = True

quantiles_68 = [0.16, 0.5, 0.84] # 1-sigma
quantiles_95 = [0.0228, 0.5, 0.9772] # 2-sigma ~ 95.4%
percentiles_68 = 100*np.array(quantiles_68)
percentiles_95 = 100*np.array(quantiles_95)
perc_int_95 = [2.28, 97.72]
perc_int_99_7 = [0.13, 99.87] # 3 sigma

CILevels=[90.0, 68.26895, 95.44997, 99.73002]

######################################################
# Function to make parameter bias plots
######################################################
linestyles = ['-', '--', '-.', '-x', '--o']
linecolors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
plotdir = 'plots/'
gmean = (5**0.5 + 1)/2.

# Figure settings
ppi=72.0
aspect=(5.**0.5 - 1) * 0.5
size=4.0 * 2# was 6
figsize=(size,aspect*size)
#plt.rcParams.update({\
#'legend.fontsize':16, \
#'text.fontsize':16,\
#'axes.labelsize':16,\
#'font.family':'serif',\
#'font.size':16,\
#'xtick.labelsize':16,\
#'ytick.labelsize':16,\
#'figure.subplot.bottom':0.2,\
#'figure.figsize':figsize, \
#'savefig.dpi': 500.0, \
#'figure.autolayout': True})

_itime = time.time()
######################################################################
__author__   = "Prayush Kumar <prayush@astro.cornell.edu>"
PROGRAM_NAME = os.path.abspath(sys.argv[0])
verbose      = True



######################################################################
######################################################################
#
#     UTILITIES FOR 1D / ND DISTRIBUTIONS
#
######################################################################
######################################################################
def KDEMeanOverRange(kde_func, x_range):
    total_area = si.quad(kde_func, low, high)
    low = np.min(x_range)
    high = np.max(x_range)
    def give_integrand(xval): return kde_func(xval) * xval
    return si.quad(give_integrand, low, high) / total_area

def KDEMedianOverRange(kde_func, x_range):
    low = np.min(x_range)
    high = np.max(x_range)
    def give_pdf(xval): return -1.0 * kde_func(xval)
    return so.minimize(give_pdf, H0).x[0]


#######################################################
### CONTAINER and CALCULATION CLASSES
#######################################################
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
    def __init__(self, data, data_kde=None,\
                 kernel='gau',\
                 bw_method='scott',\
                 kernel_cut=3.0,\
                 xlimits = None,\
                 verbose=True, debug=False):
        if debug: print "Initializing OneDDistribution object.."
        self.input_data = np.array(data)
        self.bw_method  = bw_method
        self.kernel     = kernel
        self.kernel_cut = kernel_cut
        if xlimits:
            self.xllimit, self.xulimit = xlimits
        else:
            self.xllimit, self.xulimit = min(self.input_data), max(self.input_data)
        if data_kde != None:
            print "GOING TO INSERT KDE IN SELF...."
            self.kde = data_kde
            print """WARNING: Be careful when providing a kernel density estimator directly.
                              We assume, but not check, that it matches the input sample set.."""
        self.verbose = verbose
        self.debug = debug
        return
    ###
    def data(self):
        return self.input_data
    ###
    def xlimits(self):
        return [self.xllimit, self.xulimit]
    ##
    def normalization(self, xllimit = None, xulimit = None):
        if hasattr(self, "norm"):
            return self.norm
        if self.verbose:
            print "NORMALIZING 1D KDE"
        input_kde_func        = self.kde()
        if xllimit == None:
            xllimit, _ = self.xlimits()
        if xulimit == None:
            _, xulimit = self.xlimits()
        input_data_norm = si.quad(input_kde_func, xllimit, xulimit,\
                                        epsabs=1.e-16, epsrel=1.e-16)[0]
        self.norm = input_data_norm
        return input_data_norm
    ###
    def mean(self, xllimit = None, xulimit = None):
        tmp_data = self.input_data
        if xllimit is not None:
            tmp_data = tmp_data[tmp_data >= xllimit]
        if xulimit is not None:
            tmp_data = tmp_data[tmp_data >= xllimit]
        return np.mean(tmp_data)
    def median(self, xllimit = None, xulimit = None):
        tmp_data = self.input_data
        if xllimit is not None:
            tmp_data = tmp_data[tmp_data >= xllimit]
        if xulimit is not None:
            tmp_data = tmp_data[tmp_data >= xllimit]
        return np.median(tmp_data)
    def percentile(self, perc, xllimit = None, xulimit = None):
        tmp_data = self.input_data
        if xllimit is not None:
            tmp_data = tmp_data[tmp_data >= xllimit]
        if xulimit is not None:
            tmp_data = tmp_data[tmp_data >= xllimit]
        return np.percentile(tmp_data, perc)
    ###
    def kde(self):
        if hasattr(self, "evaluate_kde"):
            return self.evaluate_kde
        if self.verbose:
            print "INITALIZING 1D KDE"
        kde = KDEUnivariate(self.input_data)
        try:
            kde.fit(kernel=self.kernel, bw=self.bw_method, fft=True, cut=self.kernel_cut)
        except:
            kde.fit(kernel=self.kernel, bw=self.bw_method, fft=False, cut=self.kernel_cut)
        self.evaluate_kde          = kde.evaluate
        self.kde_object            = kde
        return self.evaluate_kde
    ###
    def sliced(self, i):
        if i!= 0:
            raise RuntimeError("One-D data has only one slice, indexed by 0 (asked for %d)" % i)
        return self
    ###oneD_
    def pdf_over_range(self, x_range):
        self.kde()
        return self.evaluate_kde(x_range)
    ###
    def mean_in_range(self, x_range):
        kde_func = self.kde()
        return KDEMeanOverRange(kde_func, x_range)
    ##
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
    def __init__(self, datadir, result_tag, event_ids, var_type,\
                 verbose=True):
        ### =======         INPUT CHECKING/HANDLING              ======= ###
        if not os.path.exists(datadir):
            raise IOError("DATA Directory %s does not exist.." % datadir)
        self.datadir     = datadir
        self.result_tag  = result_tag
        self.event_ids   = event_ids # Event IDs (int) labeling events and their order
        self.var_type    = var_type
        self.data        = self.read_distributions()
        self.verbose     = verbose

        ### =======         INITLIAZE              ======= ###
        self.event          = {}
        self.pdf_event      = {}
        self.pdf_norm_event = {}
        self.pdf_cum_events = {}
        ##
        return
    ###
    def read_distributions(self):
        datadir, result_tag, event_ids =\
            self.datadir, self.result_tag, self.event_ids
        data = {}
        id_cnt = 0
        for c in result_tag:
            if c=='%': id_cnt += 1
        for event_id in event_ids:
            if verbose: print "REEADING POSTERIOR FOR EVENT ", event_id
            event_id_pattern = tuple(np.ones(id_cnt) * event_id)
            res_file = os.path.join(datadir, result_tag % event_id_pattern)
            data[event_id] = np.loadtxt(res_file)
        return data
    ###
    def process_oned_slices(self, kernel_cut=3.0, reprocess=False):
        for id_cnt, id in enumerate(self.event_ids):
            if id in self.event and not reprocess:
                continue
            if self.verbose:
                print "\n READING EVENT %d" % (id)

            self.event[id] = MultiDDistribution(self.data[id], self.var_type,\
                                                oneD_kernel_cut=kernel_cut)
            self.pdf_norm_event[id] = self.event[id].sliced(0).normalization()
        return self.event
    ###
    def combine_oned_slices(self, x_range, prior_func, event_ids = None):
        if len(self.event) == 0:
            print "Please process oneD slices first"
            return
        ####
        if event_ids == None: event_ids = self.event_ids
        x_range = np.array(x_range)
        self.XRANGE    = x_range
        self.PRIORDATA = prior_func(x_range)
        for id_cnt, id in enumerate(event_ids):
            self.pdf_norm_event[id] = self.event[id].sliced(0).normalization(xllimit=x_range.min(),\
                                                                             xulimit=x_range.max())
            self.pdf_event[id] = self.event[id].sliced(0).kde()(x_range) / self.pdf_norm_event[id]
            if id_cnt != 0:
                self.pdf_cum_events[id] = self.pdf_cum_events[event_ids[id_cnt-1]] * self.pdf_event[id]
                self.pdf_cum_events[id] /= self.PRIORDATA**2
                AREA = np.sum(self.pdf_cum_events[id]) * (x_range.max() - x_range.min()) / len(x_range)
                self.pdf_cum_events[id] /= AREA
            else:
                self.pdf_cum_events[id] = self.pdf_event[id] / self.pdf_norm_event[id]
        #for id_cnt, id in enumerate(event_ids):
        #    self.pdf_cum_events[id] /= self.PRIORDATA**id_cnt
        #    AREA = np.sum(self.pdf_cum_events[id]) * (x_range.max() - x_range.min()) / len(x_range)
        #    self.pdf_cum_events[id] /= AREA
        return self.pdf_cum_events[id]
    ###
    def plot_combined_oned_slices(self, x_range, prior_func, event_ids = None,\
                                  labels_per_column = 15,\
                                  label_every_nth_curve=1,\
                                  cum_pdf_ylim=[0,1.2],\
                                  pdf_ylim=[0,0.22],\
                                 cum_pdf_title='CUMULATIVE PDFS',\
                                 pdf_title='INDIVIDUAL EVENT PDFS AND KDES',\
                                 xlabel='$H_0$'):
        if event_ids == None: event_ids = self.event_ids
        self.combine_oned_slices( x_range, prior_func, event_ids=event_ids)
        # MAKE FIGURE
        ax0 = plt.figure(figsize=(8,6)).add_subplot(111)
        ax2 = plt.figure(figsize=(8,6)).add_subplot(111)
        #
        label_cnt = 0
        for id_cnt, id in enumerate(event_ids):
            label_txt = None
            if label_every_nth_curve > 1 and ((id_cnt+1) % label_every_nth_curve) == 0:
                label_txt = '%d'% (id_cnt+1)
                label_cnt += 1
            ax0.plot(x_range, self.pdf_cum_events[id], 'k', lw=2,
                     alpha=0.1+id_cnt*0.7/len(event_ids), label=label_txt)

            label_txt = None
            if label_every_nth_curve > 1 and ((id_cnt+1) % label_every_nth_curve) == 0:
                label_txt = '%d'% (id_cnt+1)
            ax2.plot(x_range, self.pdf_event[id] / self.pdf_norm_event[id],
                     label=label_txt, color='k', lw=2,
                     alpha=0.1+id_cnt*0.7/len(event_ids))
            _ = ax2.hist(self.event[id].sliced(0).input_data, bins=50, normed=True, alpha=0.03);

            ax0.axvline(self.event[id].sliced(0).mean(), color='g', alpha=0.5, ls='--')
            ax0.axvline([self.event[id].sliced(0).median()], color='r', alpha=0.6, ls='--')
            ax0.axvline(KDEMedianOverRange(UnivariateSpline(x_range, self.pdf_cum_events[id]), x_range),
                        color='k', alpha=0.1+id_cnt*0.7/len(event_ids), ls='--')

            ax2.axvline(self.event[id].sliced(0).mean(), color='g', alpha=0.5, ls='--')
            ax2.axvline([self.event[id].sliced(0).median()], color='r', alpha=0.6, ls='--')
        ####
        ax0.set_title(cum_pdf_title)
        ax2.set_title(pdf_title)

        ax0.legend(loc="upper left", bbox_to_anchor=(1,1), ncol=label_cnt/labels_per_column+1)
        ax2.legend(loc="upper left", bbox_to_anchor=(1,1), ncol=label_cnt/labels_per_column+1)

        ax0.set_ylim(cum_pdf_ylim)
        ax2.set_ylim(pdf_ylim)

        ## COMMON FORMATTING
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
    def __init__(self, data, var_type, var_names=[], data_kde=None,\
                 oneD_kernel='gau',\
                 oneD_kernel_cut=3.0,\
                 oneD_bw_method='normal_reference',\
                 mulD_kernel='gau',\
                 mulD_bw_method='normal_reference',\
                 xlimits = None,\
                 verbose=True, debug=True):
        ## CHECK INPUTS
        self.verbose = verbose
        self.debug   = debug
        if len(np.shape(np.array(data))) == 1:
            print "WARNING: Returning OneDDistribution for 1-D distributions"
            return OneDDistribution(data, data_kde=data_kde,\
                                   kernel=oneD_kernel,\
                                   kernel_cut=oneD_kernel_cut,\
                                   bw_method=oneD_bw_method,\
                                   verbose=verbose, debug=debug)

        ## ENSURE DATA SHAPE, EXTRACT DIMENSION
        if verbose: print "Initializing MultiDDistribution object.."
        data            = np.array(data)
        #if np.shape(data)[0] < np.shape(data)[1]:
        #    print "WARNING: FEWER ROWS than COLUMNS, assuming its a row-wise distribution"
        #    data = np.transpose(data)
        self.input_data = np.array(data)
        self.var_names = var_names
        self.dim        = np.shape(data)[-1]

        ## PROCESS 1-D SLICES (assuming independently sampled variables)
        if verbose:
            print "PROCESSING 1-D SLICES .."
        self.slices = [OneDDistribution(self.sliced(i),\
                                        kernel_cut=oneD_kernel_cut\
                                       ) for i in range(self.dim)]

        ## UPPER AND LOWER LIMITS IN 1-D SLICES
        if verbose:
            print "COMPUTING /LOWER AND UPPER LIMITS IN EACH DIMENSION"
        if xlimits:
            self.xllimit, self.xulimit = xlimits
        else:
            self.xllimit = np.array([min(sl.input_data) for sl in self.slices])
            self.xulimit = np.array([max(sl.input_data) for sl in self.slices])

        ## STORE
        self.var_type   = var_type
        self.bw_method  = mulD_bw_method
        self.kernel     = mulD_kernel

        if data_kde:
            self.kde = data_kde
            print """WARNING: Be careful when providing a kernel density estimator directly.
                              We assume, but not check, that it matches the input sample set.."""
        return
    ###
    def structured_data(self):
        self.structured_data = cp.deepcopy(self.input_data)
        if debug:
            print "Shape of structured_data: ", np.shape(self.structured_data)
        if len(var_names) == np.shape(self.input_data)[-1]:
            self.structured_data = np.array(self.structured_data,
                                            dtype = [(h, '<f8') for h in var_names])
        return self.structured_data
    ###
    def index_of_name(self, name):
        return np.where([n == name for n in self.var_names])[0][0]
    ###
    def sliced(self, i):
        """
        This function returns a numpy.array if passed an int,
        or a OneDDistribution if that is available.
        """
        if type(i)==int:
            if hasattr(self, "slices"):
                return self.slices[i]
            else:
                return self.input_data[:,i]
        elif type(i) == str and i in self.var_names:
            return self.slices[self.index_of_name(i)]
        else:
            raise IOError("Could not find slice labelled "+str(i))
    ###
    def mean(self):
        return np.array([self.sliced(i).mean() for i in range(self.dim)])
    def median(self):
        return np.array([self.sliced(i).median() for i in range(self.dim)])
    ###
    def kde(self):
        if hasattr(self, "kde"):
            return self.kde
        kde = KDEMultivariate(self.input_data, var_type=self.var_type, bw=self.bw_method)
        self.kde          = kde
        self.evaluate_kde = kde.pdf
        return kde
    ##
    def xlimits(self):
        return [self.xllimit, self.xulimit]
    ##
    def normalization(self):
        if hasattr(self, "norm"):
            return self.norm
        # ELSE COMPUTE NORM OF MULTI-D DISTRIBUTION
        self.norm = 1.0
        for i in range(self.dim):
            self.norm *= self.sliced(i).normalization()
        return self.norm
    ##
    def plot_twod_slice(self, params_plot,
                    fig=None, # CAN PLOT ON EXISTING FIGURE
                    label='', # LABEL THAT GOES ON EACH PANEL
                    color=None,
                    params_labels=None,
                    nhbins=30,  # NO OF BINS IN HISTOGRAMS
                    plim_low=None, plim_high=None,
                    legend_fontsize=12,
                    plot_type='scatter', # SCATTER OR CONTOUR
                    scatter_alpha=0.2, # Transparency of scatter points
                    param_color=None, # 3RD DIMENSION SHOWN AS COLOR
                    param_color_label=None, # LABEL of 3RD DIMENSION SHOWN AS COLOR
                    color_max=None, color_min=None, cmap=cm.plasma_r,
                    contour_levels=[90.0],
                    contour_lstyles=["solid" , "dashed" , "dashdot" , "dotted"],
                    return_areas_in_contours=False,
                    npixels=50,
                    debug=False, verbose=None
                    ):
        #{{{
        """
Generates a corner plot for given parameters. 2D panels can have data points
directly or percentile contours, not both simultaneously yet.

When plotting data points, user can also add colors to it based on a 3rd parameter
[Use param_color].

 When plotting contours, user can ask for the area inside the contours to be
 returned. However, if the contours are railing against boundaries, there is no
 guarantee that the areas will be correct. Multiple disjoint closed contours are
 supported though [Use return_areas_in_contours].

Input:
(1) [REQUIRED] params_plot: list of parameters to plot. String names or Indices work.
(2) [OPTIONAL] params_labels: list of parameter names to use in latex labels.
(2a,b) [OPTIONAL] xlim_low/high: lists of parameter lower & upper limits
(3) [REQUIRED] plot_type: Either "scatter" or "contour"
(4) [OPTIONAL] contour_levels: credible interval levels for "contour" plots
(5) [OPTIONAL] contour_lstyles: line styles for contours. REQUIRED if >4 contour levels
                                are to be plotted.
(6) [OPTIONAL] npixels: number of pixels/bins along each dimension in contour plots.
(7) [OPTIONAL] label: String label to label the entire data.
(8) [OPTIONAL] color / param_color: Single color for "scatter"/"contour" plots
                                          OR
                        One parameter that 2D scatter plots will show as color.
(9) [OPTIONAL] color_max=None, color_min=None: MAX/MIN values of parameter
                                              "param_color" to use in "scatter"
                                              plots
(10)[OPTIONAL] nhbins=30 : NO OF BINS IN HISTOGRAMS
        """
        ## Preliminary checks on inputs
        if len(contour_levels) > len(contour_lstyles):
            if plot_type == 'contour':
                raise IOError("Please provide as many linestyles as contour levels")

        if param_color is not None and "scatter" not in plot_type:
            raise IOError("Since you passed a 3rd dimension, only plot_type=scatter is allowed")

        ## Local verbosity level takes precedence, else the class's is used
        if verbose == None: verbose = self.verbose

        ## IF no labels are provided by User, use default Latex labels for CBCs
        if params_labels is None:
            params_labels = ParamLatexLabels()

        def get_param_label(pp):
            if params_labels is not None and pp in params_labels:
                return params_labels[pp]
            if pp in param_color and param_color_label is not None:
                return param_color_label
            return pp.replace('_','-')

        ## This is the label for the entire data set, not individual quantities
        label      = label.replace('_','-')

        ## Configure the full figure panel
        no_of_rows = len(params_plot)
        no_of_cols = len(params_plot)

        if type(fig) != matplotlib.figure.Figure:
            fig = plt.figure(figsize=(6*no_of_cols,4*no_of_rows))

        fig.hold(True)

        ## Pre-choose color for 1D histograms (and scatter plots, if applicable)
        rand_color = np.random.rand(3,1)
        if color != None: rand_color = color

        if return_areas_in_contours: contour_areas = {}
        contour_levels = sorted(contour_levels, reverse=True)

        ## Start drawing panels
        if True:
            if True:
                ## If execution reaches here, the current panel is in the lower diagonal half
                if verbose:
                    print "Making plot (%d,%d,%d)" % (no_of_rows, no_of_cols, (nr*no_of_cols) + nc)

                ## If user asks for scatter-point colors to be a 3rd dimension
                if param_color in self.var_names:
                    ax = fig.add_subplot(no_of_rows, no_of_cols, (nr*no_of_cols) + nc + 1)
                    p1 = params_plot[nc]
                    p2 = params_plot[nr]
                    p1label = get_param_label(p1)
                    p2label = get_param_label(p2)
                    cblabel = get_param_label(param_color)
                    if verbose:
                        print "Scatter plot w color: %s vs %s vs %s" % (p1,p2, param_color)
                    _d1, _d2  = self.sliced(p1).data(), self.sliced(p2).data()
                    im = ax.scatter(_d1, _d2, c=self.sliced(param_color).data(),
                                    alpha=scatter_alpha,
                                    edgecolors=None, linewidths=0,
                                    vmin=color_min, vmax=color_max, cmap=cmap,
                                    label=label)
                    cb = fig.colorbar(im, ax=ax)
                    if nc == (no_of_cols-1): cb.set_label(cblabel)
                    if nr == (no_of_rows-1): ax.set_xlabel(p1label)
                    if nc == 0: ax.set_ylabel(p2label)
                    ## set X/Y axis limits
                    if plim_low is not None and plim_high is not None:
                        ax.set_xlim( plim_low[nc], plim_high[nc] )
                        ax.set_ylim( plim_low[nr], plim_high[nr] )
                    else:
                        ax.set_xlim( 0.95 * np.min(_d1), 1.05 * np.max(_d1) )
                        ax.set_ylim( 0.95 * np.min(_d2), 1.05 * np.max(_d2) )
                    ax.legend(loc='best', fontsize=legend_fontsize)
                    ax.grid()
                elif param_color is not None:
                    raise IOError("Could not find parameter %s to show" % param_color)
                elif plot_type=='scatter':
                    ax = fig.add_subplot(no_of_rows, no_of_cols, (nr*no_of_cols) + nc + 1)
                    p1 = params_plot[nc]
                    p2 = params_plot[nr]
                    p1label = get_param_label(p1)
                    p2label = get_param_label(p2)
                    if verbose: print "Scatter plot: %s vs %s" % (p1,p2)
                    _d1, _d2  = self.sliced(p1).data(), self.sliced(p2).data()
                    im = ax.scatter(_d1, _d2, c=rand_color,
                                    alpha=scatter_alpha,
                                    edgecolors=None, linewidths=0,
                                    label=label)
                    if nr == (no_of_rows-1): ax.set_xlabel(p1label)
                    if nc == 0: ax.set_ylabel(p2label)
                    if plim_low is not None and plim_high is not None:
                        ax.set_xlim( plim_low[nc], plim_high[nc] )
                        ax.set_ylim( plim_low[nr], plim_high[nr] )
                    else:
                        ax.set_xlim( 0.95 * np.min(_d1), 1.05 * np.max(_d1) )
                        ax.set_ylim( 0.95 * np.min(_d2), 1.05 * np.max(_d2) )
                    ax.legend(loc='best', fontsize=legend_fontsize)
                    ax.grid()
                elif plot_type=='contour':
                    ax = fig.add_subplot(no_of_rows, no_of_cols, (nr*no_of_cols) + nc + 1)
                    p1 = params_plot[nc]
                    p2 = params_plot[nr]
                    p1label = get_param_label(p1)
                    p2label = get_param_label(p2)
                    if verbose: print "Contour plot: %s vs %s" % (p1,p2)
                    ## Get data
                    d1 = self.sliced(p1).data()
                    d2 = self.sliced(p2).data()
                    dd = np.column_stack([d1, d2])
                    if verbose: print np.shape(d1), np.shape(d2), np.shape(dd)
                    pdf = gaussian_kde(dd.T)
                    ## Get contour levels
                    zlevels = [np.percentile(pdf(dd.T), 100.0 - lev) for lev in contour_levels]
                    x11vals = np.linspace(dd[:,0].min(), dd[:,0].max(),npixels)
                    x12vals = np.linspace(dd[:,1].min(), dd[:,1].max(),npixels)
                    q, w = np.meshgrid(x11vals, x12vals)
                    r1 = pdf([q.flatten(),w.flatten()])
                    r1.shape=q.shape
                    ## Draw contours
                    im = ax.contour(x11vals, x12vals, r1, zlevels,
                                    colors=rand_color,
                                    linestyles=contour_lstyles[:len(contour_levels)],
                                    label=label)

                    ## Get area inside contour
                    if return_areas_in_contours:
                        if verbose: print "Computing area inside contours."
                        contour_areas[p1+p2] = []
                        for ii in range(len(zlevels)):
                            contour = im.collections[ii]
                            # Add areas inside all independent contours, in case
                            # there are multiple disconnected ones
                            contour_areas[p1+p2].append(\
                                  np.sum([area_inside_contour(vs.vertices)\
                                    for vs in contour.get_paths()]) )
                            if verbose:
                                print "Total area = %.9f, %.9f" % (contour_areas[p1+p2][-1])
                            if debug:
                                for _i, vs in enumerate(contour.get_paths()):
                                    print "sub-area %d: %.8e" % (_i, area_inside_contour(vs.vertices))
                        contour_areas[p1+p2] = np.array( contour_areas[p1+p2] )

                    ## BEAUTIFY contour labeling..!
                    # Define a class that forces representation of float to look
                    # a certain way. This remove trailing zero so '1.0' becomes '1'
                    class nf(float):
                        def __repr__(self):
                            str = '%.1f' % (self.__float__(),)
                            if str[-1] == '0': return '%.0f' % self.__float__()
                            else: return '%.1f' % self.__float__()
                    # Recast levels to new class
                    im.levels = [nf(val) for val in contour_levels]
                    # Label levels with specially formatted floats
                    if plt.rcParams["text.usetex"]: fmt = r'%r \%%'
                    else: fmt = '%r %%'
                    ax.clabel(im, im.levels, inline=True, fmt=fmt, fontsize=10)

                    if nr == (no_of_rows-1): ax.set_xlabel(p1label)
                    if nc == 0: ax.set_ylabel(p2label)
                    if plim_low is not None and plim_high is not None:
                        ax.set_xlim( plim_low[nc], plim_high[nc] )
                        ax.set_ylim( plim_low[nr], plim_high[nr] )
                    else:
                        ax.set_xlim( 0.95 * np.min(d1), 1.05 * np.max(d1) )
                        ax.set_ylim( 0.95 * np.min(d2), 1.05 * np.max(d2) )
                    #ax.legend(loc='best', fontsize=legend_fontsize)
                    ax.grid(True)
                else:
                    raise IOError("plot type %s not supported.." % plot_type)
        ##
        if plot_type=='contour' and return_areas_in_contours and debug:
            return fig, contour_areas, contour.get_paths(), im
        elif plot_type=='contour' and return_areas_in_contours:
            return fig, contour_areas
        else:
            return fig
    #}}}
    ##
    def corner_plot(self, params_plot,
                    params_true_vals=None, # TRUE / KNOWN values of PARAMETERS
                    params_oned_priors=None, # PRIOR DISTRIBUTION FOR PARAMETERS
                    fig=None, # CAN PLOT ON EXISTING FIGURE
                    axes_array=None, # NEED ARRAY OF AXES IF PLOTTING ON EXISTING FIGURE
                    panel_size=(6,4),
                    histogram_type='bar', # Histogram type (bar / step / barstacked)
                    priors_histogram_type='stepfilled',
                    nhbins=30,  # NO OF BINS IN HISTOGRAMS
                    projection='rectilinear',
                    label='', # LABEL THAT GOES ON EACH PANEL
                    params_labels=None,
                    plim_low=None, plim_high=None,
                    legend_fontsize=18,
                    plot_type='scatter', # SCATTER OR CONTOUR
                    color=None,
                    hist_alpha=0.3, # Transparency of histograms
                    scatter_alpha=0.2, # Transparency of scatter points
                    npixels=50,
                    param_color=None, # 3RD DIMENSION SHOWN AS COLOR
                    param_color_label=None, # LABEL of 3RD DIMENSION SHOWN AS COLOR
                    color_max=None,
                    color_min=None,
                    cmap=cm.plasma_r,
                    contour_levels=[90.0],
                    contour_lstyles=["solid" , "dashed" , "dashdot" , "dotted"],
                    label_contours=True, #Whether or not to label individuals
                    contour_labels_inline=True,
                    contour_labels_loc="upper center",
                    return_areas_in_contours=False,
                    label_oned_hists=-1, # Which one-d histograms to label?
                    skip_oned_hists=False,
                    rotate_last_oned_hist=True,
                    label_oned_loc='outside',
                    label_oned_bbox=[(1.3, 0.9)],
                    show_oned_median=False,
                    grid_oned_on=False,
                    debug=False, verbose=None
                    ):
        #{{{
        """
Generates a corner plot for given parameters. 2D panels can have data points
directly or percentile contours, not both simultaneously yet.

When plotting data points, user can also add colors to it based on a 3rd parameter
[Use param_color].

 When plotting contours, user can ask for the area inside the contours to be
 returned. However, if the contours are railing against boundaries, there is no
 guarantee that the areas will be correct. Multiple disjoint closed contours are
 supported though [Use return_areas_in_contours].

Input:
(1) [REQUIRED] params_plot: list of parameters to plot. String names or Indices work.
(2) [OPTIONAL] params_labels: list of parameter names to use in latex labels.
(2a,b) [OPTIONAL] xlim_low/high: lists of parameter lower & upper limits
(3) [REQUIRED] plot_type: Either "scatter" or "contour"
(4) [OPTIONAL] contour_levels: credible interval levels for "contour" plots
(5) [OPTIONAL] contour_lstyles: line styles for contours. REQUIRED if >4 contour levels
                                are to be plotted.
(6) [OPTIONAL] npixels: number of pixels/bins along each dimension in contour plots.
(7) [OPTIONAL] label: String label to label the entire data.
(8) [OPTIONAL] color / param_color: Single color for "scatter"/"contour" plots
                                          OR
                        One parameter that 2D scatter plots will show as color.
(9) [OPTIONAL] color_max=None, color_min=None: MAX/MIN values of parameter
                                              "param_color" to use in "scatter"
                                              plots
(10)[OPTIONAL] nhbins=30 : NO OF BINS IN HISTOGRAMS
(11)[OPTIONAL] params_oned_priors=None: PRIOR SAMPLES to be overplotted onto
                                        1D histograms. Dictionary.
        """
        ## Preliminary checks on inputs
        if len(contour_levels) > len(contour_lstyles):
            if plot_type == 'contour':
                raise IOError("Please provide as many linestyles as contour levels")

        if param_color is not None and "scatter" not in plot_type:
            raise IOError("Since you passed a 3rd dimension, only plot_type=scatter is allowed")

        ## Local verbosity level takes precedence, else the class's is used
        if verbose == None: verbose = self.verbose

        ## IF no labels are provided by User, use default Latex labels for CBCs
        if params_labels is None:
            params_labels = ParamLatexLabels()

        def get_param_label(pp):
            if params_labels is not None and pp in params_labels:
                return params_labels[pp]
            if pp in param_color and param_color_label is not None:
                return param_color_label
            return pp.replace('_','-')

        ## This is the label for the entire data set, not individual quantities
        label      = label.replace('_','-')

        ## Configure the full figure panel
        no_of_rows = len(params_plot)
        no_of_cols = len(params_plot)

        if type(fig) != matplotlib.figure.Figure or axes_array is None:
            if no_of_rows == 2:
                import matplotlib.gridspec as gridspec
                _nrows = _ncols = 3
                fig = plt.figure(figsize=(panel_size[0]*_ncols, panel_size[1]*_nrows))
                gs = gridspec.GridSpec(2, 2,
                       width_ratios=[2, 1],
                       height_ratios=[1, 2]
                       )
                gs.update(hspace=0, wspace=0)
                ax1 = fig.add_subplot(gs[0])
                ax2 = fig.add_subplot(gs[1])
                ax3 = fig.add_subplot(gs[2])
                ax4 = fig.add_subplot(gs[3])
                axes_array = [[ax1, ax2], [ax3, ax4]]
            else:
                fig, axes_array = plt.subplots(no_of_rows, no_of_cols,
                    figsize=(panel_size[0]*no_of_cols, panel_size[1]*no_of_rows),
                    gridspec_kw = {'wspace':0, 'hspace':0})

        fig.hold(True)

        ## Pre-choose color for 1D histograms (and scatter plots, if applicable)
        rand_color = np.random.rand(3)
        if color != None: rand_color = color

        if return_areas_in_contours: contour_areas = {}
        contour_levels = sorted(contour_levels, reverse=True)

        ## Start drawing panels
        for nr in range(no_of_rows):
            for nc in range(no_of_cols):
                ## We keep the upper diagonal half of the figure empty.
                ## FIXME: Could we use it for same data, different visualization?
                if nc > nr:
                    ax = axes_array[nr][nc]
                    try:
                        fig.delaxes(ax)
                    except: pass
                    continue

                # Make 1D histograms along the diagonal
                if nc == nr:
                    if skip_oned_hists: continue
                    hist_orientation = 'vertical'
                    if rotate_last_oned_hist and nc == (no_of_cols - 1):
                        hist_orientation = 'horizontal'
                    p1 = params_plot[nc]
                    p1label = get_param_label(p1)
                    #ax = fig.add_subplot(no_of_rows, no_of_cols, (nr*no_of_cols) + nc + 1)
                    if no_of_rows == 1 and no_of_cols == 1:
                        ax = axes_array
                    else:
                        ax = axes_array[nr][nc]
                    # Plot known / injected / true value if given
                    if params_true_vals != None:
                        p_true_val = params_true_vals[nc]
                        if p_true_val != None:
                            if rotate_last_oned_hist and nc == (no_of_cols - 1):
                                ax.axhline(p_true_val,
                                           lw = 0.5, ls='solid', color = rand_color)
                            else:
                                ax.axvline(p_true_val,
                                           lw = 0.5, ls='solid', color = rand_color)
                    # Plot one-d posterior
                    _data = self.sliced(p1).data()
                    im = ax.hist(_data, bins=nhbins,
                                  histtype=histogram_type,
                                  normed=True, alpha=hist_alpha,
                                  color=rand_color, label=label,
                                  orientation=hist_orientation)
                    if rotate_last_oned_hist and nc == (no_of_cols - 1):
                        ax.axhline(np.percentile(_data, 5), lw=1, ls = 'dashed',
                                  color = rand_color, alpha=1)
                        ax.axhline(np.percentile(_data, 95), lw=1, ls = 'dashed',
                                  color = rand_color, alpha=1)
                    else:
                        ax.axvline(np.percentile(_data, 5), lw=1, ls = 'dashed', color = rand_color, alpha=1)
                        ax.axvline(np.percentile(_data, 95), lw=1, ls = 'dashed', color = rand_color, alpha=1)
                    if show_oned_median:
                        if rotate_last_oned_hist and nc == (no_of_cols - 1):
                            ax.axhline(np.median(_data), ls = '-', color = rand_color)
                        else:
                            ax.axvline(np.median(_data), ls = '-', color = rand_color)
                    try:
                        if label_oned_hists == -1 or nc in label_oned_hists:
                            if label_oned_loc is not 'outside' and label_oned_loc is not '':
                                ax.legend(loc=label_oned_loc, fontsize=legend_fontsize)
                            else:
                                ax.legend(fontsize=legend_fontsize,
                                          bbox_to_anchor=label_oned_bbox[0])
                    except TypeError: raise TypeError("Pass a list for label_oned_hists")
                    if params_oned_priors is not None and p1 in params_oned_priors:
                        _data = params_oned_priors[p1]
                        if plim_low is not None and plim_high is not None:
                            _prior_xrange = (plim_low[nc], plim_high[nc])
                        else: _prior_xrange = None
                        im = ax.hist(_data, bins=nhbins,
                                    histtype=priors_histogram_type, color='k', alpha=0.25,
                                    range=_prior_xrange, normed=True,
                                    orientation=hist_orientation
                                    )
                    if plim_low is not None and plim_high is not None:
                        if rotate_last_oned_hist and nc == (no_of_cols - 1):
                            ax.set_ylim(plim_low[nc], plim_high[nc])
                        else:
                            ax.set_xlim(plim_low[nc], plim_high[nc])
                    ax.grid(grid_oned_on)
                    if nr == (no_of_rows-1) and (no_of_cols > 2 or no_of_rows > 2):
                        ax.set_xlabel(p1label)
                    if nc == 0 and (no_of_cols > 2 or no_of_rows > 2):
                        ax.set_ylabel(p1label)
                    ax.set_yticklabels([])
                    if nr < (no_of_rows-1) or rotate_last_oned_hist:
                        ax.set_xticklabels([])
                    continue

                ## If execution reaches here, the current panel is in the lower diagonal half
                if verbose:
                    print "Making plot (%d,%d,%d)" % (no_of_rows, no_of_cols, (nr*no_of_cols) + nc)

                ## Get plot for this panel
                #ax = fig.add_subplot(no_of_rows, no_of_cols, (nr*no_of_cols) + nc + 1,
                #                          projection=projection)
                ax = axes_array[nr][nc]

                ## Plot known / injected / true value if given
                if params_true_vals != None:
                    pc_true_val = params_true_vals[nc]
                    pr_true_val = params_true_vals[nr]
                    if pc_true_val != None:
                        ax.axvline(pc_true_val,
                                   lw = 0.5, ls='solid', color = rand_color)
                    if pr_true_val != None:
                        ax.axhline(pr_true_val,
                                   lw = 0.5, ls='solid', color = rand_color)
                    if pc_true_val != None and pr_true_val != None:
                        ax.plot([pc_true_val], [pr_true_val],
                                's', color = rand_color)

                ## Now plot what the user requested
                ### If user asks for scatter-point colors to be a 3rd dimension
                if param_color in self.var_names:
                    p1 = params_plot[nc]
                    p2 = params_plot[nr]
                    p1label = get_param_label(p1)
                    p2label = get_param_label(p2)
                    cblabel = get_param_label(param_color)
                    if verbose:
                        print "Scatter plot w color: %s vs %s vs %s" % (p1,p2, param_color)
                    _d1, _d2  = self.sliced(p1).data(), self.sliced(p2).data()
                    im = ax.scatter(_d1, _d2, c=self.sliced(param_color).data(),
                                    alpha=scatter_alpha,
                                    edgecolors=None, linewidths=0,
                                    vmin=color_min, vmax=color_max, cmap=cmap,
                                    label=label)
                    cb = fig.colorbar(im, ax=ax)
                    if nc == (no_of_cols-1): cb.set_label(cblabel)
                    if nr == (no_of_rows-1): ax.set_xlabel(p1label)
                    if nc == 0: ax.set_ylabel(p2label)
                    ## set X/Y axis limits
                    if plim_low is not None and plim_high is not None:
                        ax.set_xlim( plim_low[nc], plim_high[nc] )
                        ax.set_ylim( plim_low[nr], plim_high[nr] )
                    else:
                        ax.set_xlim( 0.95 * np.min(_d1), 1.05 * np.max(_d1) )
                        ax.set_ylim( 0.95 * np.min(_d2), 1.05 * np.max(_d2) )
                    ax.legend(loc='best', fontsize=legend_fontsize)
                    ax.grid()
                elif param_color is not None:
                    raise IOError("Could not find parameter %s to show" % param_color)
                ### If user asks for scatter plot without 3rd Dimension info
                elif plot_type=='scatter':
                    p1 = params_plot[nc]
                    p2 = params_plot[nr]
                    p1label = get_param_label(p1)
                    p2label = get_param_label(p2)
                    if verbose: print "Scatter plot: %s vs %s" % (p1,p2)
                    _d1, _d2  = self.sliced(p1).data(), self.sliced(p2).data()
                    im = ax.scatter(_d1, _d2, c=rand_color,
                                    alpha=scatter_alpha,
                                    edgecolors=None, linewidths=0,
                                    label=label)
                    if nr == (no_of_rows-1): ax.set_xlabel(p1label)
                    if nc == 0: ax.set_ylabel(p2label)
                    if plim_low is not None and plim_high is not None:
                        ax.set_xlim( plim_low[nc], plim_high[nc] )
                        ax.set_ylim( plim_low[nr], plim_high[nr] )
                    else:
                        ax.set_xlim( 0.95 * np.min(_d1), 1.05 * np.max(_d1) )
                        ax.set_ylim( 0.95 * np.min(_d2), 1.05 * np.max(_d2) )
                    ax.legend(loc='best', fontsize=legend_fontsize)
                    ax.grid()
                ### If user asks for contour plot without 3rd Dimension info
                elif plot_type=='contour':
                    p1 = params_plot[nc]
                    p2 = params_plot[nr]
                    p1label = get_param_label(p1)
                    p2label = get_param_label(p2)
                    if verbose: print "Contour plot: %s vs %s" % (p1,p2)
                    ## Get data
                    d1 = self.sliced(p1).data()
                    d2 = self.sliced(p2).data()
                    dd = np.column_stack([d1, d2])
                    if verbose: print np.shape(d1), np.shape(d2), np.shape(dd)
                    pdf = gaussian_kde(dd.T)
                    ## Get contour levels
                    zlevels = [np.percentile(pdf(dd.T), 100.0 - lev) for lev in contour_levels]
                    x11vals = np.linspace(dd[:,0].min(), dd[:,0].max(),npixels)
                    x12vals = np.linspace(dd[:,1].min(), dd[:,1].max(),npixels)
                    q, w = np.meshgrid(x11vals, x12vals)
                    r1 = pdf([q.flatten(),w.flatten()])
                    r1.shape=q.shape
                    ## Draw contours
                    im = ax.contour(x11vals, x12vals, r1, zlevels,
                                    colors=rand_color,
                                    linestyles=contour_lstyles[:len(contour_levels)],
                                    label=label)

                    ## Get area inside contour
                    if return_areas_in_contours:
                        if verbose: print "Computing area inside contours."
                        contour_areas[p1+p2] = []
                        for ii in range(len(zlevels)):
                            contour = im.collections[ii]
                            # Add areas inside all independent contours, in case
                            # there are multiple disconnected ones
                            contour_areas[p1+p2].append(\
                                  np.sum([area_inside_contour(vs.vertices)\
                                    for vs in contour.get_paths()]) )
                            if verbose:
                                print "Total area = %.9f, %.9f" % (contour_areas[p1+p2][-1])
                            if debug:
                                for _i, vs in enumerate(contour.get_paths()):
                                    print "sub-area %d: %.8e" % (_i, area_inside_contour(vs.vertices))
                        contour_areas[p1+p2] = np.array( contour_areas[p1+p2] )

                    ####
                    ## BEAUTIFY contour labeling..!
                    # Define a class that forces representation of float to look
                    # a certain way. This remove trailing zero so '1.0' becomes '1'
                    class nf(float):
                        def __repr__(self):
                            str = '%.1f' % (self.__float__(),)
                            if str[-1] == '0': return '%.0f' % self.__float__()
                            else: return '%.1f' % self.__float__()
                    # Recast levels to new class
                    im.levels = [nf(val) for val in contour_levels]
                    # Label levels with specially formatted floats
                    if plt.rcParams["text.usetex"]: fmt = r'%r \%%'
                    else: fmt = '%r %%'
                    ####
                    if label_contours:
                        if contour_labels_inline:
                            ax.clabel(im, im.levels,
                                  inline=False,
                                  use_clabeltext=True,
                                  fmt=fmt, fontsize=10)
                        else:
                            for zdx, _ in enumerate(zlevels):
                                _ = ax.plot([], [], color=rand_color,
                                            ls=contour_lstyles[:len(contour_levels)][zdx],
                                            label=im.levels[zdx])
                            ax.legend(loc=contour_labels_loc, fontsize=legend_fontsize)
                    else: pass
                    #
                    if nr == (no_of_rows-1): ax.set_xlabel(p1label)
                    if nc == 0: ax.set_ylabel(p2label)
                    if plim_low is not None and plim_high is not None:
                        ax.set_xlim( plim_low[nc], plim_high[nc] )
                        ax.set_ylim( plim_low[nr], plim_high[nr] )
                    elif projection != "mollweide":
                        ax.set_xlim( 0.95 * np.min(d1), 1.05 * np.max(d1) )
                        ax.set_ylim( 0.95 * np.min(d2), 1.05 * np.max(d2) )
                    else: pass
                    #ax.legend(loc='best', fontsize=legend_fontsize)
                    ax.grid(True)
                else:
                    raise IOError("plot type %s not supported.." % plot_type)
                if nc != 0:
                    print "removing Yticklabels for (%d, %d)" % (nr, nc)
                    ax.set_yticklabels([])
                if nr != (no_of_rows - 1):
                    print "removing Xticklabels for (%d, %d)" % (nr, nc)
                    ax.set_xticklabels([])
        ##
        for nc in range(1, no_of_cols):
            if rotate_last_oned_hist and nc == (no_of_cols - 1):
                continue
            ax = axes_array[no_of_rows - 1][nc]
            #new_xticklabels = [ll.get_text() for ll in ax.get_xticklabels()]
            new_xticklabels = ax.get_xticks().tolist()
            new_xticklabels[0] = ''
            ax.set_xticklabels(new_xticklabels)
        #fig.subplots_adjust(wspace=0, hspace=0)
        if plot_type=='contour' and return_areas_in_contours and debug:
            return fig, axes_array, contour_areas, contour.get_paths(), im
        elif plot_type=='contour' and return_areas_in_contours:
            return fig, axes_array, contour_areas
        else:
            return fig, axes_array
    #}}}

