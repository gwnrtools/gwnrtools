# Copyright (C) 2019 Prayush Kumar
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
"""Utilities to make corner-plots with different data structures"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde

from gwnr.graph import make_filled_contour_plot, ParamLatexLabels
from gwnr.stats import MultiDDistribution
from matplotlib import cm

import logging
logging.getLogger().setLevel(logging.INFO)


def cornerplot_dataframe(df, cols=None):
    '''
Make a cornerplot using data input as a pandas.DataFrame

Inputs:q
-------
    - df   : pandas.DataFrame
    - cols : list of columns to use. Defaults to all.
    '''
    import seaborn as sns

    def my_kdeplot(x, y, **kwargs):
        sns.kdeplot(x, y, shade=True, n_levels=20, cmap=cm.autumn, **kwargs)

    def my_hist(x, **kwargs):
        #sns.kdeplot(x, **kwargs)
        _ = plt.hist(x, bins=25, alpha=0.5, density=True, color='red')
        #plt.gca().set_ylim(0, 1)

    if cols is not None:
        df = df[cols]
    g = sns.PairGrid(df)
    g.map_upper(my_kdeplot)
    g.map_lower(my_kdeplot)
    g.map_diag(my_hist, lw=3, legend=False)
    return g


class CornerPlot(MultiDDistribution):
    '''
Inputs:
-------
    - data : >=2 dimensional iterable type (list, array). Different dimensions are
in different columns, while a single column should contain samples for that
dimension

    - var_type: str
    Type of data variables:
        - c : continuous
        - u : unordered (discrete)
        - o : ordered (discrete)
    The string should contain a type specifier for each variable, so for
    example ``var_type='ccuo'``.

    - data_kde: KDE estimator class object with a member function called
                "evaluate". E.g. use KDEUnivariate class from
                `statsmodels.nonparametric.kde`
    '''
    def __init__(self, data, var_type='', *args, **kwargs):
        if var_type == '':
            n_vars = min(np.shape(data))
            var_type = 'c' * n_vars
        try:
            super(CornerPlot, self).__init__(data, var_type, *args, **kwargs)
        except TypeError:
            MultiDDistribution.__init__(self, data, var_type, *args, **kwargs)

    def draw(
            self,
            params_plot,
            true_params_vals=None,  # TRUE / KNOWN values of PARAMETERS
            true_params_color='r',
            params_oned_priors=None,  # PRIOR DISTRIBUTION FOR PARAMETERS
            fig=None,  # CAN PLOT ON EXISTING FIGURE
            axes_array=None,  # NEED ARRAY OF AXES IF PLOTTING ON EXISTING FIGURE
            histogram_type='bar',  # Histogram type (bar / step / barstacked)
            nhbins=30,  # NO OF BINS IN HISTOGRAMS
            fontsize=18,
            projection='rectilinear',
            label='',  # LABEL THAT GOES ON EACH PANEL
            params_labels={},
            plim_low=None,
            plim_high=None,
            legend=False,
            legend_fontsize=18,
            plot_type='scatter',  # SCATTER OR CONTOUR
            color=None,
            hist_alpha=0.3,  # Transparency of histograms
            scatter_alpha=0.2,  # Transparency of scatter points
            npixels=50,
            param_color=None,  # 3RD DIMENSION SHOWN AS COLOR
            param_color_label=None,  # LABEL of 3RD DIMENSION SHOWN AS COLOR
            color_max=None,
            color_min=None,
            cmap=cm.plasma_r,
            contour_args={},
            contour_levels=[68.27, 90.0, 95.45],
            contour_lstyles=[
                "solid", "dashed", "dashdot", "dotted", "solid", "dashed",
                "dashdot", "dotted"
            ],
            label_contours=True,  # Whether or not to label individuals
            contour_labels_inline=True,
            contour_labels_loc="upper center",
            return_areas_in_contours=False,
            grid_twod_on=True,
            label_oned_hists=[0],  # Which one-d histograms to label?
            skip_oned_hists=False,
            label_oned_loc='outside',
            show_oned_median=False,
            show_oned_percentiles=90.0,
            grid_oned_on=False,
            figure_title='',
            debug=False,
            verbose=None):
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
(2) [OPTIONAL] params_labels: dict of parameter names to use in latex labels.
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
        # Preliminary checks on inputs
        if len(contour_levels) > len(contour_lstyles):
            if plot_type == 'contour':
                raise IOError(
                    "Please provide as many linestyles as contour levels")

        # Local verbosity level takes precedence, else the class's is used
        if verbose == None:
            verbose = self.verbose

        # Set all fonts on figure
        plt.rcParams.update({'font.size': fontsize})

        # IF no labels are provided by User, use default Latex labels for CBCs
        cbc_labels = ParamLatexLabels()
        for p in cbc_labels:
            if p not in params_labels:
                params_labels[p] = cbc_labels[p]

        def get_param_label(pp):
            if params_labels is not None and pp in params_labels:
                return params_labels[pp]
            if param_color is not None and pp in param_color and param_color_label is not None:
                return param_color_label
            return pp.replace('_', '-')

        # This is the label for the entire data set, not individual quantities
        label = label.replace('_', '-')

        # Configure the full figure panel
        no_of_rows = len(params_plot)
        no_of_cols = len(params_plot)

        def get_current_axis(_nr, _nc):
            if no_of_rows == 1 and no_of_cols == 1:
                return axes_array
            return axes_array[_nr][_nc]

        # Assign impossible limits
        old_xaxis_lims = np.empty((no_of_rows, no_of_cols, 2))
        old_xaxis_lims[:, :, 0] = 1.e99
        old_xaxis_lims[:, :, 1] = -1.e99
        old_yaxis_lims = np.empty((no_of_rows, no_of_cols, 2))
        old_yaxis_lims[:, :, 0] = 1.e99
        old_yaxis_lims[:, :, 1] = -1.e99

        reusing_figure = False
        if type(fig) != matplotlib.figure.Figure or axes_array is None:
            fig, axes_array = plt.subplots(no_of_rows,
                                           no_of_cols,
                                           figsize=(6 * no_of_cols,
                                                    4 * no_of_rows),
                                           gridspec_kw={
                                               'wspace': 0,
                                               'hspace': 0
                                           })
        else:
            reusing_figure = True
            for nr in range(no_of_rows):
                for nc in range(no_of_cols):
                    if nc > nr:
                        continue
                    old_xaxis_lims[nr,
                                   nc, :] = get_current_axis(nr,
                                                             nc).get_xlim()
                    old_yaxis_lims[nr,
                                   nc, :] = get_current_axis(nr,
                                                             nc).get_ylim()


        # Pre-choose color for 1D histograms (and scatter plots, if applicable)
        rand_color = np.random.rand(3, )
        if color != None:
            rand_color = color

        # Will store teh colorbar axis in this, if needed
        token_cb_ax = None

        if return_areas_in_contours:
            contour_areas = {}
        contour_levels = sorted(contour_levels, reverse=True)

        # Start drawing panels
        for nr in range(no_of_rows):
            for nc in range(no_of_cols):
                # Get current axis
                ax = get_current_axis(nr, nc)

                # We keep the upper diagonal half of the figure empty.
                # FIXME: Could we use it for same data, different visualization?
                if nc > nr:
                    try:
                        fig.delaxes(ax)
                    except:
                        pass
                    continue

                # Make 1D histograms along the diagonal
                if nc == nr:
                    if skip_oned_hists:
                        continue

                    p1 = params_plot[nc]
                    p1label = get_param_label(p1)

                    # Plot known / injected / true value if given
                    if true_params_vals != None:
                        p_true_val = true_params_vals[nc]
                        if p_true_val != None:
                            ax.axvline(p_true_val,
                                       lw=0.5,
                                       ls='solid',
                                       color=true_params_color)

                    # Plot one-d posterior
                    _data = self.sliced(p1).data()
                    im = ax.hist(_data,
                                 bins=nhbins,
                                 histtype=histogram_type,
                                 density=True,
                                 alpha=hist_alpha,
                                 color=rand_color,
                                 label=label)

                    # Plot percentiles
                    if show_oned_percentiles and show_oned_percentiles > 0. and show_oned_percentiles < 100.:
                        low_perc = (100. - show_oned_percentiles) * 0.5
                        high_perc = 100. - low_perc
                        percentile_5 = np.percentile(_data, low_perc)
                        ax.axvline(percentile_5,
                                   lw=1,
                                   ls='dashed',
                                   color=rand_color,
                                   alpha=1)
                        ax.text(percentile_5,
                                ax.get_ylim()[-1],
                                '{:0.02f}'.format(percentile_5),
                                rotation=45,
                                rotation_mode='anchor')
                        # 95%ile
                        percentile_95 = np.percentile(_data, high_perc)
                        ax.axvline(percentile_95,
                                   lw=1,
                                   ls='dashed',
                                   color=rand_color,
                                   alpha=1)
                        ax.text(percentile_95,
                                ax.get_ylim()[-1],
                                '{:0.02f}'.format(percentile_95),
                                rotation=45,
                                rotation_mode='anchor')

                    # Plot median
                    if show_oned_median:
                        ax.axvline(np.median(_data), ls='-', color=rand_color)
                        ax.text(np.median(_data),
                                ax.get_ylim()[-1],
                                '{:0.02f}'.format(np.median(_data)),
                                rotation=45,
                                rotation_mode='anchor')

                    # Add legends to 1D panels
                    try:
                        if legend and (label_oned_hists == -1
                                       or nc in label_oned_hists):
                            if label_oned_loc is not 'outside' and label_oned_loc is not '':
                                ax.legend(loc=label_oned_loc,
                                          fontsize=legend_fontsize)
                            else:
                                ax.legend(fontsize=legend_fontsize,
                                          bbox_to_anchor=(1.3, 0.9))
                    except TypeError:
                        raise TypeError(
                            "Pass a list of labels using `label_oned_hists`")

                    if params_oned_priors is not None and p1 in params_oned_priors:
                        _data = params_oned_priors[p1]
                        if plim_low is not None and plim_high is not None:
                            _prior_xrange = (plim_low[nc], plim_high[nc])
                        else:
                            _prior_xrange = None
                        im = ax.hist(_data,
                                     bins=nhbins,
                                     histtype="step",
                                     color='k',
                                     range=_prior_xrange,
                                     density=True)

                    if plim_low is not None and plim_high is not None:
                        ax.set_xlim(plim_low[nc], plim_high[nc])
                    else:
                        ax.set_xlim(
                            np.min(_data) - 0.1 *
                            (np.max(_data) - np.min(_data)),
                            np.max(_data) + 0.1 *
                            (np.max(_data) - np.min(_data)))

                    ax.grid(grid_oned_on)

                    if nr == (no_of_rows - 1):
                        ax.set_xlabel(p1label)
                    if nc == 0 and (no_of_cols > 1 or no_of_rows > 1):
                        ax.set_ylabel(p1label)
                    if nr < (no_of_rows - 1):
                        ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    continue

                # The following draws 2D panels

                # Plot known / injected / true value if given
                if true_params_vals != None:
                    pc_true_val = true_params_vals[nc]
                    pr_true_val = true_params_vals[nr]
                    if pc_true_val != None:
                        ax.axvline(pc_true_val,
                                   lw=0.5,
                                   ls='solid',
                                   color=true_params_color)
                    if pr_true_val != None:
                        ax.axhline(pr_true_val,
                                   lw=0.5,
                                   ls='solid',
                                   color=true_params_color)
                    if pc_true_val != None and pr_true_val != None:
                        ax.plot([pc_true_val], [pr_true_val],
                                's',
                                color=true_params_color)

                # Now plot what the user requested
                # If user asks for scatter-point colors to be a 3rd dimension
                if param_color in self.var_names:
                    if plot_type == 'scatter':
                        p1 = params_plot[nc]
                        p2 = params_plot[nr]
                        p1label = get_param_label(p1)
                        p2label = get_param_label(p2)
                        cblabel = get_param_label(param_color)
                        if verbose:
                            logging.info(
                                "Scatter plot w color: %s vs %s vs %s" %
                                (p1, p2, param_color))
                        _d1, _d2 = self.sliced(p1).data(), self.sliced(
                            p2).data()
                        im = ax.scatter(_d1,
                                        _d2,
                                        c=self.sliced(param_color).data(),
                                        alpha=scatter_alpha,
                                        edgecolors=None,
                                        linewidths=0,
                                        vmin=color_min,
                                        vmax=color_max,
                                        cmap=cmap,
                                        label=label)

                        token_cb_ax = (im, ax)

                        if nr == (no_of_rows - 1):
                            ax.set_xlabel(p1label)
                        if nc == 0:
                            ax.set_ylabel(p2label)
                        # set X/Y axis limits
                        if plim_low is not None and plim_high is not None:
                            ax.set_xlim(plim_low[nc], plim_high[nc])
                            ax.set_ylim(plim_low[nr], plim_high[nr])
                        else:
                            ax.set_xlim(
                                np.min(_d1) - 0.1 *
                                (np.max(_d1) - np.min(_d1)),
                                np.max(_d1) + 0.1 *
                                (np.max(_d1) - np.min(_d1)))
                            ax.set_ylim(
                                np.min(_d2) - 0.1 *
                                (np.max(_d2) - np.min(_d2)),
                                np.max(_d2) + 0.1 *
                                (np.max(_d2) - np.min(_d2)))

                        ax.grid(grid_twod_on)
                    elif 'contour' in plot_type:
                        p1 = params_plot[nc]
                        p2 = params_plot[nr]
                        p1label = get_param_label(p1)
                        p2label = get_param_label(p2)
                        cblabel = get_param_label(param_color)

                        _d1 = self.sliced(p1).data()
                        _d2 = self.sliced(p2).data()
                        _d3 = self.sliced(param_color).data()

                        token_cb_ax = make_filled_contour_plot(
                            _d1,
                            _d2,
                            _d3,
                            ax=ax,
                            interp_func='griddata',
                            interp_func_args=contour_args,
                            add_colorbar=False)

                        if nr == (no_of_rows - 1):
                            ax.set_xlabel(p1label)
                        if nc == 0:
                            ax.set_ylabel(p2label)
                        # set X/Y axis limits
                        if plim_low is not None and plim_high is not None:
                            ax.set_xlim(plim_low[nc], plim_high[nc])
                            ax.set_ylim(plim_low[nr], plim_high[nr])
                        else:
                            ax.set_xlim(
                                np.min(_d1) - 0.1 *
                                (np.max(_d1) - np.min(_d1)),
                                np.max(_d1) + 0.1 *
                                (np.max(_d1) - np.min(_d1)))
                            ax.set_ylim(
                                np.min(_d2) - 0.1 *
                                (np.max(_d2) - np.min(_d2)),
                                np.max(_d2) + 0.1 *
                                (np.max(_d2) - np.min(_d2)))

                elif param_color is not None:
                    raise IOError("Could not find parameter %s to show" %
                                  param_color)
                # If user asks for scatter plot without 3rd Dimension info
                elif plot_type == 'scatter':
                    p1 = params_plot[nc]
                    p2 = params_plot[nr]
                    p1label = get_param_label(p1)
                    p2label = get_param_label(p2)
                    if verbose:
                        logging.info("Scatter plot: %s vs %s" % (p1, p2))
                    _d1, _d2 = self.sliced(p1).data(), self.sliced(p2).data()
                    im = ax.scatter(_d1,
                                    _d2,
                                    c=rand_color,
                                    alpha=scatter_alpha,
                                    edgecolors=None,
                                    linewidths=0,
                                    label=label)
                    if nr == (no_of_rows - 1):
                        ax.set_xlabel(p1label)
                    if nc == 0:
                        ax.set_ylabel(p2label)
                    if plim_low is not None and plim_high is not None:
                        ax.set_xlim(plim_low[nc], plim_high[nc])
                        ax.set_ylim(plim_low[nr], plim_high[nr])
                    else:
                        ax.set_xlim(
                            np.min(_d1) - 0.1 * (np.max(_d1) - np.min(_d1)),
                            np.max(_d1) + 0.1 * (np.max(_d1) - np.min(_d1)))
                        ax.set_ylim(
                            np.min(_d2) - 0.1 * (np.max(_d2) - np.min(_d2)),
                            np.max(_d2) + 0.1 * (np.max(_d2) - np.min(_d2)))

                    ax.grid(grid_twod_on)
                # If user asks for contour plot without 3rd Dimension info
                elif plot_type == 'contour':
                    p1 = params_plot[nc]
                    p2 = params_plot[nr]
                    p1label = get_param_label(p1)
                    p2label = get_param_label(p2)
                    if verbose:
                        logging.info("Contour plot: {} vs {}".format(p1, p2))
                    # Get data
                    d1 = self.sliced(p1).data()
                    d2 = self.sliced(p2).data()
                    dd = np.column_stack([d1, d2])
                    pdf = gaussian_kde(dd.T)
                    # Get contour levels
                    zlevels = [
                        np.percentile(pdf(dd.T), 100.0 - lev)
                        for lev in contour_levels
                    ]
                    x11vals = np.linspace(dd[:, 0].min(), dd[:, 0].max(),
                                          npixels)
                    x12vals = np.linspace(dd[:, 1].min(), dd[:, 1].max(),
                                          npixels)
                    q, w = np.meshgrid(x11vals, x12vals)
                    r1 = pdf([q.flatten(), w.flatten()])
                    r1.shape = q.shape
                    # Draw contours
                    im = ax.contour(
                        x11vals,
                        x12vals,
                        r1,
                        zlevels,
                        colors=[rand_color],
                        linestyles=contour_lstyles[:len(contour_levels)],
                        label=label)

                    # Get area inside contour
                    if return_areas_in_contours:
                        if verbose:
                            logging.info("Computing area inside contours.")
                        contour_areas[p1 + p2] = []
                        for ii in range(len(zlevels)):
                            contour = im.collections[ii]
                            # Add areas inside all independent contours, in case
                            # there are multiple disconnected ones
                            contour_areas[p1 + p2].append(
                                np.sum([
                                    area_inside_contour(vs.vertices)
                                    for vs in contour.get_paths()
                                ]))
                            if verbose:
                                logging.info("Total area = {}, {}".format(
                                    contour_areas[p1 + p2][-1]))
                            if debug:
                                for _i, vs in enumerate(contour.get_paths()):
                                    logging.info("sub-area {}: {}".format(
                                        _i, area_inside_contour(vs.vertices)))
                        contour_areas[p1 + p2] = np.array(contour_areas[p1 +
                                                                        p2])

                    ####
                    # BEAUTIFY contour labeling..!
                    # Define a class that forces representation of float to look
                    # a certain way. This remove trailing zero so '1.0' becomes '1'
                    class nf(float):
                        def __repr__(self):
                            str = '%.1f' % (self.__float__(), )
                            if str[-1] == '0':
                                return '%.0f' % self.__float__()
                            else:
                                return '%.1f' % self.__float__()

                    # Recast levels to new class
                    im.levels = [nf(val) for val in contour_levels]
                    # Label levels with specially formatted floats
                    if plt.rcParams["text.usetex"]:
                        fmt = r'%r \%%'
                    else:
                        fmt = '%r %%'
                    ####
                    if label_contours:
                        if contour_labels_inline:
                            ax.clabel(im,
                                      im.levels,
                                      inline=False,
                                      use_clabeltext=True,
                                      fmt=fmt,
                                      fontsize=10)
                        else:
                            for zdx, _ in enumerate(zlevels):
                                _ = ax.plot(
                                    [], [],
                                    color=rand_color,
                                    ls=contour_lstyles[:len(contour_levels)]
                                    [zdx],
                                    label=im.levels[zdx])
                            if legend:
                                ax.legend(loc=contour_labels_loc,
                                          fontsize=legend_fontsize)
                    else:
                        pass
                    #
                    if nr == (no_of_rows - 1):
                        ax.set_xlabel(p1label)
                    if nc == 0:
                        ax.set_ylabel(p2label)
                    if plim_low is not None and plim_high is not None:
                        ax.set_xlim(plim_low[nc], plim_high[nc])
                        ax.set_ylim(plim_low[nr], plim_high[nr])
                    elif projection != "mollweide":
                        ax.set_xlim(
                            np.min(d1) - 0.1 * (np.max(d1) - np.min(d1)),
                            np.max(d1) + 0.1 * (np.max(d1) - np.min(d1)))
                        ax.set_ylim(
                            np.min(d2) - 0.1 * (np.max(d2) - np.min(d2)),
                            np.max(d2) + 0.1 * (np.max(d2) - np.min(d2)))
                    else:
                        pass
                    ax.grid(grid_twod_on)
                else:
                    raise IOError("plot type %s not supported.." % plot_type)
                if nc != 0:
                    ax.set_yticklabels([])
                if nr != (no_of_rows - 1):
                    ax.set_xticklabels([])
        ##
        # Draw colorbar on its own axis
        if token_cb_ax is not None:
            im, ax = token_cb_ax
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.35, 0.05, 0.7])
            cb = fig.colorbar(im, cax=cbar_ax)
            cb.set_label(cblabel)
        ##
        # Now, adjust axis limits
        final_xlims = np.empty((no_of_cols, 2))
        for nc in range(no_of_cols):
            final_xlims[nc, :] = (np.min(old_xaxis_lims[:, nc, 0]),
                                  np.max(old_xaxis_lims[:, nc, 1]))

        for nr in range(no_of_rows):
            for nc in range(no_of_cols):
                if nc > nr:
                    continue
                ax = get_current_axis(nr, nc)

                curr_xlim = ax.get_xlim()
                final_xlims[nc, :] = (np.min([
                    final_xlims[nc, 0], curr_xlim[0]
                ]), np.max([final_xlims[nc, 1], curr_xlim[1]]))

        # Impose axis limits, y axis' limits are imposed by symmetry
        for nr in range(no_of_rows):
            for nc in range(no_of_cols):
                if nc > nr:
                    continue
                ax = get_current_axis(nr, nc)
                ax.set_xlim(*final_xlims[nc, :])
                if nc < nr:
                    ax.set_ylim(*final_xlims[nr, :])

        # Adjust figure title
        if len(figure_title) > 0:
            fig.suptitle(figure_title)

        # Adjust xticklabels to remove the leftmost tick
        for nc in range(1, no_of_cols):
            ax = axes_array[no_of_rows - 1][nc]
            #new_xticklabels = [ll.get_text() for ll in ax.get_xticklabels()]
            new_xticklabels = ax.get_xticks().tolist()
            new_xticklabels[0] = ''
            ax.set_xticklabels(new_xticklabels)

        # Conditionally return objects
        if plot_type == 'contour' and return_areas_in_contours and debug:
            return fig, axes_array, contour_areas, contour.get_paths(), im
        elif plot_type == 'contour' and return_areas_in_contours:
            return fig, axes_array, contour_areas
        else:
            return fig, axes_array