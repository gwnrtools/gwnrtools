"""
Generate 2-D, bounded-kde contour plots and 1D histograms of the masses for all events (Figure 2 right now)

"""

from __future__ import division

import numpy as np

import os
import cPickle as pickle

import pandas as pd

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

from matplotlib.ticker import Formatter
from matplotlib import patheffects as PathEffects

from scipy.spatial import Delaunay

#import seaborn as sns
import seaborn.apionly as sns

sns.set_style('ticks')
sns.set_context('talk')
#sns.set_style({'font.family':'Times New Roman'})

rc_params = {
    'backend': 'ps',
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'font.size': 13,
    'legend.fontsize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    #'text.usetex': True,
    'font.family': 'Times New Roman'
}  #,
#'font.sans-serif': ['Bitstream Vera Sans']}#,

#matplotlib.rc('font', family='serif', serif='Times New Roman')
plt.rcParams.update(rc_params)

annotate = False

plot_m1m2_detector = False
plot_m1m2 = True
plot_m1m2_allevents = False
plot_distinc = False
plot_effspin = True
plot_finalma = False
plot_a1a2 = False
plot_spindisk = False

# Define model colors (IMRPhenom, EOBNR)
colors = ["steelblue", "firebrick"]


class DegreeFormatter(Formatter):
    def __call__(self, x, pos=None):
        return r"${:3.0f}^{{\circ}}$".format(x)


def ms2q(pts):
    pts = np.atleast_2d(pts)

    m1 = pts[:, 0]
    m2 = pts[:, 1]
    mc = np.power(m1 * m2, 3. / 5.) * np.power(m1 + m2, -1. / 5.)
    q = m2 / m1
    return np.column_stack([mc, q])


def plot_bounded_2d_kde(data,
                        data2=None,
                        contour_data=None,
                        levels=None,
                        xlow=None,
                        xhigh=None,
                        ylow=None,
                        yhigh=None,
                        transform=None,
                        shade=False,
                        vertical=False,
                        bw="scott",
                        gridsize=500,
                        cut=3,
                        clip=None,
                        legend=True,
                        shade_lowest=True,
                        ax=None,
                        **kwargs):

    if ax is None:
        ax = plt.gca()

    if transform is None:
        transform = lambda x: x

    data = data.astype(np.float64)
    if data2 is not None:
        data2 = data2.astype(np.float64)

    bivariate = False
    if isinstance(data, np.ndarray) and np.ndim(data) > 1:
        bivariate = True
        x, y = data.T
    elif isinstance(data, pd.DataFrame) and np.ndim(data) > 1:
        bivariate = True
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
    elif data2 is not None:
        bivariate = True
        x = data
        y = data2
    if bivariate == False:
        raise TypeError("Bounded 2D KDE are only available for"
                        "bivariate distributions.")
    else:
        # Determine the clipping
        if clip is None:
            clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
        elif np.ndim(clip) == 1:
            clip = [clip, clip]

        xx = contour_data['xx']
        yy = contour_data['yy']
        z = contour_data['z']
        den = contour_data['kde']
        kde_sel = contour_data['kde_sel']
        Nden = len(den)

        # Calculate the KDE
        if levels is None:
            kde_pts = transform(np.array([x, y]).T)
            #post_kde = Bounded_2d_kde(kde_pts, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh)
        else:
            # pts=np.array([x, y]).T
            # Npts=pts.shape[0]
            # kde_pts = transform(pts[:Npts//2,:])
            # untransformed_den_pts = pts[Npts//2:,:]
            # den_pts = transform(untransformed_den_pts)

            # Nden = den_pts.shape[0]

            # post_kde=Bounded_2d_kde(kde_pts, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh)
            # den = post_kde(den_pts)
            # inds = np.argsort(den)[::-1]
            # den = den[inds]

            zvalues = np.empty(len(levels))
            for i, level in enumerate(levels):
                ilevel = int(np.ceil(Nden * level))
                ilevel = min(ilevel, Nden - 1)
                zvalues[i] = den[ilevel]
            zvalues.sort()

        # deltax = x.max() - x.min()
        # deltay = y.max() - y.min()
        # x_pts = np.linspace(x.min() - .1*deltax, x.max() + .1*deltax, gridsize)
        # y_pts = np.linspace(y.min() - .1*deltay, y.max() + .1*deltay, gridsize)

        # xx, yy = np.meshgrid(x_pts, y_pts)

        # positions = np.column_stack([xx.ravel(), yy.ravel()])

        # z = np.reshape(post_kde(transform(positions)), xx.shape)

        # Black (thin) contours with while outlines by default
        kwargs['linewidths'] = kwargs.get('linewidths', 1.)

        # Plot the contours
        n_levels = kwargs.pop("n_levels", 10)
        # cmap = kwargs.get("cmap", None)

        # if cmap is None:
        #     kwargs['colors'] = kwargs.get('colors', 'k')
        # if isinstance(cmap, string_types):
        #     if cmap.endswith("_d"):
        #         pal = ["#333333"]
        #         pal.extend(sns.palettes.color_palette(cmap.replace("_d", "_r"), 2))
        #         cmap = sns.palettes.blend_palette(pal, as_cmap=True)
        #     else:
        #         cmap = plt.cm.get_cmap(cmap)

        #kwargs["cmap"] = cmap
        contour_func = ax.contourf if shade else ax.contour
        if levels:
            cset = contour_func(xx, yy, z, zvalues, **kwargs)

            for i, coll in enumerate(cset.collections):
                level = coll.get_paths()[0]
                contour_hull = Delaunay(level.vertices)
                #print("{}% of points found within {}% contour".format(int(100*np.count_nonzero(contour_hull.find_simplex(untransformed_den_pts) > -1)/Nden),
                #                                                      int(100*levels[::-1][i])))

            # Add white outlines
            if kwargs['colors'] == 'k':
                plt.setp(cset.collections,
                         path_effects=[
                             PathEffects.withStroke(linewidth=1.5,
                                                    foreground="w")
                         ])
            fmt = {}
            strs = ['{}%'.format(int(100 * level)) for level in levels[::-1]]
            for l, s in zip(cset.levels, strs):
                fmt[l] = s

            plt.clabel(cset,
                       cset.levels,
                       fmt=fmt,
                       fontsize=11,
                       manual=False,
                       **kwargs)
            plt.setp(cset.labelTexts,
                     color='k',
                     path_effects=[
                         PathEffects.withStroke(linewidth=1.5, foreground="w")
                     ])
        else:
            cset = contour_func(xx, yy, z, n_levels, **kwargs)

        if shade and not shade_lowest:
            cset.collections[0].set_alpha(0)

        # Avoid gaps between patches when saving as PDF
        if shade:
            for c in cset.collections:
                c.set_edgecolor("face")

        kwargs["n_levels"] = n_levels

        # Label the axes
        if hasattr(x, "name") and legend:
            ax.set_xlabel(x.name)
        if hasattr(y, "name") and legend:
            ax.set_ylabel(y.name)

    return ax


def plot_bounded_2d_kde_otherevents(data,
                                    data2=None,
                                    contour_data=None,
                                    levels=None,
                                    xlow=None,
                                    xhigh=None,
                                    ylow=None,
                                    yhigh=None,
                                    transform=None,
                                    shade=False,
                                    vertical=False,
                                    bw="scott",
                                    gridsize=500,
                                    cut=3,
                                    clip=None,
                                    legend=True,
                                    shade_lowest=True,
                                    ax=None,
                                    **kwargs):

    if ax is None:
        ax = plt.gca()

    if transform is None:
        transform = lambda x: x

    data = data.astype(np.float64)
    if data2 is not None:
        data2 = data2.astype(np.float64)

    bivariate = False
    if isinstance(data, np.ndarray) and np.ndim(data) > 1:
        bivariate = True
        x, y = data.T
    elif isinstance(data, pd.DataFrame) and np.ndim(data) > 1:
        bivariate = True
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
    elif data2 is not None:
        bivariate = True
        x = data
        y = data2

    if bivariate == False:
        raise TypeError("Bounded 2D KDE are only available for"
                        "bivariate distributions.")
    else:
        # Determine the clipping
        if clip is None:
            clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
        elif np.ndim(clip) == 1:
            clip = [clip, clip]

        xx = contour_data['xx']
        yy = contour_data['yy']
        z = contour_data['z']
        den = contour_data['kde']
        kde_sel = contour_data['kde_sel']
        Nden = len(den)

        # Calculate the KDE
        if levels is None:
            kde_pts = transform(np.array([x, y]).T)
            #post_kde = Bounded_2d_kde(kde_pts, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh)
        else:
            # pts=np.array([x, y]).T
            # Npts=pts.shape[0]
            # kde_pts = transform(pts[:Npts//2,:])
            # untransformed_den_pts = pts[Npts//2:,:]
            # den_pts = transform(untransformed_den_pts)

            # Nden = den_pts.shape[0]

            # post_kde=Bounded_2d_kde(kde_pts, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh)
            # den = post_kde(den_pts)
            # inds = np.argsort(den)[::-1]
            # den = den[inds]

            zvalues = np.empty(len(levels))
            for i, level in enumerate(levels):
                ilevel = int(np.ceil(Nden * level))
                ilevel = min(ilevel, Nden - 1)
                zvalues[i] = den[ilevel]
            zvalues.sort()

        # deltax = x.max() - x.min()
        # deltay = y.max() - y.min()
        # x_pts = np.linspace(x.min() - .1*deltax, x.max() + .1*deltax, gridsize)
        # y_pts = np.linspace(y.min() - .1*deltay, y.max() + .1*deltay, gridsize)

        # xx, yy = np.meshgrid(x_pts, y_pts)

        # positions = np.column_stack([xx.ravel(), yy.ravel()])

        # z = np.reshape(post_kde(transform(positions)), xx.shape)

        # Black (thin) contours with while outlines by default
        kwargs['linewidths'] = kwargs.get('linewidths', 1.5)

        # Plot the contours
        n_levels = kwargs.pop("n_levels", 10)
        # cmap = kwargs.get("cmap", None)

        # if cmap is None:
        #     kwargs['colors'] = kwargs.get('colors', 'k')
        # if isinstance(cmap, string_types):
        #     if cmap.endswith("_d"):
        #         pal = ["#333333"]
        #         pal.extend(sns.palettes.color_palette(cmap.replace("_d", "_r"), 2))
        #         cmap = sns.palettes.blend_palette(pal, as_cmap=True)
        #     else:
        #         cmap = plt.cm.get_cmap(cmap)

        # #kwargs["cmap"] = cmap
        contour_func = ax.contourf if shade else ax.contour

        if levels:
            cset = contour_func(xx, yy, z, zvalues, **kwargs)

            for i, coll in enumerate(cset.collections):
                level = coll.get_paths()[0]
                contour_hull = Delaunay(level.vertices)
                #print("{}% of points found within {}% contour".format(int(100*np.count_nonzero(contour_hull.find_simplex(untransformed_den_pts) > -1)/Nden),
                #                                                      int(100*levels[::-1][i])))

            # Add white outlines
            if kwargs['colors'] == 'k':
                plt.setp(cset.collections,
                         path_effects=[
                             PathEffects.withStroke(linewidth=1.5,
                                                    foreground="w")
                         ])
            fmt = {}
            strs = ['{}%'.format(int(100 * level)) for level in levels[::-1]]
            for l, s in zip(cset.levels, strs):
                fmt[l] = s

            plt.clabel(cset,
                       cset.levels,
                       fmt=fmt,
                       fontsize=7,
                       manual=False,
                       **kwargs)
            plt.setp(cset.labelTexts,
                     color='k',
                     path_effects=[
                         PathEffects.withStroke(linewidth=1.5, foreground="w")
                     ])
        else:
            cset = contour_func(xx, yy, z, n_levels, **kwargs)

        if shade and not shade_lowest:
            cset.collections[0].set_alpha(0)

        # Avoid gaps between patches when saving as PDF
        if shade:
            for c in cset.collections:
                c.set_edgecolor("face")

        kwargs["n_levels"] = n_levels

        # Label the axes
        if hasattr(x, "name") and legend:
            ax.set_xlabel(x.name)
        if hasattr(y, "name") and legend:
            ax.set_ylabel(y.name)

    return ax


def unpickle_contour_data(post_name):
    #infile = os.path.join('', 'folded_{}_contour_data.pkl'.format(post_name))
    infile = os.path.join('', '{}_contour_data.pkl'.format(post_name))
    with open(infile, 'r') as inp:
        cdata = pickle.load(inp)
    return cdata


#xlim=None, ylim=None, cmap='Purples',
def plot_2d_mass(params,
                 pos_HL,
                 cdata_HL,
                 model_posts=None,
                 labels=None,
                 xlim=None,
                 ylim=None,
                 cmap='Purples',
                 colors=colors,
                 **kwargs):
    """ **kwargs are passed to plot_bounded_2d_kde """
    if labels is None:
        labels = [None for p in params]

    postHL_series = [
        pd.Series(pos_HL[p], name=label) for p, label in zip(params, labels)
    ]
    #postHLV_series = [pd.Series(pos_HLV[p], name=label) for p, label in zip(params, labels)]
    #postHLV_series = [pd.Series(pos_HLV[p], name=label) for p, label in zip(params, labels)]

    model_post_series = []
    if model_posts is not None:
        for p, label in zip(params, labels):
            model_post_series.append(
                [pd.Series(pos[p], name=label) for pos in model_posts])

    ratio = 5
    credible_interval = np.array([5, 95])
    gs = plt.GridSpec(ratio + 1, ratio + 1)

    g = sns.JointGrid(postHL_series[0],
                      postHL_series[1],
                      xlim=xlim,
                      ylim=ylim,
                      space=0,
                      ratio=ratio)  # Need to specify xlim and ylim?
    g = g.plot_joint(plot_bounded_2d_kde,
                     contour_data=cdata_HL,
                     cmap=cmap,
                     shade=True,
                     shade_lowest=False,
                     n_levels=30,
                     **kwargs)
    #g = g.plot_joint(plot_bounded_2d_kde, contour_data=cdata_HL,colors='k', levels=[.5, .9], **kwargs)
    plot_bounded_2d_kde_otherevents(postHL_series[0],
                                    postHL_series[1],
                                    cdata_HL,
                                    xlim=xlim,
                                    ylim=ylim,
                                    levels=[.5, .9],
                                    colors='k',
                                    gridsize=500,
                                    **kwargs)

    n, _, _ = g.ax_marg_x.hist(postHL_series[0],
                               color=".7",
                               bins=50,
                               histtype='stepfilled',
                               linewidth=1.5,
                               normed=True)
    nmax = np.max(n)

    n, _, _ = g.ax_marg_x.hist(postHL_series[0],
                               color="k",
                               bins=50,
                               histtype='step',
                               linewidth=1.5,
                               normed=True)
    nmax = max(nmax, np.max(n))

    # n, _, _ = g.ax_marg_x.hist(postHLV_series[0], color="seagreen", bins=50, histtype='step',linewidth=1,normed=True)
    # nmax = max(nmax,np.max(n))

    # n, _, _ = g.ax_marg_x.hist(postHLV_series[0], color="k", bins=30, histtype='step',linewidth=1.5,normed=True)
    nmax = max(nmax, np.max(n))

    g.ax_marg_x.set_ylim(ymax=1.01 * nmax)

    qvalues = np.percentile(postHL_series[0], [5, 95])
    for q in qvalues:
        g.ax_marg_x.axvline(q, ls="dashed", color='k')
    g.ax_marg_x.set_position(gs[ratio, :-1].get_position(g.fig))

    n, _, _ = g.ax_marg_y.hist(postHL_series[1],
                               orientation='horizontal',
                               color='0.7',
                               bins=30,
                               histtype='stepfilled',
                               linewidth=1.5,
                               normed=True)
    nmax = np.max(n)

    n, _, _ = g.ax_marg_y.hist(postHL_series[1],
                               orientation='horizontal',
                               color='k',
                               bins=30,
                               histtype='step',
                               linewidth=1.5,
                               normed=True)
    nmax = max(nmax, np.max(n))

    # n, _, _  = g.ax_marg_y.hist(postHLV_series[1], orientation='horizontal', color='seagreen', bins=30, histtype='step',linewidth=1,normed=True)
    # nmax = max(nmax,np.max(n))

    # n, _, _  = g.ax_marg_y.hist(postHLV_series[1], orientation='horizontal', color="k", bins=30, histtype='step',linewidth=1.5,normed=True)
    # nmax = np.max(n)

    g.ax_marg_y.set_xlim(xmax=1.01 * nmax)

    qvalues = np.percentile(postHL_series[1], [5, 95])
    for q in qvalues:
        g.ax_marg_y.axhline(q, ls="dashed", color='k')
    g.ax_marg_y.set_position(gs[1:, 0].get_position(g.fig))

    return g


def plot_2d_chieff(params,
                   pos_HL,
                   pos_prior,
                   cdata_HL,
                   model_posts=None,
                   labels=None,
                   xlim=None,
                   ylim=None,
                   cmap='Purples',
                   colors=colors,
                   **kwargs):
    """ **kwargs are passed to plot_bounded_2d_kde """
    if labels is None:
        labels = [None for p in params]

    postHL_series = [
        pd.Series(pos_HL[p], name=label) for p, label in zip(params, labels)
    ]
    postprior_series = [
        pd.Series(pos_prior[p], name=label)
        for p, label in zip(params, labels)
    ]
    #postHLV_series = [pd.Series(pos_HLV[p], name=label) for p, label in zip(params, labels)]

    model_post_series = []
    if model_posts is not None:
        for p, label in zip(params, labels):
            model_post_series.append(
                [pd.Series(pos[p], name=label) for pos in model_posts])

    ratio = 5
    credible_interval = np.array([5, 95])
    gs = plt.GridSpec(ratio + 1, ratio + 1)

    g = sns.JointGrid(postHL_series[0],
                      postHL_series[1],
                      xlim=xlim,
                      ylim=ylim,
                      space=0,
                      ratio=ratio)  # Need to specify xlim and ylim?
    g = g.plot_joint(plot_bounded_2d_kde,
                     contour_data=cdata_HL,
                     cmap=cmap,
                     shade=True,
                     shade_lowest=False,
                     n_levels=30,
                     **kwargs)
    #g = g.plot_joint(plot_bounded_2d_kde, contour_data=cdata_HL,colors='k', levels=[.5, .9], **kwargs)

    plot_bounded_2d_kde_otherevents(postHL_series[0],
                                    postHL_series[1],
                                    cdata_HL,
                                    xlim=xlim,
                                    ylim=ylim,
                                    levels=[.5, .9],
                                    colors='k',
                                    gridsize=500,
                                    **kwargs)

    n, _, _ = g.ax_marg_x.hist(postHL_series[0],
                               color=".7",
                               bins=50,
                               histtype='stepfilled',
                               linewidth=1.5,
                               normed=True)
    nmax = np.max(n)

    n, _, _ = g.ax_marg_x.hist(postHL_series[0],
                               color="k",
                               bins=50,
                               histtype='step',
                               linewidth=1.5,
                               normed=True)
    nmax = max(nmax, np.max(n))

    n, _, _ = g.ax_marg_x.hist(postprior_series[0],
                               color="seagreen",
                               bins=50,
                               histtype='step',
                               linewidth=1,
                               normed=True)
    nmax = max(nmax, np.max(n))

    # n, _, _ = g.ax_marg_x.hist(postHLV_series[0], color="k", bins=30, histtype='step',linewidth=1.5,normed=True)
    nmax = max(nmax, np.max(n))

    g.ax_marg_x.set_ylim(ymax=1.01 * nmax)

    qvalues = np.percentile(postHL_series[0], [5, 95])
    for q in qvalues:
        g.ax_marg_x.axvline(q, ls="dashed", color='k')
    g.ax_marg_x.set_position(gs[ratio, :-1].get_position(g.fig))

    n, _, _ = g.ax_marg_y.hist(postHL_series[1],
                               orientation='horizontal',
                               color='0.7',
                               bins=30,
                               histtype='stepfilled',
                               linewidth=1.5,
                               normed=True)
    nmax = np.max(n)

    n, _, _ = g.ax_marg_y.hist(postHL_series[1],
                               orientation='horizontal',
                               color='k',
                               bins=30,
                               histtype='step',
                               linewidth=1.5,
                               normed=True)
    nmax = max(nmax, np.max(n))

    n, _, _ = g.ax_marg_y.hist(postprior_series[1],
                               orientation='horizontal',
                               color='seagreen',
                               bins=30,
                               histtype='step',
                               linewidth=1,
                               normed=True)
    nmax = max(nmax, np.max(n))

    # n, _, _  = g.ax_marg_y.hist(postHLV_series[1], orientation='horizontal', color="k", bins=30, histtype='step',linewidth=1.5,normed=True)
    # nmax = np.max(n)

    g.ax_marg_y.set_xlim(xmax=1.01 * nmax)

    qvalues = np.percentile(postHL_series[1], [5, 95])
    for q in qvalues:
        g.ax_marg_y.axhline(q, ls="dashed", color='k')
    g.ax_marg_y.set_position(gs[1:, 0].get_position(g.fig))

    return g


###########################################################################
#Posterior Samples are in                                                 #
# https://git.ligo.org/chris-pankow/lvc_pe_samples/tree/svn-migration     #
###########################################################################

# pos_HL = np.genfromtxt('HL_posterior_samples.dat', names=True)
# pos_HLV= np.genfromtxt('HLV_posterior_samples.dat', names=True)

# pos_HL = np.genfromtxt('GW170814_HLV_combined_Prod-1and7-HLVClean_equaISnumbers_addedFMS.dat', names=True)

# pos_HL= np.genfromtxt('GW170814_HLV_combined_Prod-1and7-HLVClean_equaISnumbers_addedFMS.dat', names=True)
# pos_prior= np.genfromtxt('prior.dat', names=True)

# np.random.shuffle(pos_HL)
# np.random.shuffle(pos_prior)

# if plot_distinc:
#     if annotate:
#         labels = ["primary mass $(\mathrm{M}_\odot)$",
#                   "secondary mass $(\mathrm{M}_\odot)$"]
#     else:
#         labels = [r"$\theta_{JN}$", r"$D_\mathrm{L}/\mathrm{Mpc}$"]
#         #labels = ["a","b"]

#     # xlim=(-40, 120)
#     # ylim=(-100, 1300)
#     xlim=(-80, 210)
#     ylim=(-100, 1300)
#     xlow, xhigh = 0., 180.

#     pos_HL['theta_jn'] *=  180./np.pi
#     pos_HLV['theta_jn'] *=  180./np.pi

#     # for i in range(len(pos_HL['theta_jn'])):
#     #     if pos_HL['theta_jn'][i] > 90:
#     #         pos_HL['theta_jn'][i] = 180-pos_HL['theta_jn'][i]
#     # for i in range(len(pos_HLV['theta_jn'])):
#     #     if pos_HLV['theta_jn'][i] > 90:
#     #         pos_HLV['theta_jn'][i] = 180-pos_HLV['theta_jn'][i]

#     cdata_HL = unpickle_contour_data('GW170814_HL_dist_inc')
#     cdata_HLV = unpickle_contour_data('GW170814_HLV_dist_inc')

#     g = plot_2d_allevents(['theta_jn', 'distance'], pos_HL, pos_HLV, cdata_HL, cdata_HLV, xlim=xlim, ylim=ylim, labels=labels, transform=ms2q, yhigh=1.)

#     if annotate:
#         x = np.linspace(xlim[0], xlim[1], 100)
#         plt.fill_between(x, x, ylim[1]*np.ones_like(x), hatch='\\', color='none', edgecolor="k")
#         g.ax_marg_x.patch.set_alpha(0.)
#         g.ax_marg_y.patch.set_alpha(0.)

#     else:
#        g.fig.get_axes()[0].legend([plt.Line2D([0,0], [1,0], color='k', linewidth=1.5),
#                                    plt.Line2D([0,0], [1,0], color="firebrick", linewidth=1.5)],
#                                   ['HL','HLV'], loc='upper right', frameon=False)

#     g.ax_marg_x.axis('off')
#     g.ax_marg_y.axis('off')

#     ax = g.ax_joint
#     ax.xaxis.set_major_formatter(DegreeFormatter())
#     ax.set_yticks([100., 300., 500., 700, 900., 1100.])
#     #ax.set_xticks([0., 30., 60., 90])
#     ax.set_xticks([0., 30., 60., 90,120,150,180])

#     sns.despine(fig=g.fig, trim=True)

#     ax.arrow(190,317, 0, 355, lw=1, width=.002, color='firebrick',head_length=1,head_width=0.1)
#     ax.arrow(200,343, 0, 526, lw=1, width=.002, color='k',head_width=01,head_length=0.1)

#     plt.plot((200,200),(343,343), 'k', marker='v',markersize=5)
#     plt.plot((200,200),(869,869), 'k', marker='^',markersize=5)
#     plt.plot((190,190),(317,317), 'firebrick', marker='v',markersize=5)
#     plt.plot((190,190),(672,672), 'firebrick', marker='^',markersize=5)

#     g.fig.savefig('distance-thetajn.pdf', bbox_inches='tight', bbox_extra_artists=[])
#     g.fig.savefig('distance-thetajn.png', bbox_inches='tight', bbox_extra_artists=[], dpi=600)

if plot_m1m2:
    # if annotate:
    #     labels = ["primary mass $(\mathrm{M}_\odot)$",
    #               "secondary mass $(\mathrm{M}_\odot)$"]
    # else:

    pos_HL = np.genfromtxt(
        'GW170814_HLV_combined_Prod-1and7-HLVClean_equaISnumbers_addedFMS.dat',
        names=True)

    np.random.shuffle(pos_HL)

    labels = ["$m_{1} (\mathrm{M}_\odot)$", "$m_{2} (\mathrm{M}_\odot)$"]
    #labels = ["a","b"]

    # xlim=(-40, 120)
    # ylim=(-100, 1300)
    xlim = (15, 45.)
    ylim = (15, 35.)

    # pos_HL['theta_jn'] *=  180./np.pi
    # pos_HLV['theta_jn'] *=  180./np.pi

    # for i in range(len(pos_HL['theta_jn'])):
    #     if pos_HL['theta_jn'][i] > 90:
    #         pos_HL['theta_jn'][i] = 180-pos_HL['theta_jn'][i]
    # for i in range(len(pos_HLV['theta_jn'])):
    #     if pos_HLV['theta_jn'][i] > 90:
    #         pos_HLV['theta_jn'][i] = 180-pos_HLV['theta_jn'][i]

    cdata_HL = unpickle_contour_data('GW170814_equal_m1_m2')
    # cdata_1 = unpickle_contour_data('GW170104_m1_m2')
    # cdata_2 = unpickle_contour_data('GW150914_m1_m2')
    # cdata_3 = unpickle_contour_data('GW151226_m1_m2')
    # cdata_4 = unpickle_contour_data('LVT151012_m1_m2')

    g = plot_2d_mass(['m1_source', 'm2_source'],
                     pos_HL,
                     cdata_HL,
                     xlim=xlim,
                     ylim=ylim,
                     labels=labels,
                     transform=ms2q,
                     yhigh=1.)

    # if annotate:
    #     x = np.linspace(xlim[0], xlim[1], 100)
    #     plt.fill_between(x, x, ylim[1]*np.ones_like(x), hatch='\\', color='none', edgecolor="k")
    #     g.ax_marg_x.patch.set_alpha(0.)
    #     g.ax_marg_y.patch.set_alpha(0.)

    # else:
    #    g.fig.get_axes()[0].legend([plt.Line2D([0,0], [1,0], color='k', linewidth=1.5),
    #                                plt.Line2D([0,0], [1,0], color="firebrick", linewidth=1.5)],
    #                               ['HL','HLV'], loc='upper right', frameon=False)

    g.ax_marg_x.axis('off')
    g.ax_marg_y.axis('off')

    ax = g.ax_joint
    #ax.xaxis.set_major_formatter(DegreeFormatter())
    ax.set_yticks([15, 20., 25., 30, 35])
    ax.set_xticks([15, 20., 25., 30, 35, 40, 45])
    # ax.set_xticks([0., 30., 60., 90,120,150,180])

    sns.despine(fig=g.fig, trim=True)

    # label_locs = {'GW150914':(43, 26),
    #               'LVT151012':(48, 8.5),
    #               'GW151226':(30, 3),
    #               'GW170104':(43, 18)
    # event_colors = {'GW150914': u'#c4c4c4',
    #                 'LVT151012': u'#c4c4c4',
    #                 'GW151226': u'#c4c4c4',
    #                 'GW170104': 'k'}

    # for event, loc in label_locs.iteritems():
    #     ax.text(loc[0], loc[1], event, color=event_colors[event], fontsize=13)

    # m1, m2 = np.meshgrid(np.linspace(20,50,100), np.linspace(5,30,100))
    # mchirp = ((m1*m2)**(3./5.))*((m1+m2)**(-1./5.))
    # mchirp[m1-m2<0] = np.nan
    # g.ax_joint.contour(m1,m2,mchirp,[21.2237407784], colors='black', linestyles='dashed',linewidth=0.5)

    g.fig.savefig('m1_m2.pdf', bbox_inches='tight', bbox_extra_artists=[])
    g.fig.savefig('m1_m2.png',
                  bbox_inches='tight',
                  bbox_extra_artists=[],
                  dpi=600)

if plot_effspin:
    # if annotate:
    #     labels = ["primary mass $(\mathrm{M}_\odot)$",
    #               "secondary mass $(\mathrm{M}_\odot)$"]
    # else:

    pos_HL = np.genfromtxt('IMR.dat', names=True)
    pos_prior = np.genfromtxt('prior.dat', names=True)

    np.random.shuffle(pos_HL)
    np.random.shuffle(pos_prior)

    labels = ['$\chi_\mathrm{p}$', '$\chi_\mathrm{eff}$']
    #labels = ["a","b"]

    # xlim=(-40, 120)
    # ylim=(-100, 1300)
    xlim = (-0.4, 1.1)
    ylim = (-1.1, 1.1)

    # pos_HL['theta_jn'] *=  180./np.pi
    # pos_HLV['theta_jn'] *=  180./np.pi

    # for i in range(len(pos_HL['theta_jn'])):
    #     if pos_HL['theta_jn'][i] > 90:
    #         pos_HL['theta_jn'][i] = 180-pos_HL['theta_jn'][i]
    # for i in range(len(pos_HLV['theta_jn'])):
    #     if pos_HLV['theta_jn'][i] > 90:
    #         pos_HLV['theta_jn'][i] = 180-pos_HLV['theta_jn'][i]

    cdata_HL = unpickle_contour_data('GW170814_IMR_chieff_chip')
    #cdata_HLV = unpickle_contour_data('GW170814_HLV_m1_m2')

    g = plot_2d_chieff(['chi_p', 'chi_eff'],
                       pos_HL,
                       pos_prior,
                       cdata_HL,
                       xlim=xlim,
                       ylim=ylim,
                       labels=labels,
                       transform=ms2q,
                       yhigh=1.)

    # if annotate:
    #     x = np.linspace(xlim[0], xlim[1], 100)
    #     plt.fill_between(x, x, ylim[1]*np.ones_like(x), hatch='\\', color='none', edgecolor="k")
    #     g.ax_marg_x.patch.set_alpha(0.)
    #     g.ax_marg_y.patch.set_alpha(0.)

    # else:
    #    g.fig.get_axes()[0].legend([plt.Line2D([0,0], [1,0], color='k', linewidth=1.5),
    #                                plt.Line2D([0,0], [1,0], color="firebrick", linewidth=1.5)],
    #                               ['HL','HLV'], loc='upper right', frameon=False)
    g.fig.get_axes()[0].legend([
        plt.Line2D([0, 0], [1, 0], color='k', linewidth=1.5),
        plt.Line2D([0, 0], [1, 0], color='seagreen', linewidth=1)
    ], ['Posterior', 'Prior'],
                               loc='upper right',
                               frameon=False)

    g.ax_marg_x.axis('off')
    g.ax_marg_y.axis('off')

    # ax = g.ax_joint
    # #ax.xaxis.set_major_formatter(DegreeFormatter())
    # ax.set_yticks([15, 20., 25., 30,35])
    # ax.set_xticks([15, 20., 25., 30,35,40,45])
    # # ax.set_xticks([0., 30., 60., 90,120,150,180])

    ax = g.ax_joint
    ax.set_yticks([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1])
    ax.set_xticks([0, .25, .5, .75, 1])

    sns.despine(fig=g.fig, trim=True)

    # label_locs = {'GW150914':(43, 26),
    #               'LVT151012':(48, 8.5),
    #               'GW151226':(30, 3),
    #               'GW170104':(43, 18)
    # event_colors = {'GW150914': u'#c4c4c4',
    #                 'LVT151012': u'#c4c4c4',
    #                 'GW151226': u'#c4c4c4',
    #                 'GW170104': 'k'}

    # for event, loc in label_locs.iteritems():
    #     ax.text(loc[0], loc[1], event, color=event_colors[event], fontsize=13)

    # m1, m2 = np.meshgrid(np.linspace(20,50,100), np.linspace(5,30,100))
    # mchirp = ((m1*m2)**(3./5.))*((m1+m2)**(-1./5.))
    # mchirp[m1-m2<0] = np.nan
    # g.ax_joint.contour(m1,m2,mchirp,[21.2237407784], colors='black', linestyles='dashed',linewidth=0.5)

    g.fig.savefig('chieff.pdf', bbox_inches='tight', bbox_extra_artists=[])
    g.fig.savefig('chieff.png',
                  bbox_inches='tight',
                  bbox_extra_artists=[],
                  dpi=600)
