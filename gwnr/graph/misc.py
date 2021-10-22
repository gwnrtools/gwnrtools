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

from gwnr.utils.support import insert_min_max_into_array
from gwnr.nr.analysis.types import (Overlaps, SimulationErrors,
                                    EffectualnessAndBias)
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib as mp

mp.rc('text', usetex=True)
plt.rcParams.update({'text.usetex': True})


def make_filled_contour_plot(x,
                             y,
                             z,
                             ax=None,
                             n_pixels=50,
                             interp_func='Rbf',
                             contour_levels=[
                                 1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 60.0, 80.0,
                                 90.0, 95.0, 99.99
                             ],
                             cmap=None,
                             add_colorbar=True,
                             interp_func_args={},
                             interp_func_default_args={
                                 'Rbf': {
                                     'function': 'quintic',
                                     'smooth': 0.01
                                 },
                                 'griddata': {
                                     'method': 'cubic',
                                     'fill_value': 0
                                 },
                                 'SmoothBivariateSpline': {
                                     'kx': 5,
                                     'ky': 5,
                                     's': 0.00002
                                 }
                             }):
    '''
Makes filled contours for Z data on X-Y axes (where these 3 are the
first three arguments)

Input:
------

x, y, z : 1-D scalar arrays
ax : matplotlib.Figure.axes object on which to draw the contours.
     If not provided, make a default figure and add an axis.
n_pixels : number of pixels or grains along both X-Y axes. default: 50
interp_func : Interpolation method to use to convert the 1-D input data
              to 2-D gridded data that can be plotted as contours
interp_func_args : Arguments to be passed to the interpolation method
interp_func_default_args : Default arguments to be passed to the
                           interpolation method
contour_levels : Percentage levels of Z at which to draw contours.
                 Default: many contours from 1% to 99%.

Output:
-------
If an axis is passed in:
im, ax : (contour object, matplotlib.Figure.axes object)

If no axis is passed in:
im, ax, fig :  (contour object, matplotlib.Figure.axes object,
                matplotlib.Figure object)

    '''
    if interp_func not in interp_func_default_args:
        raise IOError(
            "Interpolation function {} not supported. Use one of {}.".format(
                interp_func, list(interp_func_default_args.keys())))

    interp_kwargs = {a: interp_func_args[a] for a in interp_func_args}
    for a in interp_func_default_args[interp_func]:
        if a not in interp_kwargs:
            interp_kwargs[a] = interp_func_default_args[interp_func][a]

    x1vals = np.linspace(min(x), max(x), n_pixels)
    x2vals = np.linspace(min(y), max(y), n_pixels)
    q, w = np.meshgrid(x1vals, x2vals)

    if interp_func == 'Rbf':
        from scipy.interpolate import Rbf
        z_int = Rbf(x, y, z, **interp_kwargs)
        r1 = z_int(q, w)

    elif interp_func == 'griddata':
        from scipy.interpolate import griddata
        points = np.column_stack(
            [np.array(x).flatten(),
             np.array(y).flatten()])
        r1 = griddata(points, z, (q, w), **interp_kwargs)

    elif interp_func == 'SmoothBivariateSpline':
        from scipy.interpolate import SmoothBivariateSpline
        z_int = SmoothBivariateSpline(x, y, z, **interp_kwargs)
        r1 = z_int.ev(q, w)

    else:
        raise RuntimeError("How did we even get here?")

    # Create axis (and figure) if not provided
    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # Create contour levels
    zlevels = [
        z.min() + lev * (z.max() - z.min()) / 100.0 for lev in contour_levels
    ]

    # Make contour plot
    im = ax.contourf(x1vals, x2vals, r1, zlevels, cmap=cmap)

    # Beautify
    ax.grid(True)
    if add_colorbar:
        cb = plt.colorbar(im, ax=ax)

    if fig == None:
        return im, ax
    else:
        return im, ax, fig


def make_2Dplot_errorbars(Xs,
                          Ys,
                          Xerrs=None,
                          Yerrs=None,
                          xlabel='',
                          ylabel='',
                          title='',
                          logy=True,
                          xmin=-0.9,
                          xmax=0.9,
                          ymin=None,
                          ymax=None,
                          addlines=True,
                          legendplacement='best',
                          labels=None,
                          fmts=['bs', 'ko', 'r^'],
                          savefig='plots/plot.png'):
    # {{{
    if len(Xs) != len(Ys):
        raise IOError("Length of lists to be plotted not the same")
    nrows, ncols = 1, 1  # np.shape(Xs)
    ncurves = len(Xs)
    print("No of rows = %d, columns = %d, curves = %d" %
          (nrows, ncols, ncurves))
    if labels == None:
        labels = range(ncurves)
        for idx in range(len(labels)):
            labels[idx] = str(labels[idx])
    #
    gmean = (5.**0.5 - 1) * 0.5
    fig = plt.figure(int(1e7 * np.random.random()),
                     figsize=(4 * ncols / gmean, 4 * nrows))
    nplot = 0
    ax = fig.add_subplot(nrows, ncols, nplot)
    #
    pcolor = ['b', 'r', 'k']
    for curveid in range(3):
        print("Adding curve {}".format(curveid))
        X, Y, Xerr, Yerr = Xs[curveid], Ys[curveid], None, None
        if Xerrs is not None:
            Xerr = Xerrs[curveid]
        if Yerrs is not None:
            Yerr = Yerrs[curveid]
        # Make one marker hollow
        if curveid != 1:
            ax.errorbar(X,
                        Y,
                        yerr=Yerr,
                        xerr=Xerr,
                        mfc=pcolor[curveid],
                        mec=pcolor[curveid],
                        color=pcolor[curveid],
                        ecolor=pcolor[curveid],
                        fmt=fmts[curveid],
                        elinewidth=2,
                        label=labels[curveid])
        else:
            ax.errorbar(X,
                        Y,
                        yerr=Yerr,
                        xerr=Xerr,
                        mfc='none',
                        mec=pcolor[curveid],
                        color=pcolor[curveid],
                        ecolor=pcolor[curveid],
                        fmt=fmts[curveid],
                        elinewidth=2,
                        label=labels[curveid])

        ax.hold(True)
        if addlines:
            ax.plot([-1, 1], [curveid + 1, curveid + 1],
                    pcolor[curveid] + '--',
                    lw=1.6)
            ax.hold(True)
    if logy:
        ax.set_yscale('log')
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xmin, xmax)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.legend(ncol=ncurves, loc=legendplacement)
    fig.tight_layout()  # rect=(0,0,0.93,1))
    fig.savefig(savefig, dpi=600)
    return
    # }}}


def make_scatter_plot3D_mult(X1,
                             Y1,
                             Z1,
                             C1,
                             X2,
                             Y2,
                             Z2,
                             C2,
                             X3,
                             Y3,
                             Z3,
                             C3,
                             elevation=30,
                             azimuthal=30,
                             alpha=0.8,
                             xlabel='',
                             ylabel='',
                             zlabel='',
                             clabel='',
                             title='',
                             bounds=None,
                             equal_mass=1,
                             label=None,
                             logC=True,
                             cmin=0.8,
                             cmax=1.,
                             savefig='plots/plot.png'):
    # {{{
    S = 30  # 50*Z
    if logC:
        print("Using logscale on Z", file=sys.stderr)
        C1 = np.log10(np.abs(C1))
        C2 = np.log10(np.abs(C2))
        C3 = np.log10(np.abs(C3))
        if 'FF' in clabel:
            cmin = -2.3
        else:
            cmin = np.round(min(min(C1), min(C2), min(C3)) * 100) / 100.
        cmax = np.round(max(max(C1), max(C2), max(C3)) * 100) / 100.
        clabel = clabel + ' (Log)'
        # Insert logic here to ensure cmin and cmax are the limits on bounds
    else:
        print("NOT using logscale on Z", file=sys.stderr)
        if 'FF' in clabel:
            cmin = 10**-2.3
        else:
            cmin = np.round(min(min(C1), min(C2), min(C3)) * 100) / 100.
        cmax = np.round(max(max(C1), max(C2), max(C3)) * 100) / 100.
        # Insert logic here to ensure cmin and cmax are the limits on bounds
    #
    if bounds != None and (type(bounds) == np.ndarray or type(bounds) == list):
        bounds = insert_min_max_into_array(bounds, cmin, cmax)
    #
    if bounds is None and logC:
        if 'FF' in clabel or 'mathcal{M}' in clabel:
            bounds = np.log10([0.0001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 1])
        elif '\mathcal{M}_c' in clabel:
            print("CHIRP MASS PLOT")
            bounds = np.log10([0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2])
        elif '\Delta' in clabel:
            bounds = np.log10([0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.])
        else:
            bounds = np.linspace(cmin, cmax, 10)
            #bounds = np.append( bounds - 0.1, 0 )
    elif bounds is None:
        #raise IOError("Non-log default colorbar bounds not supported")
        print(" >> >> >> Non-log default colorbar bounds not supported")
    #xlabel = xlabel.replace('_', '-')
    #ylabel = ylabel.replace('_', '-')
    #zlabel = zlabel.replace('_', '-')
    #clabel = clabel.replace('_', '-')
    title = title.replace('_', '-')
    fig = plt.figure(int(1e7 * np.random.random()), figsize=(12, 4))
    #
    #cmap = plt.cm.Spectral
    cmap = plt.cm.RdYlGn  # bwr
    #cmap = plt.get_cmap('jet', 20)
    #cmaplist = [cmap(i) for i in range(cmap.N)]
    #cmap = cmap.from_list('Custom map', cmaplist, cmap.N)
    # cmap.set_under('gray')
    print("bounds = ", bounds)
    if type(bounds) == np.ndarray or type(bounds) == list:
        norm = mp.colors.BoundaryNorm(bounds, cmap.N)
    else:
        tmp_bounds = np.linspace(cmin, cmax, 10)
        norm = mp.colors.BoundaryNorm(tmp_bounds, cmap.N)
    # 1
    pltnum = 1
    ax = fig.add_subplot(131, projection='3d')
    ax.view_init(elev=elevation, azim=azimuthal)
    scat = ax.scatter(X1,
                      Y1,
                      Z1,
                      c=C1,
                      s=S,
                      lw=0,
                      alpha=alpha,
                      vmin=cmin,
                      vmax=cmax,
                      cmap=cmap,
                      norm=norm)
    # Add points in the bottom plane marking spins
    ax.plot(X1, Y1, (min(Z1) - 5.) * np.ones(len(Z1)), 'kx', markersize=4)
    # Add mirrored points if q == 1
    if equal_mass == pltnum:
        scat = ax.scatter(Y1,
                          X1,
                          Z1,
                          c=C1,
                          s=S,
                          lw=0,
                          alpha=alpha,
                          vmin=cmin,
                          vmax=cmax,
                          cmap=cmap,
                          norm=norm)
        ax.plot(Y1, X1, (min(Z1) - 5.) * np.ones(len(Z1)), 'kx', markersize=4)
    #
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(zlabel, rotation=90)
    ax.set_title('$q = 1$', verticalalignment='bottom')
    ax.grid()
    # 2
    pltnum = 2
    ax = fig.add_subplot(132, projection='3d')
    ax.view_init(elev=elevation, azim=azimuthal)
    scat = ax.scatter(X2,
                      Y2,
                      Z2,
                      c=C2,
                      s=S,
                      lw=0,
                      alpha=alpha,
                      vmin=cmin,
                      vmax=cmax,
                      cmap=cmap,
                      norm=norm)
    if equal_mass == pltnum:
        scat = ax.scatter(Y1,
                          X1,
                          Z1,
                          c=C1,
                          s=S,
                          lw=0,
                          alpha=alpha,
                          vmin=cmin,
                          vmax=cmax,
                          cmap=cmap,
                          norm=norm)
    # Add points in the bottom plane marking spins
    ax.plot(X2, Y2, (min(Z2) - 5.) * np.ones(len(Z2)), 'kx', markersize=4)
    #
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_title(title + '\n$q = 2$')
    ax.grid()
    # 3
    pltnum = 3
    ax = fig.add_subplot(133, projection='3d')
    ax.view_init(elev=elevation, azim=azimuthal)
    scat = ax.scatter(X3,
                      Y3,
                      Z3,
                      c=C3,
                      s=S,
                      lw=0,
                      alpha=alpha,
                      vmin=cmin,
                      vmax=cmax,
                      cmap=cmap,
                      norm=norm)
    if equal_mass == pltnum:
        scat = ax.scatter(Y1,
                          X1,
                          Z1,
                          c=C1,
                          s=S,
                          lw=0,
                          alpha=alpha,
                          vmin=cmin,
                          vmax=cmax,
                          cmap=cmap,
                          norm=norm)
    # Add points in the bottom plane marking spins
    ax.plot(X3, Y3, (min(Z3) - 5.) * np.ones(len(Z3)), 'kx', markersize=4)
    #
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.grid()
    ax.set_title('$q = 3$', verticalalignment='bottom')
    # ax.set_zlabel(zlabel)
    # ax.set_title(title)
    #
    #ax2 = fig.add_axes([0.95, 0.1, 0.01, 0.7])
    ax2 = fig.add_axes([0.2, 0.05, 0.6, 0.02])
    # Make the colorbar
    if type(bounds) == np.ndarray or type(bounds) == list:
        cb = mp.colorbar.ColorbarBase(ax2,
                                      cmap=cmap,
                                      norm=norm,
                                      spacing='uniform',
                                      format='%.2f',
                                      orientation=u'horizontal',
                                      ticks=bounds,
                                      boundaries=bounds)
    else:
        # How does this colorbar know what colors to span??
        cb = mp.colorbar.ColorbarBase(ax2,
                                      cmap=cmap,
                                      norm=norm,
                                      spacing='uniform',
                                      format='%.2f',
                                      orientation=u'horizontal',
                                      ticks=tmp_bounds)
    # Add tick labels
    if logC and (type(bounds) == np.ndarray or type(bounds) == list):
        cb.set_ticklabels(np.round(10**bounds, decimals=3))
    elif type(bounds) == np.ndarray or type(bounds) == list:
        cb.set_ticklabels(np.round(bounds, decimals=3))
    # cb = fig.colorbar(scat, shrink=0.5, aspect=30, spacing='proportional',\
    #  ticks=[0,-0.5,-1,-1.3,-1.6,-1.9,-2.1,-2.4,-2.7,-3])
    #ax2.set_title(clabel, loc='left')
    cb.set_label(clabel, labelpad=-0.3, y=0)
    #if max(C) < cmax and min(C) > cmin: cb.set_clim([cmin,cmax])
    fig.tight_layout()
    print(savefig)
    if '.png' in savefig:
        savefig = savefig.split('.png')[0] + '_q123.png'
    elif '.pdf' in savefig:
        savefig = savefig.split('.pdf')[0] + '_q123.pdf'
    print(savefig)
    fig.savefig(savefig)
    return
    # }}}


# WRITE A NEW FUNCTION THAT CAN PLOT 2-3 (rows) X 3 (COLUMN) plots:


def make_scatter_plot3D_multrow(Xs,
                                Ys,
                                Zs,
                                Cs,
                                elevation=30,
                                azimuthal=30,
                                alpha=0.8,
                                xlabel='',
                                ylabel='',
                                zlabel='',
                                clabel='',
                                title='',
                                bounds=None,
                                equal_mass=1,
                                colormin=None,
                                colormax=None,
                                label=None,
                                logC=True,
                                cmin=0.8,
                                cmax=1.,
                                savefig='plots/plot.png'):
    # {{{
    if len(Xs) != len(Ys) or len(Xs) != len(Zs) or len(Xs) != len(Cs):
        raise IOError("Length of lists to be plotted not the same")
    # Known failure mode: when all arrays in Xrows are of the same length
    nrows, ncols = np.shape(Xs)
    print("No of rows = %d, columns = %d" % (nrows, ncols))
    #
    gmean = (5.**0.5 - 1) * 0.5
    S = 30  # 50*Z
    if logC:
        bounds = np.log10(bounds)
        print("Using logscale on Z", file=sys.stderr)
        for ridx, C in enumerate(Cs):
            for cidx, R in enumerate(C):
                Cs[ridx][cidx] = np.log10(R)
        if 'FF' in clabel:
            cmin = -2.3
        else:
            Cs = np.array(Cs)
            print("Shape of C = ", np.shape(Cs), "dtype of C = ", type(Cs))
            cmin = np.inf
            for tmpr in Cs:
                for tmpc in tmpr:
                    print(np.shape(tmpc), np.shape(tmpr))
                    if cmin > np.min(tmpc):
                        cmin = np.min(tmpc)
            print("Min = ", cmin)
            cmin = np.round(cmin, decimals=3)
        #clabel = clabel + ' (Log)'
    else:
        print("NOT using logscale on Z", file=sys.stderr)
        if 'FF' in clabel:
            cmin = 10**-2.3
        else:
            Cs = np.array(Cs)
            print("Shape of C = ", np.shape(Cs), "dtype of C = ", type(Cs))
            cmin = np.inf
            for tmpr in Cs:
                for tmpc in tmpr:
                    print(np.shape(tmpc), np.shape(tmpr))
                    if cmin > np.min(tmpc):
                        cmin = np.min(tmpc)
            print("Min = ", cmin)
            cmin = np.round(cmin, decimals=3)
    cmax = -np.inf
    for tmpr in Cs:
        for tmpc in tmpr:
            if cmax < np.max(tmpc):
                cmax = np.max(tmpc)
    cmax = np.round(cmax, decimals=2)
    # After having computed cmin, cmax, check for user inputs
    if colormin is not None:
        cmin = colormin
    if colormax is not None:
        cmax = colormax
    #
    if bounds != None and (type(bounds) == np.ndarray or type(bounds) == list):
        print("bounds before insert  : ", bounds)
        bounds = insert_min_max_into_array(bounds, cmin, cmax)
        print("bounds after insert  : ", bounds)
    #
    # Insert default values of bounds
    if bounds is None and logC:
        if 'FF' in clabel or 'mathcal{M}' in clabel:
            bounds = np.log10([0.0001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 1])
        elif '\mathcal{M}_c' in clabel:
            print("CHIRP MASS PLOT")
            bounds = np.log10([0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2])
        elif '\Delta' in clabel:
            bounds = np.log10([0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.])
        else:
            bounds = np.linspace(cmin, cmax, 10)
            #bounds = np.append( bounds - 0.1, 0 )
    elif bounds is None:
        #raise IOError("Non-log default colorbar bounds not supported")
        print(">> >> Non-log default colorbar bounds not supported")
    #xlabel = xlabel.replace('_', '-')
    #ylabel = ylabel.replace('_', '-')
    #zlabel = zlabel.replace('_', '-')
    #clabel = clabel.replace('_', '-')
    for tid, t in enumerate(title):
        t[tid].replace('_', '-')
    fig = plt.figure(int(1e7 * np.random.random()),
                     figsize=(4 * ncols, 4 * nrows))
    # cmap = plt.cm.RdYlGn_r#gist_heat_r##winter#gnuplot#PiYG_r#RdBu_r#jet#rainbow#RdBu_r#Spectral_r
    cmap = plt.cm.jet  # seismic#RdBu_r#RdYlGn_r#PiYG_r#jet#
    #cmap = plt.get_cmap('jet', 20)
    #cmaplist = [cmap(i) for i in range(cmap.N)]
    #cmap = cmap.from_list('Custom map', cmaplist, cmap.N)
    # cmap.set_under('gray')
    print("bounds = ", bounds)
    if type(bounds) == np.ndarray or type(bounds) == list:
        norm = mp.colors.BoundaryNorm(bounds, cmap.N)
    else:
        tmp_bounds = np.linspace(cmin, cmax, 20)
        norm = mp.colors.BoundaryNorm(tmp_bounds, cmap.N)
    #
    # Begin plotting loop
    nplot = 0
    for rowid in range(nrows):
        for colid in range(ncols):
            nplot += 1
            ax = fig.add_subplot(nrows, ncols, nplot, projection='3d')
            ax.view_init(elev=elevation, azim=azimuthal)
            X, Y, Z, C = Xs[rowid][colid], Ys[rowid][colid], Zs[rowid][
                colid], Cs[rowid][colid]
            # Add the actual color plot
            scat = ax.scatter(X,
                              Y,
                              Z,
                              c=C,
                              s=S,
                              lw=0,
                              alpha=alpha,
                              vmin=cmin,
                              vmax=cmax,
                              cmap=cmap,
                              norm=norm)
            # Add points in the bottom plane marking spins
            ax.plot(X,
                    Y, (np.min(Z) - 2.) * np.ones(len(Z)),
                    'kx',
                    markersize=4)
            if equal_mass == colid + 1:
                scat = ax.scatter(Y,
                                  X,
                                  Z,
                                  c=C,
                                  s=S,
                                  lw=0,
                                  alpha=alpha,
                                  vmin=cmin,
                                  vmax=cmax,
                                  cmap=cmap,
                                  norm=norm)
                ax.plot(Y,
                        X, (np.min(Z) - 2.) * np.ones(len(Z)),
                        'kx',
                        markersize=4)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.zaxis.set_rotate_label(False)
            ax.set_zlabel(zlabel, rotation=108)
            ax.set_zlim(zmin=40)
            ax.locator_params(axis='z', nbins=5)
            ax.locator_params(axis='z', prune='upper')
            if colid == ncols / 2:
                ax.set_title(title[rowid] + '\n $q=%d$' % (colid + 1),
                             verticalalignment='bottom')
            else:
                ax.set_title('$q=%d$' % (colid + 1),
                             verticalalignment='bottom')
            ax.grid()
    #
    # ax2 = fig.add_axes([0.9, 0.3, 0.01, 0.35]) # Short colorbar
    ax2 = fig.add_axes([0.9, 0.1, 0.01, 0.7])

    #ax2 = fig.add_axes([0.2, 0.05, 0.6, 0.02])
    # Make the colorbar
    if type(bounds) == np.ndarray or type(bounds) == list:
        cb = mp.colorbar.ColorbarBase(ax2,
                                      cmap=cmap,
                                      norm=norm,
                                      spacing='uniform',
                                      format='%.2f',
                                      orientation=u'vertical',
                                      ticks=bounds,
                                      boundaries=bounds)
    else:
        # How does this colorbar know what colors to span??
        cb = mp.colorbar.ColorbarBase(ax2,
                                      cmap=cmap,
                                      norm=norm,
                                      spacing='uniform',
                                      format='%.3f',
                                      orientation=u'vertical')  # ,\
        #      #ticks=tmp_bounds)
        # cb = mp.colorbar.Colorbar(ax2,\
        #      spacing='uniform', format='%.2f', orientation=u'vertical')#,\
    # Add tick labels
    if logC and (type(bounds) == np.ndarray or type(bounds) == list):
        cb.set_ticklabels(np.round(10**bounds, decimals=4))
    elif type(bounds) == np.ndarray or type(bounds) == list:
        cb.set_ticklabels(np.round(bounds, decimals=4))
    # cb = fig.colorbar(scat, shrink=0.5, aspect=30, spacing='proportional',\
    #  ticks=[0,-0.5,-1,-1.3,-1.6,-1.9,-2.1,-2.4,-2.7,-3])
    #ax2.set_title(clabel, loc='left')
    cb.set_label(clabel,
                 verticalalignment='top',
                 horizontalalignment='center',
                 size=22)
    # ,labelpad=-0.3,y=1.1,x=-0.5)
    #if max(C) < cmax and min(C) > cmin: cb.set_clim([cmin,cmax])
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    if '.png' in savefig:
        savefig = savefig.split('.png')[0] + '_q123.png'
    elif '.pdf' in savefig:
        savefig = savefig.split('.pdf')[0] + '_q123.pdf'
    fig.savefig(savefig)
    return
    # }}}


# WRITE A NEW FUNCTION THAT CAN PLOT 2-3 (rows) X 3 (COLUMN) plots


def make_contour_plot_multrow(Xs,
                              Ys,
                              Cs,
                              elevation=30,
                              azimuthal=30,
                              alpha=0.8,
                              xlabel='',
                              ylabel='',
                              zlabel='',
                              clabel='',
                              title='',
                              titles=[],
                              bounds=None,
                              colors=[],
                              equal_mass=1,
                              colorbartype='simple',
                              label=None,
                              logC=True,
                              cmin=0.8,
                              cmax=1.,
                              savefig='plots/plot.png'):
    # {{{
    if not len(Xs) == len(Ys) == len(Cs):
        raise IOError("Length of lists to be plotted not the same")
    # Known failure mode: when all arrays in Xrows are of the same length
    nrows, ncols = np.shape(Xs)
    print("No of rows = %d, columns = %d" % (nrows, ncols))
    #
    gmean = (5.**0.5 - 1) * 0.5
    if logC:
        bounds = np.log10(bounds)
        print("Using logscale on Z", file=sys.stderr)
        for ridx, C in enumerate(Cs):
            for cidx, R in enumerate(C):
                Cs[ridx][cidx] = np.log10(R)
        if 'FF' in clabel:
            cmin = -2.3
        else:
            Cs = np.array(Cs)
            print("Shape of C = ", np.shape(Cs), "dtype of C = ", type(Cs))
            cmin = np.inf
            for tmpr in Cs:
                for tmpc in tmpr:
                    print(np.shape(tmpc), np.shape(tmpr))
                    if cmin > np.min(tmpc):
                        cmin = np.min(tmpc)
            print("Min = ", cmin)
            cmin = np.round(cmin, decimals=3)
        #clabel = clabel + ' (Log)'
    else:
        print("NOT using logscale on Z", file=sys.stderr)
        if 'FF' in clabel:
            cmin = 10**-2.3
        else:
            Cs = np.array(Cs)
            print("Shape of C = ", np.shape(Cs), "dtype of C = ", type(Cs))
            cmin = np.inf
            for tmpr in Cs:
                for tmpc in tmpr:
                    cmin = min(cmin, np.min(tmpc))
            print("Min = ", cmin)
            cmin = np.round(cmin, decimals=3)
    cmax = -np.inf
    for tmpr in Cs:
        for tmpc in tmpr:
            cmax = max(cmax, np.max(tmpc))
    cmax = np.round(cmax, decimals=3)
    #
    if bounds != None and (type(bounds) == np.ndarray or type(bounds) == list):
        print("bounds before insert  : ", bounds)
        bounds = insert_min_max_into_array(bounds, cmin, cmax)
        print("bounds after insert  : ", bounds)
    #
    # Insert default values of bounds
    if bounds is None and logC:
        if 'FF' in clabel or 'mathcal{M}' in clabel:
            bounds = np.log10([0.0001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 1])
        elif '\mathcal{M}_c' in clabel:
            print("CHIRP MASS PLOT")
            bounds = np.log10([0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2])
        elif '\Delta' in clabel:
            bounds = np.log10([0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.])
        else:
            bounds = np.linspace(cmin, cmax, 10)
            #bounds = np.append( bounds - 0.1, 0 )
    elif bounds is None:
        raise IOError("Non-log default colorbar bounds not supported")
    for tid, t in enumerate(title):
        t[tid].replace('_', '-')
    fig = plt.figure(int(1e7 * np.random.random()),
                     figsize=(4 * ncols, 4 * nrows))
    # cmap = plt.cm.RdYlGn_r#gist_heat_r##winter#gnuplot#PiYG_r#RdBu_r#jet#rainbow#RdBu_r#Spectral_r
    # cmap = plt.cm.PiYG_r#rainbow#RdBu_r#RdYlGn_r
    #cmap = plt.get_cmap('jet_r', 20)
    cmap = plt.cm.OrRd
    #cmaplist = [cmap(i) for i in range(cmap.N)]
    #cmap = cmap.from_list('Custom map', cmaplist, cmap.N)
    # cmap.set_under('gray')
    print("bounds = ", bounds)
    if type(bounds) == np.ndarray or type(bounds) == list:
        norm = mp.colors.BoundaryNorm(bounds, cmap.N)
    else:
        tmp_bounds = np.linspace(cmin, cmax, 10)
        norm = mp.colors.BoundaryNorm(tmp_bounds, cmap.N)
    #
    # Begin plotting loop
    nplot = 0
    allaxes = []
    for rowid in range(nrows):
        for colid in range(ncols):
            nplot += 1
            ax = fig.add_subplot(nrows, ncols, nplot)
            allaxes.append(ax)
            #
            X, Y, C = Xs[rowid][colid], Ys[rowid][colid], Cs[rowid][colid]
            print(np.shape(X), np.shape(Y), np.shape(C))
            # Add points in the bottom plane marking spins
            if equal_mass == colid + 1:
                tmpX, tmpY = X, Y
                X = np.append(X, tmpY)
                Y = np.append(Y, tmpX)
                C = np.append(C, C)
            #
            Xrange = np.linspace(min(X), max(X), 2000)
            Yrange = np.linspace(min(Y), max(Y), 2000)
            #Xrange = np.linspace( -1, 1, 100 )
            #Yrange = np.linspace( -1, 1, 100)
            Xmap, Ymap = np.meshgrid(Xrange, Yrange)
            print(np.shape(X), np.shape(Y), np.shape(C))
            colormap = plt.mlab.griddata(X, Y, C, Xmap, Ymap, interp='linear')
            #
            import scipy.interpolate as si
            rbfi = si.SmoothBivariateSpline(X, Y, C, kx=4, ky=4)
            #colormap = rbfi(Xrange, Yrange)
            #
            # New interpolation scheme
            #
            #xyzData = np.append( np.append( [X], [Y], axis=0 ), [C], axis=0 )
            #xyzData = scipy.ndimage.zoom(xyzData, 3)
            #Xmap = xyzData[:,0]
            #Ymap = xyzData[:,1]
            #colormap = xyzData[:,2]
            print("Shape pof Xmap, Ymap, colormap = ", np.shape(Xmap),
                  np.shape(Ymap), np.shape(colormap))
            #Xmap = scipy.ndimage.zoom(Xmap, 3)
            #Ymap = scipy.ndimage.zoom(Ymap, 3)
            #colormap = scipy.ndimage.zoom(colormap, 3)
            if len(colors) == (len(bounds) - 1):
                CS = ax.contourf(Xmap, Ymap, colormap,
                                 levels=bounds,
                                 colors=colors,
                                 alpha=0.75,\
                                 # cmap=plt.cm.spectral,\
                                 linestyles='dashed')
                '''CS = ax.tricontourf(X,Y,C,\
                    levels=bounds,\
                    colors=colors,\
                    alpha=0.9)'''
                '''CS1 = ax.scatter(X, Y, c='k', s=5)#, \
                    #reduce_C_function=np.max)'''
            else:
                ax.contourf(Xmap,
                            Ymap,
                            colormap,
                            bounds,
                            cmap=cmap,
                            linestyles='dashed')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            # ax.zaxis.set_rotate_label(False)
            #ax.set_zlabel(zlabel, rotation=90)
            print("Len(titles) = %d, NCOLS = %d" % (len(titles), ncols))
            if len(titles) == ncols:
                ax.set_title(titles[colid], verticalalignment='bottom')
            elif colid == ncols / 2:
                ax.set_title(title[rowid] + '\n $q=%d$' % (colid + 1),
                             verticalalignment='bottom')
            else:
                ax.set_title('$q=%d$' % (colid + 1),
                             verticalalignment='bottom')
            ax.grid()
    #
    if colorbartype == 'simple':
        ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.7])
        cb = plt.colorbar(CS, cax=ax2, orientation=u'vertical', format='%.3f')
        cb.set_label(clabel)
    else:
        ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.7])
        #ax2 = fig.add_axes([0.2, 0.05, 0.6, 0.02])
        # Make the colorbar
        if type(bounds) == np.ndarray or type(bounds) == list:
            cb = mp.colorbar.ColorbarBase(ax2,
                                          cmap=cmap,
                                          norm=norm,
                                          spacing='uniform',
                                          format='%.2f',
                                          orientation=u'vertical',
                                          ticks=bounds,
                                          boundaries=bounds)
        else:
            # How does this colorbar know what colors to span??
            cb = mp.colorbar.ColorbarBase(ax2,
                                          cmap=cmap,
                                          norm=norm,
                                          spacing='uniform',
                                          format='%.2f',
                                          orientation=u'vertical',
                                          ticks=tmp_bounds)
        # Add tick labels
        if logC and (type(bounds) == np.ndarray or type(bounds) == list):
            cb.set_ticklabels(np.round(10**bounds, decimals=4))
        elif type(bounds) == np.ndarray or type(bounds) == list:
            cb.set_ticklabels(np.round(bounds, decimals=4))
        # cb = fig.colorbar(scat, shrink=0.5, aspect=30, spacing='proportional',\
        #  ticks=[0,-0.5,-1,-1.3,-1.6,-1.9,-2.1,-2.4,-2.7,-3])
        #ax2.set_title(clabel, loc='left')
        # ,labelpad=-0.3,y=1.1,x=-0.5)
        cb.set_label(clabel,
                     verticalalignment='top',
                     horizontalalignment='center')
        #if max(C) < cmax and min(C) > cmin: cb.set_clim([cmin,cmax])
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    if '.png' in savefig:
        savefig = savefig.split('.png')[0] + '_q123.png'
    elif '.pdf' in savefig:
        savefig = savefig.split('.pdf')[0] + '_q123.pdf'
    fig.savefig(savefig)
    return
    # }}}


def make_scatter_plot(X,
                      Y,
                      Z,
                      xlabel='',
                      ylabel='',
                      zlabel='',
                      title='',
                      logz=True,
                      cmin=0.8,
                      cmax=1.,
                      savefig='plots/plot.png'):
    # {{{
    S = 30  # 50*Z
    if logz:
        print("Using logscale on Z", file=sys.stderr)
        Z = np.log10(1. - Z)
        cmin, cmax = -2.75, max(Z)
        zlabel = 'Log[Mismatch]'
    xlabel = xlabel.replace('_', '-')
    ylabel = ylabel.replace('_', '-')
    zlabel = zlabel.replace('_', '-')
    title = title.replace('_', '-')
    plt.figure(int(1e7 * np.random.random()))
    plt.scatter(X, Y, c=Z, s=S, lw=0, alpha=0.8)
    cb = plt.colorbar()
    cb.set_label(zlabel)
    if max(Z) < cmax and min(Z) > cmin:
        cb.set_clim([cmin, cmax])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.savefig(savefig)
    return
    # }}}


def make_scatter_plot3D(X,
                        Y,
                        Z,
                        C,
                        elevation=30,
                        azimuthal=30,
                        alpha=0.8,
                        xlabel='',
                        ylabel='',
                        zlabel='',
                        clabel='',
                        title='',
                        label=None,
                        logC=True,
                        cmin=0.8,
                        cmax=1.,
                        savefig='plots/plot.png'):
    # {{{
    S = 30  # 50*Z
    if logC:
        print("Using logscale on Z", file=sys.stderr)
        C = np.log10(1. - C)
        cmin, cmax = -2.75, np.round(max(C) * 100) / 100.
        clabel = 'Log[Mismatch]'
    xlabel = xlabel.replace('_', '-')
    ylabel = ylabel.replace('_', '-')
    zlabel = zlabel.replace('_', '-')
    clabel = clabel.replace('_', '-')
    title = title.replace('_', '-')
    fig = plt.figure(int(1e7 * np.random.random()))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elevation, azim=azimuthal)
    #
    cmap = plt.cm.jet
    #cmap = plt.get_cmap('jet', 20)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom map', cmaplist, cmap.N)
    cmap.set_under('gray')
    bounds = np.linspace(cmin, cmax, 10)
    bounds = np.append(bounds - 0.1, 0)
    norm = mp.colors.BoundaryNorm(bounds, cmap.N)
    #
    scat = ax.scatter(X,
                      Y,
                      Z,
                      c=C,
                      s=S,
                      lw=0,
                      alpha=alpha,
                      vmin=cmin,
                      vmax=cmax,
                      cmap=cmap,
                      norm=norm)
    ax2 = fig.add_axes([0.9, 0.1, 0.01, 0.7])
    cb = mp.colorbar.ColorbarBase(ax2,
                                  cmap=cmap,
                                  norm=norm,
                                  spacing='uniform',
                                  format='%.1f',
                                  ticks=bounds,
                                  boundaries=bounds)
    # cb = fig.colorbar(scat, shrink=0.5, aspect=30, spacing='proportional',\
    #  ticks=[0,-0.5,-1,-1.3,-1.6,-1.9,-2.1,-2.4,-2.7,-3])
    ax2.set_label(clabel)
    #if max(C) < cmax and min(C) > cmin: cb.set_clim([cmin,cmax])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.grid()
    fig.tight_layout()
    print(savefig)
    if label is not None:
        savefig = savefig.strip('png')[:-1] + ('_%s.png' % label)
    fig.savefig(savefig)
    return
    # }}}


# This function has not been tested and in all likelihood it does not work


def make_contourf_mult(X1,
                       Y1,
                       C1,
                       X2,
                       Y2,
                       C2,
                       X3,
                       Y3,
                       C3,
                       elevation=30,
                       azimuthal=30,
                       alpha=0.8,
                       xlabel='',
                       ylabel='',
                       zlabel='',
                       clabel='',
                       title='',
                       label=None,
                       logC=True,
                       cmin=0.8,
                       cmax=1.,
                       savefig='plots/plot.png'):
    # {{{
    S = 30  # 50*Z
    if logC:
        print("Using logscale on Z", file=sys.stderr)
        C1 = np.log10(1. - C1)
        C2 = np.log10(1. - C2)
        C3 = np.log10(1. - C3)
        cmin, cmax = -2.75, np.round(
            max(max(C1), max(C2), max(C3)) * 100) / 100.
        clabel = 'Log[1-FF]'
    xlabel = xlabel.replace('_', '-')
    ylabel = ylabel.replace('_', '-')
    zlabel = zlabel.replace('_', '-')
    clabel = clabel.replace('_', '-')
    title = title.replace('_', '-')
    fig = plt.figure(int(1e7 * np.random.random()), figsize=(12, 4))
    #
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom map', cmaplist, cmap.N)
    cmap.set_under('gray')
    #bounds = np.linspace(cmin, cmax, 10)
    #bounds = np.append( bounds - 0.1, 0 )
    #norm = mp.colors.BoundaryNorm(bounds, cmap.N)
    # 1
    Xrange = np.linspace(min(X1), max(X1), 400)
    Yrange = np.linspace(min(Y1), max(Y1), 400)
    Xmap, Ymap = np.meshgrid(Xrange, Yrange)
    etamap, incmap = Xmap, Ymap
    colormap = plt.mlab.griddata(X1, Y1, C1, Xmap, Ymap)
    ax = fig.add_subplot(131)
    #pylab.contour(etamap, incmap, colormap,  [0.947, .965], colors='black', linestyles='dashed' )
    ax.contourf(etamap,
                incmap,
                colormap, [.92, .93, .94, .947, .96, .965, .97, .98, .99, 1.],
                cmap=cmap,
                linestyles='dashed')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('q = 1')
    ax.grid()
    # 2
    Xrange = np.linspace(min(X2), max(X2), 400)
    Yrange = np.linspace(min(Y2), max(Y2), 400)
    Xmap, Ymap = np.meshgrid(Xrange, Yrange)
    colormap = plt.mlab.griddata(X2, Y2, C2, Xmap, Ymap)
    ax = fig.add_subplot(132)
    ax.contourf(etamap,
                incmap,
                colormap, [.92, .93, .94, .947, .96, .965, .97, .98, .99, 1.],
                cmap=cmap,
                linestyles='dashed')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title + '\n q = 2')
    ax.grid()
    # 3
    Xrange = np.linspace(min(X3), max(X3), 400)
    Yrange = np.linspace(min(Y3), max(Y3), 400)
    Xmap, Ymap = np.meshgrid(Xrange, Yrange)
    colormap = plt.mlab.griddata(X3, Y3, C3, Xmap, Ymap)
    ax = fig.add_subplot(133)
    ax.contourf(etamap,
                incmap,
                colormap, [.92, .93, .94, .947, .96, .965, .97, .98, .99, 1.],
                cmap=cmap,
                linestyles='dashed')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.set_title('q = 3')
    #ax2 = fig.add_axes([0.95, 0.1, 0.01, 0.7])
    # cb = mp.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, \
    #      spacing='uniform', format='%.1f', ticks=bounds, boundaries=bounds)
    # cb = fig.colorbar(scat, shrink=0.5, aspect=30, spacing='proportional',\
    #  ticks=[0,-0.5,-1,-1.3,-1.6,-1.9,-2.1,-2.4,-2.7,-3])
    # ax2.set_title(clabel)
    cb = plt.colorbar(ax, cmap=cmap, format='%.2f')
    cb.set_title(clabel)
    #if max(C) < cmax and min(C) > cmin: cb.set_clim([cmin,cmax])
    fig.tight_layout()
    print(savefig)
    savefig = savefig.strip('png')[:-1] + '_q123.png'
    fig.savefig(savefig)
    return
    # }}}


def make_parameters_plot(X,
                         Y,
                         Z,
                         elevation=30,
                         azimuthal=30,
                         xlabel='',
                         ylabel='',
                         zlabel='',
                         title='',
                         savefig='plots/plot.png'):
    # {{{
    S = 30  # 50*Z
    if logC:
        print("Using logscale on Z", file=sys.stderr)
        C = np.log10(1. - C)
        cmin, cmax = -2.75, np.round(max(C) * 100) / 100.
        clabel = 'Log[Mismatch]'
    xlabel = xlabel.replace('_', '-')
    ylabel = ylabel.replace('_', '-')
    zlabel = zlabel.replace('_', '-')
    clabel = clabel.replace('_', '-')
    title = title.replace('_', '-')
    fig = plt.figure(int(1e7 * np.random.random()))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elevation, azim=azimuthal)
    #
    cmap = plt.cm.jet
    #cmap = plt.get_cmap('jet', 20)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom map', cmaplist, cmap.N)
    cmap.set_under('gray')
    bounds = np.linspace(cmin, cmax, 10)
    bounds = np.append(bounds - 0.1, 0)
    norm = mp.colors.BoundaryNorm(bounds, cmap.N)
    #
    scat = ax.scatter(X,
                      Y,
                      Z,
                      c=C,
                      s=S,
                      lw=0,
                      alpha=alpha,
                      vmin=cmin,
                      vmax=cmax,
                      cmap=cmap,
                      norm=norm)
    ax2 = fig.add_axes([0.9, 0.1, 0.01, 0.7])
    cb = mp.colorbar.ColorbarBase(ax2,
                                  cmap=cmap,
                                  norm=norm,
                                  spacing='uniform',
                                  format='%.1f',
                                  ticks=bounds,
                                  boundaries=bounds)
    # cb = fig.colorbar(scat, shrink=0.5, aspect=30, spacing='proportional',\
    #  ticks=[0,-0.5,-1,-1.3,-1.6,-1.9,-2.1,-2.4,-2.7,-3])
    ax2.set_label(clabel)
    #if max(C) < cmax and min(C) > cmin: cb.set_clim([cmin,cmax])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.grid()
    fig.tight_layout()
    print(savefig)
    if label is not None:
        savefig = savefig.strip('png')[:-1] + ('_%s.png' % label)
    fig.savefig(savefig)
    return
    # }}}
