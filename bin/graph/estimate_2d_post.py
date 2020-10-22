#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate 2-D bounded-kde estimates of the combined posterior
"""

from __future__ import division

import os

import numpy as np
import cPickle as pickle

from common import contour_dir, events, sample_files
from bounded_2d_kde import Bounded_2d_kde


def estimate_2d_post(params,
                     post,
                     data2=None,
                     xlow=None,
                     xhigh=None,
                     ylow=None,
                     yhigh=None,
                     transform=None,
                     gridsize=500,
                     **kwargs):
    x = post[params[0]]
    y = post[params[1]]

    if transform is None:
        transform = lambda x: x

    deltax = x.max() - x.min()
    deltay = y.max() - y.min()
    x_pts = np.linspace(x.min() - .1 * deltax, x.max() + .1 * deltax, gridsize)
    y_pts = np.linspace(y.min() - .1 * deltay, y.max() + .1 * deltay, gridsize)

    xx, yy = np.meshgrid(x_pts, y_pts)

    positions = np.column_stack([xx.ravel(), yy.ravel()])

    # Calculate the KDE
    pts = np.array([x, y]).T
    selected_indices = np.random.choice(len(pts), len(pts) // 2, replace=False)
    kde_sel = np.zeros(len(pts), dtype=bool)
    kde_sel[selected_indices] = True
    kde_pts = transform(pts[kde_sel])
    untransformed_den_pts = pts[~kde_sel]
    den_pts = transform(untransformed_den_pts)

    Nden = den_pts.shape[0]

    post_kde = Bounded_2d_kde(kde_pts,
                              xlow=xlow,
                              xhigh=xhigh,
                              ylow=ylow,
                              yhigh=yhigh)
    den = post_kde(den_pts)
    inds = np.argsort(den)[::-1]
    den = den[inds]

    z = np.reshape(post_kde(transform(positions)), xx.shape)

    return {'xx': xx, 'yy': yy, 'z': z, 'kde': den, 'kde_sel': kde_sel}


def pickle_contour_data(event, post_name, cdata):
    outfile = os.path.join(contour_dir,
                           '{}_{}_contour_data.pkl'.format(event, post_name))
    with open(outfile, 'w') as outp:
        pickle.dump(cdata, outp)


for event in events:
    # Load posterior samples, replacing the final spin with more detailed estimates
    pos = np.genfromtxt(sample_files[event], names=True)
    #final_spin_pos = np.genfromtxt(sample_files[event]['comb_final_spin'], names=True)
    #pos['af'] = final_spin_pos['af']

    # m1-m2
    # post_name = 'm1_m2'
    # contour_data = estimate_2d_post(['m1_source', 'm2_source'], pos, transform=ms2q, yhigh=1.)
    # pickle_contour_data(event, post_name, contour_data)

    xlow, xhigh = 0.0, 1.0
    ylow, yhigh = -1.0, 1.0
    post_name = 'chieff_chip'
    contour_data = estimate_2d_post(['chi_p', 'chi_eff'],
                                    pos,
                                    xlow=xlow,
                                    xhigh=xhigh,
                                    ylow=ylow,
                                    yhigh=yhigh)
    pickle_contour_data(event, post_name, contour_data)

    # # final mass and final spin
    # post_name = 'mf_af'
    # ylow, yhigh = 0.0, 1.0
    # contour_data = estimate_2d_post(['mf_source', 'af'], pos, ylow=ylow, yhigh=yhigh)
    # pickle_contour_data(event, post_name, contour_data)

    # #final mass and final spin
    # post_name = 'q_chieff'
    # xlow, xhigh = 0.0, 1.0
    # ylow, yhigh = -1.0, 1.0
    # contour_data = estimate_2d_post(['q', 'chi_eff'], pos, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh)
    # pickle_contour_data(event, post_name, contour_data)

    # post_name = 'dist_inc'
    # pos['theta_jn'] *=  180./np.pi

    # # for i in range(len(pos['theta_jn'])):
    # #     if pos['theta_jn'][i] > 90:
    # #         pos['theta_jn'][i] = 180-pos['theta_jn'][i]

    # xlow, xhigh = 0, 180
    # contour_data = estimate_2d_post(['theta_jn', 'distance'], pos, xlow=xlow, xhigh=xhigh)
    # pickle_contour_data(event, post_name, contour_data)

    # xlow, xhigh = -1.0, 1.0
    # ylow, yhigh = 0.0, 1.0
    # post_name = 'a1_costilt1'
    # contour_data = estimate_2d_post(['costilt1', 'a1'], pos, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh)
    # pickle_contour_data(event, post_name, contour_data)

    # xlow, xhigh = -1.0, 1.0
    # ylow, yhigh = -1.0, 1.0
    # post_name = 'costilt1_costilt2'
    # contour_data = estimate_2d_post(['costilt2', 'costilt1'], pos, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh)
    # pickle_contour_data(event, post_name, contour_data)
