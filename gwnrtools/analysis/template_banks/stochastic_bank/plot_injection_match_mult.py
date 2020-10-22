#!/usr/bin/env python
import matplotlib
#matplotlib.use('Agg')

import os
from optparse import OptionParser
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from numpy import loadtxt, histogram2d
import pylab
from pylab import sqrt
import qm
import numpy
from numpy import meshgrid, linspace

### option parsing ###
parser = OptionParser()
parser.add_option('-t',
                  '--tmplt-bank-file',
                  metavar='FILE',
                  help='file containing template to plot',
                  type=str)
parser.add_option('-s',
                  '--sim-file',
                  metavar='FILE',
                  help='file containing simulated injections',
                  type=str)
parser.add_option(
    '-m',
    '--match-file',
    metavar='FILE',
    help='file containing the max over bank values for the injections')
parser.add_option('-o', '--out-file', metavar='FILE', help='output plot name')
parser.add_option("-V",
                  "--verbose",
                  action="store_true",
                  help="print extra debugging information",
                  default=False)
parser.add_option("-f", dest="f_min", help="low frequency cutoff")
parser.add_option("-n", dest="num", help="num of inj", type=int)
options, argv_frame_files = parser.parse_args()


def eta_to_q(eta):
    q = (1. + sqrt(1. - 4. * eta)) / (2. * eta) - 1.
    return q


pltid = 0

mass1 = []
mass2 = []
tau0 = []
tau3 = []
matches = []
ttau0 = []
ttau3 = []
mchirp = []
eta = []
q = []
mtotal = []
chi1 = []
chi2 = []

for n in range(options.num):
    # Load in the overlap values
    match_name = options.match_file + str(n) + ".dat"
    if (os.path.exists(match_name) == False):
        continue
    match = numpy.atleast_1d(loadtxt(match_name))
    if (len(match) == 0):
        continue
    matches.extend(match)

    # Load in the simulation list
    sim_name = options.sim_file + str(n) + ".xml"
    indoc = ligolw_utils.load_filename(sim_name, options.verbose)
    sim_table = table.get_table(indoc, lsctables.SimInspiralTable.tableName)

    for injection in sim_table:
        if injection.mass2 < injection.mass1:
            mass1.append(injection.mass1)
            mass2.append(injection.mass2)
        else:
            mass1.append(injection.mass2)
            mass2.append(injection.mass1)

        t0, t3 = qm.m1_m2_to_tau0_tau3(injection.mass2 * qm.QM_MTSUN_SI,
                                       injection.mass1 * qm.QM_MTSUN_SI, 40)
        tau0.append(t0)
        tau3.append(t3)

        mc, et = qm.m1_m2_to_mchirp_eta(injection.mass1, injection.mass2)
        mchirp.append(mc)
        eta.append(et)
        q.append(eta_to_q(et))
        mtotal.append(injection.mass1 + injection.mass2)
        chi1.append(injection.spin1z)
        chi2.append(injection.spin2z)

# Load the template bank
indoc = ligolw_utils.load_filename(options.tmplt_bank_file, options.verbose)
sngl_inspiral_table = table.get_table(indoc,
                                      lsctables.SimInspiralTable.tableName)

for template in sngl_inspiral_table:
    t0, t3 = qm.m1_m2_to_tau0_tau3(template.mass2 * qm.QM_MTSUN_SI,
                                   template.mass1 * qm.QM_MTSUN_SI, 40)
    ttau0.append(t0)
    ttau3.append(t3)

print(len(sngl_inspiral_table))
print(len(matches), min(matches))
print(len(mass1))
print(len(mass2))
print('MM=%f' % min(matches))

finetachifile = open('../run01/FinalEtaChi.dat', 'r')
finetachi = loadtxt(finetachifile)
tmpltq = []
tmpltch = []
for et, ch in finetachi:
    tmpltq.append(eta_to_q(et))
    if ch > 0.9: ch = 0.9
    if ch < -0.9: ch = -0.9
    tmpltch.append(ch)

# setting some figure properties
# taken from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
fig_height_pt = 0.3 * 672.0
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_height_pt * inches_per_pt  # height in inches
#fig_height = fig_width * golden_mean      # height in inches
fig_size = [1.1 * fig_width, 1.1 * fig_height]
pylab.rcParams.update(  # try to match font sizes of document
    {'axes.labelsize': 9,
   'axes.titlesize': 9,
   'text.fontsize': 9,
   'legend.fontsize': 9,
   'xtick.labelsize': 8,
   'ytick.labelsize': 8,
   'text.usetex': True,
   'figure.figsize': fig_size,
   'font.family': 'serif',
   'font.serif': ['palatino'],
   'savefig.dpi': 300
   })

mass1cut = []
mass2cut = []
mchirpcut = []
etacut = []
mtotalcut = []
matchescut = []
chi1cut = []
chi2cut = []
mtotalthreshold = 140.

qthreshold = 4. / 3.
etathreshold = 0.  #qthreshold / (1. + qthreshold)**2.
MMthreshold = 0.97

uncfile = open('uncovered_points.dat', "w+")

index = 0
for index in range(len(mass1)):
    if (matches[index] < MMthreshold):
        if matches[index] < .97:  # and eta[index] >= (10./121.):
            #print "%f\t%f\t%f" % (mchirp[index],eta[index],matches[index])
            uncfile.write("%12.18f\t%12.18f\n" % (mtotal[index], chi1[index]))
        mass1cut.append(mass1[index])
        mass2cut.append(mass2[index])
        mtotalcut.append(mass1[index] + mass2[index])
        mchirpcut.append(mchirp[index])
        etacut.append(eta[index])
        matchescut.append(matches[index])
        chi1cut.append(chi1[index])
        chi2cut.append(chi2[index])

uncfile.close()
#exit()

#title_str = 'Injected: EOBNRv2; Recovered with: EOBNRv2'
title_str = ''

# Make the overlap histogram
nbins = 150
pylab.figure(pltid)
pltid += 1
pylab.axes([0.15, 0.125, 0.95 - 0.15, 0.95 - 0.2])
pylab.hist(matches,
           bins=nbins,
           normed=True,
           cumulative=True,
           log=True,
           color='blue',
           alpha=0.6)
pylab.grid()
pylab.xlim(xmax=1.)
pylab.xlabel("FF (X)")
pylab.ylabel("Fraction of sample points with FF $>$ X")
#pylab.title(title_str)
pylab.savefig('smallOverlaphist.pdf')
pylab.savefig("Overlaphist.png", dpi=1000)
pylab.savefig("Overlaphist.eps", dpi=1000)

# plot the mass-mass plane
x_min = min(mass1)
x_max = 100.  #max(mass1)

y_min = min(mass2cut)
y_max = 100.  #max(mass2cut)

qline1 = pylab.arange(3. * qthreshold, 25.001, 0.001)
qline2 = qline1 / qthreshold

pylab.figure(pltid)
pltid += 1
pylab.axes([0.125, 0.125, 0.95 - 0.125, 0.95 - 0.2])
#pylab.scatter(mass1,mass2,c=matches,s=.5,linewidths=0,cmap=matplotlib.colors.LinearSegmentedColormap('my_colormap',cm.datad.get('gist_heat_r'),256))
pylab.scatter(mass1, mass2, c=matches, s=.5, linewidths=0, vmin=0.98)

#pylab.hexbin(mass1cut,mass2cut,matchescut,vmin=0.96)
cb = pylab.colorbar()
cb.set_label("FF")
#pylab.scatter(sngl_inspiral_table.get_column('mass1'),sngl_inspiral_table.get_column('mass2'),s=.1,linewidths=0,c='black')

pylab.grid()
pylab.xlim(0, x_max + 1)
pylab.ylim(0, y_max + 1)
pylab.xlabel('$m_1$ ($M_\odot$)')
pylab.ylabel('$m_2$ ($M_\odot$)')
#pylab.title(title_str)
pylab.savefig("small" + options.out_file)
pylab.savefig(options.out_file, dpi=1000)

# plot the mass-mass plane
pylab.figure(pltid)
pltid += 1
pylab.axes([0.125, 0.125, 0.95 - 0.125, 0.95 - 0.2])
pylab.scatter(mass1cut, mass2cut, c=matchescut, s=.5, vmin=0.9, linewidths=0)
#pylab.hexbin(mass1cut,mass2cut,matchescut,vmin=0.96)
cb = pylab.colorbar()
cb.set_label("FF")
#pylab.scatter(sngl_inspiral_table.get_column('mass1'),sngl_inspiral_table.get_column('mass2'),s=.1,linewidths=0,c='black')

x_min = min(mass1cut)
x_max = max(mass1cut)

y_min = min(mass2cut)
y_max = x_max  #max(mass2cut)

pylab.grid()
pylab.xlim(0, x_max + 1)
pylab.ylim(0, y_max + 1)
pylab.xlabel('$m_1$ ($M_\odot$)')
pylab.ylabel('$m_2$ ($M_\odot$)')
pylab.title(title_str)
pylab.savefig("smallcut" + options.out_file)
pylab.savefig("cut" + options.out_file, dpi=1000)

# plot the tau0-tau3 plane
pylab.figure(pltid, figsize=(30, 3))
pltid += 1
pylab.scatter(tau0, tau3, c=matches, s=.05, linewidths=0)
cb = pylab.colorbar()
cb.set_label("Effectualness")
pylab.scatter(ttau0, ttau3, s=.1, linewidths=0, c='black')

pylab.grid()
pylab.xlabel('$\\tau_0$ (s)')
pylab.ylabel('$\\tau_3$ (s)')
#pylab.title(title_str)
pylab.savefig("smallTAU0TAU3" + options.out_file)
pylab.savefig("TAU0TAU3" + options.out_file, dpi=1000)

print("MAKING eta chi1 GRAPH")
pylab.figure(pltid)
pltid += 1
pylab.axes([0.125, 0.125, 0.95 - 0.125, 0.95 - 0.2])
pylab.scatter(eta, chi1, c=matches, s=.5, vmin=0.98, vmax=1., linewidths=0)
pylab.ylim([-1.05, 1.05])
pylab.xlim([min(eta) - .01, max(eta) + .01])
pylab.colorbar()

pylab.grid()
pylab.xlabel('$\eta$')
pylab.ylabel('$\chi$')
pylab.title(title_str)
pylab.savefig("smalletachi1" + options.out_file)
pylab.savefig("etachi1" + options.out_file, dpi=1000)

print("MAKING eta chi2 GRAPH")
pylab.figure(pltid)
pltid += 1
pylab.axes([0.125, 0.125, 0.95 - 0.125, 0.95 - 0.2])
pylab.scatter(eta, chi2, c=matches, s=.5, vmin=0.98, vmax=1.0, linewidths=0)
pylab.ylim([-1.05, 1.05])
pylab.xlim([0., max(eta) + .1])
pylab.colorbar()

print("MAKING q chi1 GRAPH")
pylab.figure(pltid)
pltid += 1
pylab.axes([0.125, 0.125, 0.95 - 0.125, 0.95 - 0.2])
pylab.scatter(q, chi1, c=matches, s=.5, vmin=0.98, vmax=1., linewidths=0)
pylab.ylim([-1.05, 1.05])
pylab.xlim([min(q) - 1., max(q) + 1])
pylab.colorbar()

pylab.grid()
pylab.xlabel('$q$')
pylab.ylabel('$\chi$')
pylab.title(title_str)
pylab.savefig("smallqchi1" + options.out_file)
pylab.savefig("qchi1" + options.out_file, dpi=1000)

print("MAKING q chi2 GRAPH")
pylab.figure(pltid)
pltid += 1
pylab.axes([0.125, 0.125, 0.95 - 0.125, 0.95 - 0.2])
pylab.scatter(q, chi2, c=matches, s=.5, vmin=0.98, vmax=1.0, linewidths=0)
pylab.ylim([-1.05, 1.05])
pylab.xlim([min(q) - 1., max(q) + 1])
pylab.colorbar()

pylab.grid()
pylab.xlabel('$q$')
pylab.ylabel('$\chi$')
pylab.title(title_str)
pylab.savefig("smallqchi2" + options.out_file)
pylab.savefig("qchi2" + options.out_file, dpi=1000)

print("MAKING MCHIRP/ETA GRAPH")
pylab.figure(pltid)
pltid += 1
pylab.axes([0.135, 0.125, 0.95 - 0.125, 0.95 - 0.2])
pylab.scatter(mchirp, eta, c=matches, s=.5, vmin=0.96, linewidths=0)
#pylab.hexbin(mchirpcut,etacut,matchescut,vmin=0.96)
cb = pylab.colorbar()
#cb.set_label("Recovered FF")

pylab.grid()
#pylab.xlim(0, 10.)
pylab.xlabel('$\mathcal{M}_c$ ($M_\odot$)')
pylab.ylabel('$\eta$')
#pylab.title(title_str)
pylab.savefig("smallmchirpeta" + options.out_file)
pylab.savefig("mchirpeta" + options.out_file, dpi=1000)

print("MAKING MCHIRP/MTOTAL GRAPH")
pylab.figure(pltid)
pltid += 1
pylab.axes([0.125, 0.125, 0.95 - 0.125, 0.95 - 0.2])
pylab.scatter(mchirp, mtotal, c=matches, s=.5, linewidths=0)
pylab.colorbar()

pylab.grid()
pylab.xlabel('$\mathcal{M}_c$')
pylab.ylabel('M')
pylab.title(title_str)
pylab.savefig("smallmchirpmtotal" + options.out_file)
pylab.savefig("mchirpmtotal" + options.out_file, dpi=1000)

print("MAKING mtotal eta GRAPH")
pylab.figure(pltid)
pltid += 1
pylab.axes([0.125, 0.125, 0.95 - 0.125, 0.95 - 0.2])
pylab.scatter(mtotal, eta, c=matches, s=.5, vmin=0.96, linewidths=0)
pylab.colorbar()

pylab.grid()
pylab.xlabel('M')
pylab.ylabel('$\eta$')
pylab.title(title_str)
pylab.savefig("smallmtotaleta" + options.out_file)
pylab.savefig("mtotaleta" + options.out_file, dpi=1000)

print("MAKING mtotal chi1 GRAPH")
pylab.figure(pltid)
pltid += 1
pylab.axes([0.125, 0.125, 0.95 - 0.125, 0.95 - 0.2])
pylab.scatter(mtotal, chi1, c=matches, s=.5, vmin=0.98, vmax=1., linewidths=0)
pylab.ylim([-1.05, 1.05])
pylab.xlim([0., max(mtotal) + 1.])
pylab.colorbar()

pylab.grid()
pylab.xlabel('M')
pylab.ylabel('$\chi$')
pylab.title(title_str)
pylab.savefig("smallmtotalchi1" + options.out_file)
pylab.savefig("mtotalchi1" + options.out_file, dpi=1000)

pylab.figure(pltid)
pltid += 1
pylab.axes([0.125, 0.125, 0.95 - 0.125, 0.95 - 0.2])
pylab.scatter(mtotalcut, chi1cut, c=matchescut, s=.5, linewidths=0)
pylab.colorbar()

pylab.grid()
pylab.xlabel('M')
pylab.ylabel('$\chi$')
pylab.title(title_str)
pylab.savefig("smallcutmtotalchi1" + options.out_file)
pylab.savefig("cutmtotalchi1" + options.out_file, dpi=1000)

print("MAKING mtotal chi2 GRAPH")
pylab.figure(pltid)
pltid += 1
pylab.axes([0.125, 0.125, 0.95 - 0.125, 0.95 - 0.2])
pylab.scatter(mtotal, chi2, c=matches, s=.5, vmax=1.0, linewidths=0)
pylab.ylim([-1.05, 1.05])
pylab.xlim([0., max(mtotal) + 1.])
pylab.colorbar()

pylab.grid()
pylab.xlabel('M')
pylab.ylabel('$\chi$')
pylab.title(title_str)
pylab.savefig("smallmtotalchi2" + options.out_file)
pylab.savefig("mtotalchi2" + options.out_file, dpi=1000)

pylab.figure(pltid)
pltid += 1
pylab.axes([0.125, 0.125, 0.95 - 0.125, 0.95 - 0.2])
pylab.scatter(mtotalcut, chi2cut, c=matchescut, s=.5, linewidths=0)
pylab.colorbar()

pylab.grid()
pylab.xlabel('M')
pylab.ylabel('$\chi$')
pylab.title(title_str)
pylab.savefig("smallcutmtotalchi2" + options.out_file)
pylab.savefig("cutmtotalchi2" + options.out_file, dpi=1000)

# Make a histogram of points where the bank does not cover
qcut1 = []
chi1cut1 = []
matchescut1 = []

for i in range(len(chi1)):
    if matches[i] < 0.97:
        chi1cut1.append(chi1[i])
        matchescut1.append(matches[i])
        qcut1.append(q[i])

H, xedges, yedges = histogram2d(chi1cut1, qcut1, bins=(50, 50), normed=True)
H1, xedges1, yedges1 = histogram2d(chi1, q, bins=(50, 50))
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        H[i][j] /= H1[i][j]

extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
pylab.figure(pltid)
pltid += 1
pylab.axes([0.125, 0.125, 0.95 - 0.125, 0.95 - 0.2])
pylab.imshow(H, extent=extent, interpolation='nearest')
cb = pylab.colorbar()
cb.set_label("No of samples with FF $< 0.97$")
pylab.hold(True)
pylab.plot(tmpltq, tmpltch, 'kx', markersize=3.)
#pylab.hist2d(qcut1, chi1cut1, bins = nbins, range=None, weights=None, cmin=None, cmax=None )
pylab.grid()
pylab.xlim(xmax=1.)
pylab.ylim([min(chi1cut1) - .1, max(chi1cut1) + .1])
pylab.xlim([min(qcut1) - 1., max(qcut1) + 1.])
pylab.xlabel("$q$")
pylab.ylabel("$\chi_1$")

#pylab.title(title_str)
pylab.savefig('FFqchihist.pdf')
pylab.savefig("smallFFchihist.png", dpi=1000)
pylab.savefig("smallFFchihist.eps", dpi=1000)

# Make a histogram of points
H, xedges, yedges = histogram2d(chi1, q, bins=(50, 50))
extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
pylab.figure(pltid)
pltid += 1
pylab.axes([0.125, 0.125, 0.95 - 0.125, 0.95 - 0.2])
pylab.imshow(H, extent=extent, interpolation='nearest')
cb = pylab.colorbar()
cb.set_label("FF")
#pylab.hist2d(qcut1, chi1cut1, bins = nbins, range=None, weights=None, cmin=None, cmax=None )
pylab.grid()
pylab.xlim(xmax=1.)
pylab.ylim([min(chi1) - .1, max(chi1) + .1])
pylab.xlim([min(q) - 1., max(q) + 1.])
pylab.xlabel("$q$")
pylab.ylabel("$\chi_1$")

pylab.savefig("smallqchi1hist" + options.out_file)
pylab.savefig("qchi1hist" + options.out_file, dpi=1000)

# Plotting the matches as a contour plot
x_min = min(q)
x_max = max(q)

y_min = min(chi1)
y_max = max(chi1)

# plot the mass-mass plane
pylab.figure(pltid)
pltid += 1
pylab.axes([0.125, 0.125, 0.95 - 0.125, 0.95 - 0.2])
qrange = linspace(min(q), max(q), 200)
chi1range = linspace(min(chi1), max(chi1), 200)
qmap, chi1map = meshgrid(qrange, chi1range)
colormap = matplotlib.mlab.griddata(
    q, chi1, matches, qmap,
    chi1map)  #[(1.-x) for x in matches], qmap, chi1map)

#pylab.contour(qmap, chi1map, colormap,  [0.947, .965], colors='black', linestyles='dashed' )
#pylab.hold(True)
#pylab.contourf( qmap, chi1map, colormap, [0.,.01,.02,.03,.04,.05,.06,.07])#, cmap=matplotlib.colors.LinearSegmentedColormap('my_colormap',cm.datad.get('hot_r'),256) )
pylab.contourf(
    qmap, chi1map, colormap, [.93, .95, .965, .97, .98, .99, 1.]
)  #, cmap=matplotlib.colors.LinearSegmentedColormap('my_colormap',cm.datad.get('hot_r'),256) )
pylab.hold(True)
pylab.plot(tmpltq, tmpltch, 'kx', markersize=3.)
#pylab.contourf( m1map, m2map, colormap, sort(append(linspace( min(matchescut), 0.96, 6 ), array([0.965, 0.97, 0.98, 0.99]))),cmap=matplotlib.colors.LinearSegmentedColormap('my_colormap',cm.datad.get('hot_r'),256)) # )
#CS = pylab.contour( m1map, m2map, colormap ) #, sort(append(linspace( min(matchescut), 0.96, 6 ), array([0.965, 0.97, 0.98, 0.99]))),cmap=matplotlib.colors.LinearSegmentedColormap('my_colormap',cm.datad.get('hot_r'),256)) # )
#pylab.clabel(CS, inline=1, fontsize=10 )
cb = pylab.colorbar()
cb.set_label("FF")
pylab.xlim(x_min - 1, x_max + 1)
pylab.ylim(y_min - 0.01, y_max + .01)
pylab.xlabel('$q$')
pylab.ylabel('$\chi_1$')
pylab.grid()

#pylab.savefig("smallContour"+options.out_file)
#pylab.savefig("Contour"+options.out_file,dpi=1000)
pylab.savefig("ContourmatchesQChi1.pdf", dpi=1000)
pylab.savefig("ContourmatchesQChi1.png", dpi=1000)

# plot the mass-mass plane
pylab.figure(pltid)
pltid += 1
pylab.axes([0.125, 0.125, 0.95 - 0.125, 0.95 - 0.2])
pylab.hexbin(q, chi1, C=matches, bins=50)

cb = pylab.colorbar()
cb.set_label("FF")
pylab.hold(True)
pylab.plot(tmpltq, tmpltch, 'kx', markersize=3.)
pylab.xlim(x_min - 1, x_max + 1)
pylab.ylim(y_min - 0.01, y_max + .01)
pylab.xlabel('$q$')
pylab.ylabel('$\chi_1$')
pylab.grid()

#pylab.savefig("smallContour"+options.out_file)
#pylab.savefig("Contour"+options.out_file,dpi=1000)
pylab.savefig("HistmatchesQChi1.pdf", dpi=1000)
pylab.savefig("HistmatchesQChi1.png", dpi=1000)
