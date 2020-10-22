#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import os
from optparse import OptionParser
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from numpy import loadtxt
import pylab
from pylab import sqrt
import qm
from os.path import isfile
import numpy

### option parsing ###
parser = OptionParser()
parser.add_option('-s',
                  '--sim-file',
                  metavar='FILE',
                  help='file containing simulated injections',
                  type=str)
parser.add_option('-b',
                  '--bank-file',
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
parser.add_option("--max-total-mass", type=float, default=None)
parser.add_option("-n", dest="num", help="num of inj", type=int)
options, argv_frame_files = parser.parse_args()

matches = []
injected = []
recovered = []
import numpy

template_tables = {}
index = 0
while isfile(options.bank_file + str(index) + ".xml"):
    indoc = ligolw_utils.load_filename(options.bank_file + str(index) + ".xml",
                                       options.verbose)
    tmp_table = table.get_table(indoc, lsctables.SnglInspiralTable.tableName)
    template_tables[options.bank_file + str(index) + ".xml"] = tmp_table
    index = index + 1
    #print index

for n in range(options.num):
    #print n
    # Load in the overlap values
    match_name = options.match_file + str(n) + ".dat"
    if (os.path.exists(match_name) == False):
        continue
    match = numpy.atleast_1d(loadtxt(match_name))
    if (len(match) == 0):
        continue

    for m in match:
        matches.append(m)
    #print n

    # Load in the simulation list
    sim_name = options.sim_file + str(n) + ".xml"
    indoc = ligolw_utils.load_filename(sim_name, options.verbose)
    sim_table = table.get_table(indoc, lsctables.SimInspiralTable.tableName)
    for inj in sim_table:
        injected.append(inj)

    # Load in the Recovered template list
    recover_name = match_name + ".found"
    r_file = open(recover_name)
    for line in r_file:
        part = line.split(None)
        #print part
        tmp_name = part[0]
        recovered.append(template_tables[tmp_name][int(part[1])])

rec_eta = []
inj_eta = []
rec_mchirp = []
inj_mchirp = []
m1 = []
m2 = []
inj_tau0 = []
inj_tau3 = []
rec_tau0 = []
rec_tau3 = []

# setting some figure properties
# taken from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
fig_height_pt = 0.3 * 672.0
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_height_pt * inches_per_pt  # height in inches
#fig_height = fig_width * golden_mean      # height in inches
fig_size = [fig_width, fig_height]
pylab.rcParams.update(  # try to match font sizes of document
    {'axes.labelsize': 10,
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

for index in range(0, len(matches)):
    rec_eta.append(recovered[index].eta)
    inj_eta.append(injected[index].eta)

    if hasattr(recovered[index], 'mchirp'):
        rec_mchirp.append(recovered[index].mchirp)
    else:
        mc = recovered[index].mtotal * (recovered[index].eta**0.6)
        rec_mchirp.append(mc)
    inj_mchirp.append(injected[index].mchirp)

    itau0, itau3 = qm.m1_m2_to_tau0_tau3(injected[index].mass1,
                                         injected[index].mass2, 15)
    if hasattr(recovered[index], 'mass1'):
        rtau0, rtau3 = qm.m1_m2_to_tau0_tau3(recovered[index].mass1,
                                             recovered[index].mass2, 15)
    else:
        rtau0, rtau3 = qm.m1_m2_to_tau0_tau3(1, 1, 15)
    inj_tau0.append(itau0)
    inj_tau3.append(itau3)
    rec_tau0.append(rtau0)
    rec_tau3.append(rtau3)

    if injected[index].mass1 > injected[index].mass2:
        m1.append(injected[index].mass1)
        m2.append(injected[index].mass2)
    else:
        m1.append(injected[index].mass2)
        m2.append(injected[index].mass1)

print(len(inj_eta))
print(len(matches))

rec_eta = numpy.array(rec_eta)
inj_eta = numpy.array(inj_eta)
rec_mchirp = numpy.array(rec_mchirp)
inj_mchirp = numpy.array(inj_mchirp)
inj_tau0 = numpy.array(inj_tau0)
inj_tau3 = numpy.array(inj_tau3)
rec_tau0 = numpy.array(rec_tau0)
rec_tau3 = numpy.array(rec_tau3)

mchirp_err = (inj_mchirp - rec_mchirp) / inj_mchirp
eta_err = (inj_eta - rec_eta) / inj_eta
tau0_err = (inj_tau0 - rec_tau0) / inj_tau0
tau3_err = (inj_tau3 - rec_tau3) / inj_tau3

if not options.max_total_mass:
    pylab.figure(1)
    pylab.plot([min(inj_eta), max(inj_eta)],
               [min(inj_eta), max(inj_eta)],
               '-',
               color='black')
    pylab.scatter(inj_eta, rec_eta, c=matches, s=4, linewidths=0)
    pylab.grid()
    pylab.colorbar()
    pylab.xlabel('Injected Eta')
    pylab.ylabel('Recovered Eta')
    pylab.savefig("eta" + options.out_file, dpi=300)

    pylab.figure(2)
    pylab.plot([min(inj_mchirp), max(inj_mchirp)],
               [min(inj_mchirp), max(inj_mchirp)],
               '-',
               color='black')
    pylab.scatter(inj_mchirp, rec_mchirp, c=matches, s=4, linewidths=0)
    pylab.grid()
    pylab.colorbar()
    pylab.xlabel('Injected Mchirp')
    pylab.ylabel('Recovered Mchirp')
    pylab.savefig("mchirp" + options.out_file, dpi=300)

    pylab.figure(3)
    pylab.xlabel('$\Delta \eta/\eta$')
    n, bins, patches = pylab.hist(eta_err,
                                  bins=100,
                                  normed=True,
                                  histtype="stepfilled")
    pylab.savefig("hist_eta_err" + options.out_file)

    pylab.figure(4)
    pylab.xlabel('$\Delta \mathcal{M}_c/\mathcal{M}_c$')
    n, bins, patches = pylab.hist(mchirp_err,
                                  bins=100,
                                  normed=True,
                                  histtype="stepfilled")
    pylab.savefig("hist_mchirp_err" + options.out_file)

    # plot the mass-mass plane
    pylab.figure(5)
    pylab.scatter(m1, m2, s=1, c=mchirp_err, linewidths=0)
    pylab.colorbar()
    pylab.grid()
    pylab.xlabel('Mass ($M_\odot$)')
    pylab.ylabel('Mass ($M_\odot$)')
    pylab.title("Fractional Error in Mchirp")
    pylab.savefig("m1m2_mchirp_err" + options.out_file, dpi=500)

    pylab.figure(6)
    pylab.scatter(m1, m2, s=1, c=eta_err, linewidths=0)
    pylab.colorbar()
    pylab.grid()
    pylab.xlabel('Mass ($M_\odot$)')
    pylab.ylabel('Mass ($M_\odot$)')
    pylab.title("Fractional Error in Eta")
    pylab.savefig("m1m2_eta_err" + options.out_file, dpi=500)

    from numpy import log10, abs

    pylab.figure(20)
    pylab.scatter(m1, m2, s=1, c=log10(abs(mchirp_err)), linewidths=0)
    pylab.colorbar()
    pylab.grid()
    pylab.xlabel('Mass ($M_\odot$)')
    pylab.ylabel('Mass ($M_\odot$)')
    pylab.title("Fractional Error in Mchirp")
    pylab.savefig("logm1m2_mchirp_err" + options.out_file, dpi=500)

    pylab.figure(21)
    pylab.scatter(m1, m2, s=1, c=log10(abs(eta_err)), linewidths=0)
    pylab.colorbar()
    pylab.grid()
    pylab.xlabel('Mass ($M_\odot$)')
    pylab.ylabel('Mass ($M_\odot$)')
    pylab.title("Fractional Error in Eta")
    pylab.savefig("logm1m2_eta_err" + options.out_file, dpi=500)

    pylab.figure(13)
    pylab.scatter(m1, m2, s=1, c=tau0_err, linewidths=0)
    pylab.colorbar()
    pylab.grid()
    pylab.xlabel('Mass ($M_\odot$)')
    pylab.ylabel('Mass ($M_\odot$)')
    pylab.title("Fractional Error in Tau0")
    pylab.savefig("m1m2_tau0_err" + options.out_file, dpi=500)

    pylab.figure(14)
    pylab.scatter(m1, m2, s=1, c=tau3_err, linewidths=0)
    pylab.colorbar()
    pylab.grid()
    pylab.xlabel('Mass ($M_\odot$)')
    pylab.ylabel('Mass ($M_\odot$)')
    pylab.title("Fractional Error in tau3")
    pylab.savefig("m1m2_tau3_err" + options.out_file, dpi=500)

if options.max_total_mass:
    errE = []
    errM = []
    for index in range(0, len(m1)):
        if (m1[index] + m2[index]) < options.max_total_mass:
            errE.append(eta_err[index])
            errM.append(mchirp_err[index])

    pylab.figure(7)
    pylab.xlabel(
        'Fractional Error in ($\eta$): ($\frac{\eta_injected - \eta_recovered}{\eta_injected}$)'
    )
    pylab.ylabel('Number of Monte Carlo Points')
    n, bins, patches = pylab.hist(errE, bins=100, histtype="stepfilled")
    pylab.savefig("hist_eta_err" + str(options.max_total_mass) +
                  options.out_file)

    pylab.figure(8)
    pylab.xlabel(
        'Fractional Error in ($(\mathcal{M}_c): ($\frac{\mathcal{M}_injection - \matchcal{M}_recovered}{$\mathcal{M}_injected})'
    )
    pylab.ylabel('Number of Monte Carlo Points')
    n, bins, patches = pylab.hist(errM, bins=100, histtype="stepfilled")
    pylab.savefig("hist_mchirp_err" + str(options.max_total_mass) +
                  options.out_file)
