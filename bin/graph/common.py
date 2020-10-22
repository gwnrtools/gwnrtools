import os

import numpy as np

from matplotlib import patheffects
from matplotlib.ticker import Formatter

# Make the font match the document
rc_params = {
    'backend': 'ps',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'font.size': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Bitstream Vera Sans']
}

# Figure out figure size to avoid rescaling in the document
column_width = 246.0
inches_per_pt = 1.0 / 72.27
fig_width = 2 * column_width * inches_per_pt
rc_params['figure.figsize'] = (fig_width, fig_width * 1.6)

# IFOs online
ifos = ['H1', 'L1']
ifo_labels = {'H1': 'Hanford', 'L1': 'Livingston'}
ifo_colors = {'H1': 'red', 'L1': 'blue'}

# Reference time
ref_time = 1135136350.65

# Set the random seed for reproducibility
random_seed = 1234

# Models run
models = ['seob', 'imrpp']

# Data locations
contour_dir = ''
sample_dir = ''

# events = ['GW170814_HL','GW170814_HLV']
# sample_files = {'GW170814_HL':'HL_posterior_samples.dat','GW170814_HLV':'HLV_posterior_samples.dat'}

events = ['GW170814_IMR']
sample_files = {'GW170814_IMR': 'IMR.dat'}

model_sample_files = {}

# posterior files for the final m - final a plot
model_sample_files['imrpp'] = os.path.join(
    sample_dir, 'allIMRPPsp_post_evolved_Mfaf_Erad_Lpeak.dat')
model_sample_files['seob'] = os.path.join(
    sample_dir, 'SEOBNR3_marg_sp_post_evolved_Mfaf_Erad_Lpeak.dat')
comb_sample_file = os.path.join(sample_dir,
                                'overall_3marg_sp_post_Mfaf_Erad_Lpeak.dat')

# posterior samples for all other files
model_sample_files['imrpp'] = os.path.join(sample_dir, 'allIMRPPsp_post.dat')
model_sample_files['seob'] = os.path.join(sample_dir,
                                          'SEOBNR3_marg_sp_post.dat')
comb_sample_file = os.path.join(sample_dir, 'overall_3marg_sp_post.dat')

imrpp_prior_sample_file = os.path.join(sample_dir, 'allIsp_prior.dat')

# PSD files
psd_files = [
    os.path.join(sample_dir, 'h1_psd.dat'),
    os.path.join(sample_dir, 'l1_psd.dat')
]

# Filtered input
filtered_dir = '../data/OutputFromMakeFigures/'
filtered_time_files = dict([(ifo,
                             os.path.join(filtered_dir,
                                          'Times{}.txt'.format(ifo)))
                            for ifo in ifos])
filtered_strain_files = dict([(ifo,
                               os.path.join(filtered_dir,
                                            'Filtered{}.txt'.format(ifo)))
                              for ifo in ifos])

# Locations for intermediate data products
reconstruct_dir = './reconstructions'

data_file_fmt = 'data_residuals_{}.dat.gz'
cbc_file_fmt = '{}_{}.dat.gz'

data_residual_files = dict([(ifo,
                             os.path.join(reconstruct_dir,
                                          data_file_fmt.format(ifo)))
                            for ifo in ifos])

cbc_files = {}
for model in models:
    cbc_files[model] = {}
    for ifo in ifos:
        cbc_files[model][ifo] = os.path.join(reconstruct_dir,
                                             cbc_file_fmt.format(model, ifo))

norms_file = os.path.join(reconstruct_dir, 'norms.dat.gz')


def ms2q(pts):
    """Transformation function from component masses to chirp mass and mass ratio"""
    pts = np.atleast_2d(pts)

    m1 = pts[:, 0]
    m2 = pts[:, 1]
    mc = np.power(m1 * m2, 3. / 5.) * np.power(m1 + m2, -1. / 5.)
    q = m2 / m1
    return np.column_stack([mc, q])


class DegreeFormatter(Formatter):
    def __call__(self, x, pos=None):
        return r"${:3.0f}^{{\circ}}$".format(x)


# Define colors for each model
colors = {'imrpp': 'steelblue', 'seob': 'firebrick', 'imrpp_prior': 'seagreen'}

# Path effect for white outlines
white_outline = [patheffects.withStroke(linewidth=2., foreground="w")]
