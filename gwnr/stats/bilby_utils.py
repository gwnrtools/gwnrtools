# Copyright (C) 2021 Prayush Kumar
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

import os
import glob
import logging
import numpy as np

from gwnr.stats.config_utils import ConfigBase


class InferenceConfigs(ConfigBase):
    def __init__(self, run_dir, configs={}):
        '''
        Stores config files for pycbc_inference runs

        Parameters
        ----------

        run_dir : string
        configs : dict


        Usage Notes
        -----------

        [1] Compatible with `ConfigWriter`.
        This class is easiest used with the writer it returns.

        [2] Arguments for `sampler.ini` and `inference.ini`
            are formatted in the initialization of this class

        Therefore, when configuring for Injections
        ------------------------------------------
        No special notes

        [3] Arguments for `data.ini` are not formatted in this class,
        but can be when writing it through its ConfigWriter.

        Therefore, when configuring for Events
        --------------------------------------
        Need the following named variables to be provided to the
        ConfigWriter's `write` function:

        gpstime       : int
        H1_frame_file : str
        H1_channel    : str
        L1_frame_file : str
        L1_channel    : str
        V1_frame_file : str
        V1_channel    : str
        sample_rate   : int (power of 2)

        '''
        super(InferenceConfigs, self).__init__(run_dir, configs)

        # Add prior configs
        if 'prior' not in self.configs:
            self.configs['prior'] = {}
        self.add_default_bbh_prior_config()
        self.add_bilby_prior_files_configs()

        # Add configs for injections
        if 'injection' not in self.configs:
            self.configs['injection'] = {}
        self.add_injection_configs()

        # Add event configs
        if 'event' not in self.configs:
            self.configs['event'] = {}
        from pycbc.catalog import Catalog
        self.event_names = Catalog().names
        for event_name in self.event_names:
            self.add_event_configs(event_name)

        # Initialize their config writers
        self.update_config_writers()

    def add_default_bbh_prior_config(self):
        self.set(
            'prior', 'default', '''\
mass_ratio = Uniform(name='mass_ratio', minimum=0.125, maximum=1, boundary='reflective')
chirp_mass = Uniform(name='chirp_mass', minimum=25, maximum=35, unit='$M_{\odot}$', boundary='reflective')
mass_1 = Constraint(name='mass_1', minimum=5, maximum=80)
mass_2 = Constraint(name='mass_2', minimum=5, maximum=80)
a_1 = Uniform(name='a_1', minimum=0, maximum=0.99, boundary='reflective')
a_2 = Uniform(name='a_2', minimum=0, maximum=0.99, boundary='reflective')
tilt_1 = Sine(name='tilt_1', boundary='reflective')
tilt_2 = Sine(name='tilt_2', boundary='reflective')
phi_12 = Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic')
phi_jl = Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic')
luminosity_distance = PowerLaw(alpha=2, name='luminosity_distance', minimum=50, maximum=2000, unit='Mpc', latex_label='$d_L$')
dec = Cosine(name='dec', boundary='reflective')
ra = Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')
theta_jn = Sine(name='theta_jn', boundary='reflective')
psi = Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
phase = Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')
geocent_time = Uniform(name='geocent_time', minimum =1126259460.4, maximum=1126259464.4)'''
        )
        self.set(
            'prior', 'eccentric-nonspin', '''\
eccentricity = Uniform(name='eccentricity', minimum=0, maximum=0.2, boundary='periodic')
mass_ratio = Uniform(name='mass_ratio', minimum=0.125, maximum=1, boundary='reflective')
chirp_mass = Uniform(name='chirp_mass', minimum=25, maximum=35, unit='$M_{\odot}$', boundary='reflective')
mass_1 = Constraint(name='mass_1', minimum=5, maximum=80)
mass_2 = Constraint(name='mass_2', minimum=5, maximum=80)
luminosity_distance = PowerLaw(alpha=2, name='luminosity_distance', minimum=50, maximum=2000, unit='Mpc', latex_label='$d_L$')
dec = Cosine(name='dec', boundary='reflective')
ra = Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')
theta_jn = Sine(name='theta_jn', boundary='reflective')
psi = Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
phase = Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')
geocent_time = Uniform(name='geocent_time', minimum =1126259460.4, maximum=1126259464.4)'''
        )

    def add_bilby_prior_files_configs(self):
        try:
            import bilby
            self.bilby_prior_files = {
                os.path.basename(f).split('.')[0]: f
                for f in glob.glob(
                    os.path.join(bilby.gw.prior.DEFAULT_PRIOR_DIR, '*.prior'))
            }

            for f in self.bilby_prior_files:
                self.set('prior', f,
                         ''.join(open(self.bilby_prior_files[f]).readlines()))
        except:
            logging.info("Failed to add priors from bilby repository")

    def add_event_configs(self, event_name):
        priors = self.get('prior')
        for pt in priors:
            self.set(
                'event', 'bbh-event-prior-{}'.format(pt), '''\

[workflow]
tag = test
accounting-group = ligo.dev.o3.cbc.explore.test
log-path = log
request-memory = 2G
request-cpus = 1

[inference]
duration = 4
sample_rate = 2048
lower_frequency_cutoff = 30
upper_frequency_cutoff = 1024
reference_frequency = 30
phase_marginalization =
time_marginalization =
;distance_marginalization =

[sampler]
name = dynesty
npoints = 2000
maxmcmc = 2000
n_check_point = 1000

[data]
analysis_type = event
source_type = bbh
event_names = GW150914,GW170104
interferometers = H1,L1


[template]
source_model = bilby.gw.source.lal_binary_black_hole
approximant = IMRPhenomPv2
sample_rate = 2048
lower_frequency_cutoff = 30
upper_frequency_cutoff = 1024
reference_frequency = 30


[prior]
{0}



'''.format(priors[pt]))

    def add_injection_configs(self):
        priors = self.get('prior')
        for pt in priors:
            self.set(
                'injection', 'bbh-prior-{}'.format(pt), '''\

[workflow]
tag = test
accounting-group = ligo.dev.o3.cbc.explore.test
log-path = log
request-memory = 2G
request-cpus = 1

[inference]
duration = 4
sample_rate = 2048
lower_frequency_cutoff = 30
upper_frequency_cutoff = 1024
reference_frequency = 30
phase_marginalization =
time_marginalization =
;distance_marginalization =

[sampler]
name = dynesty
npoints = 2000
maxmcmc = 2000
n_check_point = 1000


[data]
analysis_type = event
source_type = bbh
event_name = GW150914
interferometers = H1,L1


[injection]
noise_type = zero
;noise_type = gaussian
asd-file = H1:PATH,L1:PATH
approximant = IMRPhenomPv2
taper = start
num_injections = 10


[injection-static_params]
geocent_time = 1126259462.420
;mass_1 = 37
mass_2 = 32
a_1 = 0
a_2 = 0
tilt_1 = 0
tilt_2 = 0
phi_12 = 0
phi_jl = 0
ra = 2.2
dec = -1.25
theta_jn = 2.5
phase = 1.5
psi = 1.75
luminosity_distance = 1000


[injection-variable_params]
mass_1 =
eccentricity =
mean_per_ano =


[injection-prior_mass_1]
name = uniform
min-mass_1 = 10.
max-mass_1 = 80.

[injection-prior_eccentricity]
name = uniform
min-eccentricity = 0.
max-eccentricity = 0.2

[injection-prior_mean_per_ano]
name = uniform
min-mean_per_ano = 0.
max-mean_per_ano = 3.1416


[template]
source_model = bilby.gw.source.lal_binary_black_hole
approximant = IMRPhenomPv2
sample_rate = 2048
lower_frequency_cutoff = 30
upper_frequency_cutoff = 1024
reference_frequency = 30


[prior]
{0}



'''.format(priors[pt]))

            # Add configs for aligned-spin injections
            self.set(
                'injection', 'bbh-alignedspin-prior-{}'.format(pt), '''\

[workflow]
tag = test
accounting-group = ligo.dev.o3.cbc.explore.test
log-path = log
request-memory = 2G
request-cpus = 1

[inference]
duration = 4
sample_rate = 2048
lower_frequency_cutoff = 30
upper_frequency_cutoff = 1024
reference_frequency = 30
phase_marginalization =
time_marginalization =
;distance_marginalization =

[sampler]
name = dynesty
npoints = 2000
maxmcmc = 2000
n_check_point = 1000


[data]
analysis_type = event
source_type = bbh
event_name = GW150914
interferometers = H1,L1


[injection]
noise_type = gaussian
asd-file = H1:PATH,L1:PATH
approximant = IMRPhenomPv2
taper = start
num_injections = 10


[injection-static_params]
geocent_time = 1126259462.420
;mass_1 = 37
mass_2 = 32
chi_1 = 0
;chi_2 = 0
ra = 2.2
dec = -1.25
theta_jn = 2.5
phase = 1.5
psi = 1.75
luminosity_distance = 1000

[injection-variable_params]
mass_1 =
chi_2 =


[injection-prior_mass_1]
name = uniform
min-mass_1 = 10.
max-mass_1 = 80.


[injection-prior_chi_2]
name = uniform
min-chi_2 = -0.2
max-chi_2 = 0.2


[template]
source_model = bilby.gw.source.lal_binary_black_hole
approximant = IMRPhenomPv2
sample_rate = 2048
lower_frequency_cutoff = 30
upper_frequency_cutoff = 1024
reference_frequency = 30


[prior]
{0}



'''.format(priors[pt]))

            # Add configs for eccentric non-spin injections
            self.set(
                'injection', 'bbh-eccentric-nonspin-prior-{}'.format(pt), '''\

[workflow]
tag = test
accounting-group = ligo.dev.o3.cbc.explore.test
log-path = log
request-memory = 2G
request-cpus = 1

[inference]
duration = 4
sample_rate = 2048
lower_frequency_cutoff = 30
upper_frequency_cutoff = 1024
reference_frequency = 30
phase_marginalization =
time_marginalization =
;distance_marginalization =

[sampler]
name = dynesty
npoints = 2000
maxmcmc = 2000
n_check_point = 1000


[data]
analysis_type = event
source_type = bbh
event_name = GW150914
interferometers = H1,L1


[injection]
noise_type = gaussian
asd-file = H1:PATH,L1:PATH
approximant = EccentricFD
source_model = bilby.gw.source.lal_eccentric_binary_black_hole_no_spins
taper = start
num_injections = 10


[injection-static_params]
geocent_time = 1126259462.420
;mass_1 = 37
mass_2 = 32
mean_per_ano = 0
chi_1 = 0
;chi_2 = 0
ra = 2.2
dec = -1.25
theta_jn = 2.5
phase = 1.5
psi = 1.75
luminosity_distance = 1000

[injection-variable_params]
mass_1 =
eccentricity =


[injection-prior_mass_1]
name = uniform
min-mass_1 = 10.
max-mass_1 = 80.


[injection-prior_eccentricity]
name = uniform
min-eccentricity = 0.0
max-eccentricity = 0.2


[template]
source_model = bilby.gw.source.lal_binary_black_hole
approximant = IMRPhenomPv2
sample_rate = 2048
lower_frequency_cutoff = 30
upper_frequency_cutoff = 1024
reference_frequency = 30


[prior]
{0}




'''.format(priors[pt]))


class BilbyScriptWriterBase(object):
    def __init__(self,
                 tag,
                 source_type,
                 interferometer_list=['H1'],
                 inference_opts=dict(duration=4,
                                     sample_rate=2048,
                                     lower_frequency_cutoff=30.0,
                                     upper_frequency_cutoff=-1,
                                     reference_frequency=-1,
                                     phase_marginalization=True,
                                     distance_marginalization=True,
                                     time_marginalization=True),
                 template_opts=dict(
                     approximant='IMRPhenomPv2',
                     source_model='bilby.gw.source.lal_binary_black_hole',
                     sample_rate=2048,
                     lower_frequency_cutoff=30.0,
                     upper_frequency_cutoff=-1,
                     reference_frequency=-1),
                 sampler_opts=dict(name='dynesty', npoints=2000, maxmcmc=2000),
                 priors=[],
                 priors_file="priors.prior",
                 verbosity=0) -> None:
        self.verbosity = verbosity
        self.tag = tag

        self.source_type = source_type
        self.ifo_list = interferometer_list

        self.inference_opts = inference_opts
        self.template_opts = template_opts
        self.sampler_opts = sampler_opts
        self.priors = priors
        self.priors_filename = priors_file

        # Check for essential inputs
        for f in ['sample_rate', 'lower_frequency_cutoff', 'duration']:
            if f not in self.inference_opts:
                raise IOError("Please provide {} in inference_opts".format(f))

        # Use derived settings if explicit not provided
        # 1. check for various marker frequencies
        if 'upper_frequency_cutoff' not in self.inference_opts:
            self.inference_opts[
                'upper_frequency_cutoff'] = self.inference_opts[
                    'sample_rate'] // 2  # Nyquist
        elif self.inference_opts['upper_frequency_cutoff'] < 0:
            self.inference_opts[
                'upper_frequency_cutoff'] = self.inference_opts[
                    'sample_rate'] // 2  # Nyquist

        if 'reference_frequency' not in self.inference_opts:
            self.inference_opts['reference_frequency'] = self.inference_opts[
                'lower_frequency_cutoff']
        elif self.inference_opts['reference_frequency'] < 0:
            self.inference_opts['reference_frequency'] = self.inference_opts[
                'lower_frequency_cutoff']

        # 2. Disable marginalizations of phase/time/distance by default
        for f in [
                'phase_marginalization', 'distance_marginalization',
                'time_marginalization'
        ]:
            if f not in self.inference_opts:
                self.inference_opts[f] = False

        # 3. Set the same marker frequencies in template arguments
        for f in [
                'lower_frequency_cutoff', 'upper_frequency_cutoff',
                'reference_frequency'
        ]:
            if f not in self.template_opts:
                self.template_opts[f] = self.inference_opts[f]
            elif self.template_opts[f] < 0:
                self.template_opts[f] = self.inference_opts[f]

        # Enable checkpointing by default
        if 'n_check_point' not in self.sampler_opts:
            self.sampler_opts['n_check_point'] = 10000

        self._script_lines = []
        self._lines_added = []
        self.is_script_complete = False

    def add_import_lines(self):
        imps = [
            '''\
#!/usr/bin/env python
"""
This is an automatically generated script. See the `BilbyScriptWriterBase` class
and its derived in `gwnr` for details, and/or to make changes.
"""

import numpy as np
import bilby

'''
        ]
        if self.source_type == 'bns':
            imps.append(
                'from bilby.gw.eos import TabularEOS, EOSFamily # for BNS')

        if 'imports' not in self._lines_added:
            self._script_lines.extend(imps)
            self._lines_added.append('imports')
        return imps

    def add_initialization_lines(self):
        inits = [
            '''\

label = "{0}"
outdir = "outdir_" + label
logger = bilby.core.utils.logger

bilby.core.utils.setup_logger(outdir=outdir, label=label)
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)

# Analysis settings
duration = {1}
sampling_frequency = {2}
psd_duration = 32 * duration
post_trigger_duration = 2 # conservatively, ringdown cannot be longer than 2 seconds

'''.format(self.tag, self.inference_opts['duration'],
           self.inference_opts['sample_rate']),
        ]
        if 'initialization' not in self._lines_added:
            self._script_lines.extend(inits)
            self._lines_added.append('initialization')
        return inits

    def add_template_lines(self):
        tmpls = [
            '''\

# OVERVIEW OF APPROXIMANTS:
# https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/Waveforms/Overview
template_arguments = dict(waveform_approximant="{0}",
                          reference_frequency={1}, minimum_frequency={2})
conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters

template_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model={3},
    parameter_conversion=conversion,
    waveform_arguments=template_arguments)


            '''.format(self.template_opts['approximant'],
                       self.template_opts['reference_frequency'],
                       self.template_opts['lower_frequency_cutoff'],
                       self.template_opts['source_model']),
        ]
        if 'template' not in self._lines_added:
            self._script_lines.extend(tmpls)
            self._lines_added.append('template')
        return tmpls

    def add_prior_lines(self):
        if len(self.priors) == 0:
            if self.verbosity > 0:
                print("Using default priors for {}".format(self.source_type))
            if self.source_type == 'bbh':
                priors = [
                    '', '# Setup priors: the best way is to use a file',
                    'priors = bilby.gw.prior.BBHPriorDict()'
                ]
            elif self.source_type == 'bns':
                priors = ['', '', 'priors = bilby.gw.prior.BNSPriorDict()']
            else:
                raise IOError(
                    "Input `source_type` must be one of [bbh, bns, nsbh]")
        else:
            if self.verbosity > 0:
                print("Please do not forget to write '{0}'".format(
                    self.priors_filename))
            if self.source_type == 'bbh':
                priors = [
                    '', '# CHOOSE PRIOR FILE',
                    'priors = bilby.gw.prior.BBHPriorDict("{0}")'.format(
                        self.priors_filename), ''
                ]
            elif self.source_type == 'bns':
                priors = [
                    '', '# CHOOSE PRIOR FILE',
                    'priors = bilby.gw.prior.BNSPriorDict("{0}")'.format(
                        self.priors_filename), ''
                ]
            else:
                raise IOError(
                    "Input `source_type` must be one of [bbh, bns, nsbh]")
        priors.append('''\

priors["geocent_time"] = bilby.core.prior.Uniform(
    minimum=merger_time - 0.1, maximum=merger_time + 0.1,
    name="geocent_time", latex_label="$t_c$", unit="$s$")

    ''')
        if 'prior' not in self._lines_added:
            self._script_lines.extend(priors)
            self._lines_added.append('prior')
        return priors

    def add_likelihood_lines(self):
        liks = [
            '''\

# Setup the likelihood calculator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=template_generator, priors=priors,
    distance_marginalization={0}, phase_marginalization={1},
    time_marginalization={2})

'''.format(self.inference_opts['distance_marginalization'],
           self.inference_opts['phase_marginalization'],
           self.inference_opts['time_marginalization'])
        ]
        if 'likelihood' not in self._lines_added:
            self._script_lines.extend(liks)
            self._lines_added.append('likelihood')
        return liks

    def add_sampler_lines(self):
        smpl = [
            '''\

# Run the sampler
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, outdir=outdir, label=label,
    check_point_plot=True,'''
        ]
        for f in self.sampler_opts:
            v = str(self.sampler_opts[f])
            if v.isalpha():
                smpl.append('    {0}="{1}",'.format(f, v))  # string args
            else:
                smpl.append('    {0}={1},'.format(f, v))

        if self.analysis_type == 'injection':
            smpl.append('    injection_parameters=injection_parameters,')
        elif self.analysis_type == 'event':
            smpl.append('    use_ratio=False,')

        smpl.append(
            '    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters)'
        )
        if 'sampler' not in self._lines_added:
            self._script_lines.extend(smpl)
            self._lines_added.append('sampler')
        return smpl

    def add_post_processing_lines(self):
        ppl = ['''\

# Make final figures
result.plot_corner()

''']
        if 'post_processing' not in self._lines_added:
            self._script_lines.extend(ppl)
            self._lines_added.append('post_processing')
        return ppl

    def add_data_lines(self):
        raise NotImplementedError()

    def add_injection_lines(self):
        raise NotImplementedError()

    @property
    def script_lines(self):
        self.add_import_lines()
        self.add_initialization_lines()
        if self.analysis_type == 'event':
            if self.verbosity > 0:
                print("Writing script for event data")
            self.add_data_lines()
        elif self.analysis_type == 'injection':
            if self.verbosity > 0:
                print("Writing script for injection data")
            self.add_injection_lines()
        self.add_template_lines()
        self.add_prior_lines()
        self.add_likelihood_lines()
        self.add_sampler_lines()
        self.add_post_processing_lines()

        self.is_script_complete = True
        return self._script_lines

    def show_script(self):
        lines = self.script_lines
        for line in lines:
            print(line)

    def show_script_so_far(self):
        lines = self._script_lines
        for line in lines:
            print(line)

    def write_prior_file(self):
        with open(self.priors_filename, 'w') as f:
            for line in self.priors:
                f.write(line + '\n')

    def write_script(self, out_f_name=''):
        if len(out_f_name) == 0:
            out_f_name = self.tag + ".py"
        with open(out_f_name, 'w') as f:
            lines = self.script_lines
            if not self.is_script_complete:
                raise RuntimeError("Script has not been completed yet!")
            for line in lines:
                f.write(line + '\n')


class BilbyScriptWriterInjection(BilbyScriptWriterBase):
    def __init__(
            self,
            tag,
            source_type,
            injection_opts=dict(
                approximant='IMRPhenomPv2',
                source_model='bilby.gw.source.lal_binary_black_hole',
                parameters=dict(),
                noise_type='zero',  # or 'gaussian'
                asd=dict(),
                psd=dict(),
                sample_rate=2048,
                lower_frequency_cutoff=30.0,
                upper_frequency_cutoff=-1,
                reference_frequency=-1,
            ),
            interferometer_list=['H1'],
            inference_opts=dict(duration=4,
                                sample_rate=2048,
                                lower_frequency_cutoff=30.0,
                                upper_frequency_cutoff=-1,
                                reference_frequency=-1,
                                phase_marginalization=True,
                                distance_marginalization=True,
                                time_marginalization=True),
            template_opts=dict(
                approximant='IMRPhenomPv2',
                source_model='bilby.gw.source.lal_binary_black_hole',
                sample_rate=2048,
                lower_frequency_cutoff=30.0,
                upper_frequency_cutoff=-1,
                reference_frequency=-1),
            sampler_opts=dict(name='dynesty', npoints=2000, maxmcmc=2000),
            priors=[],
            priors_file="priors.prior",
            verbosity=0) -> None:
        self.analysis_type = 'injection'
        self.injection_opts = injection_opts

        # Check for compulsary input args
        for p in ['approximant', 'source_model', 'parameters']:
            if p not in self.injection_opts:
                raise IOError(
                    "Please provide {} as part of `injection_opts`".format(p))
        self.set_default_injection_parameters()

        # Initialize
        super(BilbyScriptWriterInjection,
              self).__init__(tag,
                             source_type,
                             interferometer_list=interferometer_list,
                             inference_opts=inference_opts,
                             template_opts=template_opts,
                             sampler_opts=sampler_opts,
                             priors=priors,
                             priors_file=priors_file,
                             verbosity=verbosity)
        # Set the same marker frequencies in template arguments
        for f in [
                'lower_frequency_cutoff', 'upper_frequency_cutoff',
                'reference_frequency'
        ]:
            if f not in self.injection_opts:
                self.injection_opts[f] = self.inference_opts[f]
            elif self.injection_opts[f] < 0:
                self.injection_opts[f] = self.inference_opts[f]

        # Check and set the choice of noise type for signal injection
        if 'noise_type' in self.injection_opts:
            if (self.injection_opts['noise_type'] != 'gaussian') and (
                    'zero' not in self.injection_opts['noise_type']):
                raise IOError(
                    "Injection noise_type must be either `gaussian` or `zero'")
        else:
            self.injection_opts['noise_type'] = 'zero'

    def set_default_injection_parameters(self):
        non_defaultable_parameters = ['mass_1', 'mass_2']
        for p in non_defaultable_parameters:
            if p not in self.injection_opts['parameters']:
                raise IOError(
                    "You must provide at least the mass_1 and mass_2 of the injection"
                )
        # And use default values when not provided by user
        injection_parameter_defaults = dict(chi_1=0.,
                                            chi_2=0.,
                                            a_1=0.,
                                            a_2=0.,
                                            tilt_1=0.,
                                            tilt_2=0.,
                                            phi_12=0.,
                                            phi_jl=0.,
                                            luminosity_distance=1000.,
                                            theta_jn=0.,
                                            psi=0.,
                                            phase=0.,
                                            geocent_time=1126259642.413,
                                            ra=0.,
                                            dec=0.)
        prececessing_spin_params = [
            'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl'
        ]
        aligned_spin_params = ['chi_1', 'chi_2']
        for p in injection_parameter_defaults:
            if (p not in prececessing_spin_params
                    and p not in aligned_spin_params) or (
                        p in prececessing_spin_params and np.all([
                            ap not in self.injection_opts['parameters']
                            for ap in aligned_spin_params
                        ])) or (p in aligned_spin_params and np.all([
                            ap not in self.injection_opts['parameters']
                            for ap in prececessing_spin_params
                        ])):
                if p not in self.injection_opts['parameters']:
                    self.injection_opts['parameters'][
                        p] = injection_parameter_defaults[p]

    def add_injection_lines(self):
        injs = ['# Injection parameters', 'injection_parameters = dict(']
        for p in self.injection_opts['parameters']:
            injs.append('    {}={},'.format(
                p, self.injection_opts['parameters'][p]))
        injs[-1] = injs[-1][:-1] + ')'

        injs.extend([
            '''\
merger_time = injection_parameters["geocent_time"]

# Setup the injection, starting with basic arguments
injection_arguments = dict(waveform_approximant="{0}",
                           reference_frequency={1}, minimum_frequency={2})

'''.format(self.injection_opts['approximant'],
           self.injection_opts['reference_frequency'],
           self.injection_opts['lower_frequency_cutoff']),
            '''\

injection_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model={0},
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=injection_arguments)


# GET DATA FROM INTERFEROMETER
interferometer_names = {1}

ifos = bilby.gw.detector.InterferometerList(interferometer_names)

'''.format(self.injection_opts['source_model'], self.ifo_list),
        ])
        gaussian_or_zero_noise = self.injection_opts['noise_type']
        if gaussian_or_zero_noise == 'gaussian':
            injs.append('ifos.set_strain_data_from_power_spectral_densities(')
        elif 'zero' in gaussian_or_zero_noise:
            injs.append('ifos.set_strain_data_from_zero_noise(')
        injs.extend([
            '''\
   sampling_frequency=sampling_frequency, duration=duration,
   start_time=injection_parameters[
        "geocent_time"] - duration + post_trigger_duration)
ifos.inject_signal(waveform_generator=injection_generator,
                   parameters=injection_parameters)

'''
        ])

        if 'data' not in self._lines_added:
            self._script_lines.extend(injs)
            self._lines_added.append('data')
        return injs


class BilbyScriptWriterEvent(BilbyScriptWriterBase):
    def __init__(self,
                 event_name,
                 source_type,
                 interferometer_list=['H1'],
                 inference_opts=dict(duration=4,
                                     sample_rate=2048,
                                     lower_frequency_cutoff=30.0,
                                     upper_frequency_cutoff=-1,
                                     reference_frequency=-1,
                                     phase_marginalization=True,
                                     distance_marginalization=True,
                                     time_marginalization=True),
                 template_opts=dict(
                     approximant='IMRPhenomPv2',
                     source_model='bilby.gw.source.lal_binary_black_hole',
                     sample_rate=2048,
                     lower_frequency_cutoff=30.0,
                     upper_frequency_cutoff=-1,
                     reference_frequency=-1),
                 sampler_opts=dict(name='dynesty', npoints=2000, maxmcmc=2000),
                 priors=[],
                 priors_file="priors.prior",
                 verbosity=0) -> None:
        self.analysis_type = 'event'
        self.event_name = event_name
        super(BilbyScriptWriterEvent, self).__init__(
            event_name,  # using event name as tag
            source_type,
            interferometer_list=interferometer_list,
            inference_opts=inference_opts,
            template_opts=template_opts,
            sampler_opts=sampler_opts,
            priors=priors,
            priors_file=priors_file,
            verbosity=verbosity)

    def add_import_lines(self):
        if 'imports' not in self._lines_added:
            imps = super(BilbyScriptWriterEvent, self).add_import_lines()
            ev_imps = ['from gwpy.timeseries import TimeSeries', '']
            imps.extend(ev_imps)
            self._script_lines.extend(ev_imps)
        else:
            imps = []
        return imps

    def add_data_lines(self):
        '''
TODO: Must allow users to enter arbitrary labelled inputs to be passed to
`TimeSeries.fetch_open_data()`. This would make the following future-proof
to changes in the gwpy API.
        '''
        assert (
            type(self.ifo_list) == list,
            "interferometer_names should be passed as a python list of strings"
        )
        data = [
            '''\

# Set data epochs to be used for analysis
event_name = "{0}"
time_of_event = bilby.gw.utils.get_event_time(event_name)
merger_time = time_of_event

end_time = time_of_event + post_trigger_duration
start_time = end_time - duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time

roll_off = 0.4

# GET DATA FROM INTERFEROMETER
interferometer_names = {1}

ifos = bilby.gw.detector.InterferometerList([])
for det in interferometer_names:
    logger.info("Downloading analysis data for ifo {{}}".format(det))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    data = TimeSeries.fetch_open_data(det,
                                      start_time,
                                      end_time)
    ifo.set_strain_data_from_gwpy_timeseries(data)

    # Additional arguments you might need to pass to TimeSeries.fetch_open_data:
    # - sample_rate = 4096, most data are stored by LOSC at this frequency
    # there may be event-related data releases with a 16384Hz rate.
    # - tag = 'CLN' for clean data; C00/C01 for raw data (different releases)
    # note that for O2 events a "tag" is required to download the data.
    # - channel =  {{'H1': 'H1:DCS-CALIB_STRAIN_C02',
    #               'L1': 'L1:DCS-CALIB_STRAIN_C02',
    #               'V1': 'V1:FAKE_h_16384Hz_4R'}}
    # for some events can specify channels: source data stream for LOSC data.
    logger.info("Downloading PSD data for ifo {{}}".format(det))
    psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time)
    psd_alpha = 2 * roll_off / duration # shape of tukey window
    psd = psd_data.psd(fftlength=duration,
                       overlap=0,
                       window=("tukey", psd_alpha),
                       method="median")
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value)
    ifos.append(ifo)

'''.format(self.event_name, self.ifo_list),
        ]
        if 'data' not in self._lines_added:
            self._script_lines.extend(data)
            self._lines_added.append('data')
        return data
