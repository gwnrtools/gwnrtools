#!/usr/bin/env python
#
# Copyright (C) 2020 Prayush Kumar
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

####
# **`ConfigWriter`**:
# - takes in a dictionary of `configs` that contains different ini files
# - writes them to desired output file


class ConfigWriter():
    def __init__(self, name, configs, run_dir):
        '''
        Writer class for configuration files

        Parameters
        ----------

        name : string
            The name that configuration file will be written to. 
            This does not depend on the available options 
        configs : dict
            Has key:value pairs for different ini file texts
        run_dir : string
            Run directory where the configuration files are to be written
        '''
        self.name = name
        self.configs = configs
        self.run_dir = run_dir

    def write(self, name, **formatting_kwargs):
        '''
        Config file string may have some blanks that need to 
        be filled, especially for data configs for GW events.
        '''
        out_str = self.configs[name]
        with open(os.path.join(self.run_dir, name + '.ini'), 'w') as fout:
            if len(formatting_kwargs) > 0:
                fout.write(out_str.format(**formatting_kwargs))
            else:
                fout.write(out_str)

    def types(self):
        return self.configs.keys()
####


####
# **`InferenceConfigs`**:
# - stores all `config.ini` files
# - returns on demand. Compatible with ConfigWriter


class InferenceConfigs():
    def __init__(self, run_dir, configs={}):
        '''
        Stores config files for pycbc_inference runs

        Parameters
        ----------

        run_dir : string
        configs : dict

        Usage
        -----
        Compatible with ConfigWriter.
        This class is easiest used with the writer it returns.

        Configs for Injections
        ----------------------
        No special notes


        Configs for Events
        ------------------
        Need the following named variables to be provided
        to the config writer's `write` function:

        gpstime       : int
        H1_frame_file : str
        H1_channel    : str
        L1_frame_file : str
        L1_channel    : str
        V1_frame_file : str
        V1_channel    : str
        sample_rate   : int (power of 2)

        '''
        self.run_dir = run_dir
        # Make this >>
        assert(isinstance(configs, dict))
        self.configs = configs

        # Add data configs
        if 'data' not in self.configs:
            self.configs['data'] = {}
        self.add_data_configs()

        # Add data configs for events
        import pycbc.catalog
        self.event_names = pycbc.catalog.Catalog().names
        for event_name in self.event_names:
            self.add_data_configs(event_name)

        # Add sampler configs
        if 'sampler' not in self.configs:
            self.configs['sampler'] = {}
        self.add_sampler_configs()

        # Add inference configs
        if 'inference' not in self.configs:
            self.configs['inference'] = {}
        self.add_inference_configs()

        self.config_names = self.configs.keys()

        # Initialize their config writers
        self.config_writers = {}
        for config_name in self.config_names:
            self.config_writers[config_name] = ConfigWriter(
                config_name + '.ini',
                self.configs[config_name], run_dir)

    def available_configs(self):
        return self.config_names

    def get_config_writer(self, name):
        assert(name in self.available_configs())
        return self.config_writers[name]

    def get(self, config_name, type_name=None):
        if type_name in self.configs[config_name]:
            return self.configs[config_name][type_name]
        return self.configs[config_name]

    def set(self, config_name, config):
        self.configs[config_name] = configs

    def add_data_configs(self, event_name=None):
        # Events
        if event_name is not None:
            if '150914' in event_name or '170104' in event_name or\
                    '151012' in event_name or '170608' in event_name or\
                    '151226' in event_name or '170823' in event_name:
                self.configs['data'][event_name] = """\
[data]
instruments = H1 L1
trigger-time = {gpstime}
analysis-start-time = -6
analysis-end-time = 2
psd-estimation = median-mean
psd-start-time = -256
psd-end-time = 256
psd-inverse-length = 8
psd-segment-length = 8
psd-segment-stride = 4
; The frame files must be downloaded from GWOSC before running.
frame-files = H1:{H1_frame_file} L1:{L1_frame_file}
channel-name = {H1_channel} {L1_channel}
; this will cause the data to be resampled to 2048 Hz:
sample-rate = {sample_rate}
; We'll use a high-pass filter so as not to get numerical errors from the large
; amplitude low frequency noise. Here we use 15 Hz, which is safely below the
; low frequency cutoff of our likelihood integral (20 Hz)
strain-high-pass = 15
; The pad-data argument is for the high-pass filter: 8s are added to the
; beginning/end of the analysis/psd times when the data is loaded. After the
; high pass filter is applied, the additional time is discarded. This pad is
; *in addition to* the time added to the analysis start/end time for the PSD
; inverse length. Since it is discarded before the data is transformed for the
; likelihood integral, it has little affect on the run time.
pad-data = 8
"""
            elif '170729' in event_name or '170814' in event_name or\
                    '170809' in event_name or '170818' in event_name:
                self.configs['data'][event_name] = """\
[data]
instruments = H1 L1 V1
trigger-time = {gpstime}
analysis-start-time = -6
analysis-end-time = 2
psd-estimation = median-mean
psd-start-time = -256
psd-end-time = 256
psd-inverse-length = 8
psd-segment-length = 8
psd-segment-stride = 4
; The frame files must be downloaded from GWOSC before running.
frame-files = H1:{H1_frame_file} L1:{L1_frame_file} V1:{V1_frame_file}
channel-name = {H1_channel} {L1_channel} {V1_channel}
; this will cause the data to be resampled to 2048 Hz:
sample-rate = {sample_rate}
; We'll use a high-pass filter so as not to get numerical errors from the large
; amplitude low frequency noise. Here we use 15 Hz, which is safely below the
; low frequency cutoff of our likelihood integral (20 Hz)
strain-high-pass = 15
; The pad-data argument is for the high-pass filter: 8s are added to the
; beginning/end of the analysis/psd times when the data is loaded. After the
; high pass filter is applied, the additional time is discarded. This pad is
; *in addition to* the time added to the analysis start/end time for the PSD
; inverse length. Since it is discarded before the data is transformed for the
; likelihood integral, it has little affect on the run time.
pad-data = 8
"""
            elif '170817' in event_name:
                self.configs['data'][event_name] = """\
[data]
instruments = H1 L1 V1 G1
trigger-time = {gpstime}
analysis-start-time = -6
analysis-end-time = 2
psd-estimation = median-mean
psd-start-time = -256
psd-end-time = 256
psd-inverse-length = 8
psd-segment-length = 8
psd-segment-stride = 4
; The frame files must be downloaded from GWOSC before running.
frame-files = H1:{H1_frame_file} L1:{L1_frame_file} V1:{V1_frame_file} G1:{G1_frame_file}
channel-name = {H1_channel} {L1_channel} {V1_channel} {G1_channel}
; this will cause the data to be resampled to 2048 Hz:
sample-rate = {sample_rate}
; We'll use a high-pass filter so as not to get numerical errors from the large
; amplitude low frequency noise. Here we use 15 Hz, which is safely below the
; low frequency cutoff of our likelihood integral (20 Hz)
strain-high-pass = 15
; The pad-data argument is for the high-pass filter: 8s are added to the
; beginning/end of the analysis/psd times when the data is loaded. After the
; high pass filter is applied, the additional time is discarded. This pad is
; *in addition to* the time added to the analysis start/end time for the PSD
; inverse length. Since it is discarded before the data is transformed for the
; likelihood integral, it has little affect on the run time.
pad-data = 8
"""
            return

        # Injections
        import numpy
        self.configs['data']['gw150914-like-gaussian'] = """\
[data]
instruments = H1 L1
trigger-time = 1126259462.42
analysis-start-time = -6
analysis-end-time = 2
; strain settings
sample-rate = 2048
fake-strain = H1:aLIGOaLIGODesignSensitivityT1800044 L1:aLIGOaLIGODesignSensitivityT1800044
fake-strain-seed = H1:{0} L1:{1}
; psd settings
psd-estimation = median-mean
psd-start-time = -256
psd-end-time = 256
psd-inverse-length = 8
psd-segment-length = 8
psd-segment-stride = 4
; even though we're making fake strain, the strain
; module requires a channel to be provided, so we'll
; just make one up
channel-name = H1:STRAIN L1:STRAIN
; Providing an injection file will cause a simulated
; signal to be added to the data
injection-file = injection.hdf
; We'll use a high-pass filter so as not to get numerical errors from the large
; amplitude low frequency noise. Here we use 15 Hz, which is safely below the
; low frequency cutoff of our likelihood integral (20 Hz)
strain-high-pass = 15
; The pad-data argument is for the high-pass filter: 8s are added to the
; beginning/end of the analysis/psd times when the data is loaded. After the
; high pass filter is applied, the additional time is discarded. This pad is
; *in addition to* the time added to the analysis start/end time for the PSD
; inverse length. Since it is discarded before the data is transformed for the
; likelihood integral, it has little affect on the run time.
pad-data = 8
""".format(numpy.random.randint(1, 1e5), numpy.random.randint(1, 1e5))
        self.configs['data']['gw150914-like-zeronoise'] = """\
[data]
instruments = H1 L1
trigger-time = 1126259462.42
analysis-start-time = -6
analysis-end-time = 2
; strain settings
sample-rate = 2048
fake-strain = H1:zeroNoise L1:zeroNoise
; psd settings
psd-model = aLIGOZeroDetHighPower
psd-inverse-length = 0
; even though we're making fake strain, the strain
; module requires a channel to be provided, so we'll
; just make one up
channel-name = H1:STRAIN L1:STRAIN
; Providing an injection file will cause a simulated
; signal to be added to the data
injection-file = injection.hdf
; We'll use a high-pass filter so as not to get numerical errors from the large
; amplitude low frequency noise. Here we use 15 Hz, which is safely below the
; low frequency cutoff of our likelihood integral (20 Hz)
strain-high-pass = 15
; The pad-data argument is for the high-pass filter: 8s are added to the
; beginning/end of the analysis/psd times when the data is loaded. After the
; high pass filter is applied, the additional time is discarded. This pad is
; *in addition to* the time added to the analysis start/end time for the PSD
; inverse length. Since it is discarded before the data is transformed for the
; likelihood integral, it has little affect on the run time.
pad-data = 8
"""

    def add_sampler_configs(self):
        self.configs['sampler']['emcee'] = """\
[sampler]
name = emcee
nwalkers = 1000
niterations = 2000
;##### Other possible options
effective-nsamples = 1000
max-samples-per-chain = 1000
checkpoint-interval = 2000

;[sampler-burn_in]
;burn-in-test = nacl & max_posterior
"""
        self.configs['sampler']['emcee_pt'] = """\
[sampler]
name = emcee_pt
nwalkers = 500
ntemps = 20
;##### Other possible options
effective-nsamples = 4000
checkpoint-interval = 2000
max-samples-per-chain = 1000

[sampler-burn_in]
burn-in-test = nacl & max_posterior

;
;   Sampling transforms
;
[sampling_params]
; parameters on the left will be sampled in
; parametes on the right
mass1, mass2 : mchirp, q

[sampling_transforms-mchirp+q]
; inputs mass1, mass2
; outputs mchirp, q
name = mass1_mass2_to_mchirp_q
"""
        self.configs['sampler']['dynesty'] = """\
[sampler]
name = dynesty
dlogz = 0.1
nlive = 1500

; Other arguments (see Dynesty package for details).
; https://dynesty.readthedocs.io/en/latest/quickstart.html#nested-sampling-with-dynesty
; bound, bootstrap, enlarge, update_interval, sample
; loglikelihood-function = loglr
"""
        self.configs['sampler']['ultranest'] = """\
[sampler]
name = ultranest
dlogz = 0.1

;##### Other possible options (see ultranest package for useage)
; update_interval_iter_fraction, update_interval_ncall
; log_interval, show_status, dKL, frac_remain,
; Lepsilon, min_ess, max_iters, max_ncalls,
; max_num_improvement_loops, 
min_num_live_points = 1500
; cluster_num_live_points
"""
        self.configs['sampler']['epsie'] = """\
[sampler]
name = epsie
nchains = 100
niterations = 100
ntemps = 4

;##### Other possible options
;effective-nsamples = 1000
;max-samples-per-chain = 1000
;checkpoint-interval = 2000

;[sampler-burn_in]
;burn-in-test = nacl & max_posterior

[jump_proposal-x]
name = normal
"""
        self.configs['sampler']['multinest'] = """\
[sampler]
name = multinest
nlivepoints = 1500

;##### Optional arguments
;evidence-tolerance = 0.1
;sampling-efficiency = 0.3
;checkpoint-interval = 5000
;importance-nested-sampling = True
"""
        self.configs['sampler']['cpnest'] = """\
[sampler]
;
; WARNING: this sampler requires python3 support
;
name = cpnest
nthreads = 8 
nlive = 1500 ;(anything between 1000 (faster) and 2000 (slower), should be good)
maxmcmc = 10000 ;(you should always use >= 5000)
verbose = 1

[sampler-burn_in]
burn-in-test = nacl & max_posterior

;
;   Sampling transforms
;
[sampling_params]
; parameters on the left will be sampled in
; parametes on the right
mass1, mass2 : mchirp, q

[sampling_transforms-mchirp+q]
; inputs mass1, mass2
; outputs mchirp, q
name = mass1_mass2_to_mchirp_q
"""

    def add_inference_configs(self):
        self.configs['inference']['bbh_precessing'] = """\
[model]
name = gaussian_noise
low-frequency-cutoff = 20.0

[variable_params]
; waveform parameters that will vary in MCMC
delta_tc =
mass1 =
mass2 =
spin1_a =
spin1_azimuthal =
spin1_polar =
spin2_a =
spin2_azimuthal =
spin2_polar =
distance =
coa_phase =
inclination =
polarization =
ra =
dec =

[static_params]
; waveform parameters that will not change in MCMC
approximant = IMRPhenomPv2
f_lower = 20
f_ref = 20
; we'll set the tc by using the trigger time in the data
; section of the config file + delta_tc
trigger_time = ${data|trigger-time}

[prior-delta_tc]
; coalescence time prior
name = uniform
min-delta_tc = -0.1
max-delta_tc = 0.1

[waveform_transforms-tc]
; we need to provide tc to the waveform generator
name = custom
inputs = delta_tc
tc = ${data|trigger-time} + delta_tc

;Mass1 of GW151012 $\in$ [28.7, 38.1]
;Mass1 of GW170608 $\in$ [12.7, 16.5]
;Mass1 of GW170729 $\in$ [60.4, 66.4]
;Mass1 of GW150914 $\in$ [38.7, 40.3]
;Mass1 of GW151226 $\in$ [16.9, 22.5]
;Mass1 of GW170814 $\in$ [33.6, 36.2]
;Mass1 of GW170817 $\in$ [1.56, 1.58]
;Mass1 of GW170104 $\in$ [36.4, 38.1]
;Mass1 of GW170809 $\in$ [40.9, 43.3]
;Mass1 of GW170818 $\in$ [40.1, 42.9]
;Mass1 of GW170823 $\in$ [46.2, 50.7]

[prior-mass1]
name = uniform
min-mass1 = 10.
max-mass1 = 80.

;Mass2 of GW151012 $\in$ [18.4, 17.7]
;Mass2 of GW170608 $\in$ [9.8, 9.0]
;Mass2 of GW170729 $\in$ [44.1, 43.1]
;Mass2 of GW150914 $\in$ [35.0, 33.6]
;Mass2 of GW151226 $\in$ [10.2, 9.9]
;Mass2 of GW170814 $\in$ [29.2, 28.0]
;Mass2 of GW170817 $\in$ [1.36, 1.36]
;Mass2 of GW170104 $\in$ [24.6, 24.9]
;Mass2 of GW170809 $\in$ [29.0, 28.9]
;Mass2 of GW170818 $\in$ [31.9, 31.0]
;Mass2 of GW170823 $\in$ [36.8, 35.7]

[prior-mass2]
name = uniform
min-mass2 = 10.
max-mass2 = 80.

[prior-spin1_a]
name = uniform
min-spin1_a = 0.0
max-spin1_a = 0.99

[prior-spin1_polar+spin1_azimuthal]
name = uniform_solidangle
polar-angle = spin1_polar
azimuthal-angle = spin1_azimuthal

[prior-spin2_a]
name = uniform
min-spin2_a = 0.0
max-spin2_a = 0.99

[prior-spin2_polar+spin2_azimuthal]
name = uniform_solidangle
polar-angle = spin2_polar
azimuthal-angle = spin2_azimuthal

[prior-distance]
; following gives a uniform volume prior
name = uniform_radius
min-distance = 10
max-distance = 1000

[prior-coa_phase]
; coalescence phase prior
name = uniform_angle

[prior-inclination]
; inclination prior
name = sin_angle

[prior-ra+dec]
; sky position prior
name = uniform_sky

[prior-polarization]
; polarization prior
name = uniform_angle
"""
####
