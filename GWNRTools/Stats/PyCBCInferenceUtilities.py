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

    def write(self, name):
        with open(os.path.join(self.run_dir, name + '.ini'), 'w') as fout:
            fout.write(self.configs[name])

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
        Stores all config.ini files

        Parameters
        ----------

        run_dir : string
        configs : dict

        Returns on demand. Compatible with ConfigWriter
        '''
        self.run_dir = run_dir
        # Make this >>
        assert(isinstance(configs, dict))
        self.configs = configs

        # Add data / sampler / inference configs
        if 'data' not in self.configs:
            self.configs['data'] = {}
        self.add_data_configs()

        if 'sampler' not in self.configs:
            self.configs['sampler'] = {}
        self.add_sampler_configs()

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

    def add_data_configs(self):
        self.configs['data']['gw150914-like-gaussian'] = """\
[data]
instruments = H1 L1
trigger-time = 1126259462.42
analysis-start-time = -6
analysis-end-time = 2
; strain settings
sample-rate = 2048
fake-strain = H1:aLIGOaLIGODesignSensitivityT1800044 L1:aLIGOaLIGODesignSensitivityT1800044
fake-strain-seed = H1:44 L1:45
; psd settings
psd-estimation = median-mean
psd-inverse-length = 8
psd-segment-length = 8
psd-segment-stride = 4
psd-start-time = -256
psd-end-time = 256
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
        self.configs['sampler']['emcee_pt_v1'] = """\
[sampler]
name = emcee_pt
nwalkers = 500
ntemps = 20
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
        self.configs['sampler']['cpnest_v1'] = """\
[sampler]
name = cpnest
nthreads = 8 
nlive = 1500 ;(anything between 1000 (faster) and 2000 (slower), should be good)
maxmcmc = 10000 ;(you should always use >= 5000)

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
        self.configs['inference']['gw150914_like'] = """\
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

[prior-mass1]
name = uniform
min-mass1 = 10.
max-mass1 = 80.

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

####
# **`InferenceConfigs`**:
# - stores all `config.ini` files
# - returns on demand. Compatible with ConfigWriter


class EventInferenceDataConfigs():
    def __init__(self, run_dir, configs={}):
        '''
        Stores config files for pycbc_inference that 
        provide the prescribed settings for handling data.

        Parameters
        ----------

        run_dir : string
            Directory to write the data configuration file to
        configs : dict (optional)
            User specified dictionary containing different
            config files
        '''
        import pycbc.catalog
        self.event_names = pycbc.catalog.Catalog().names

        self.run_dir = run_dir  # Make this >>
        assert(isinstance(configs, dict))
        self.configs = configs

        # Add data / sampler / inference configs
        if 'data' not in self.configs:
            self.configs['data'] = {}
        for event in self.event_names:
            self.add_data_configs(event)

        self.config_names = self.configs.keys()

        # Initialize their config writers
        self.config_writers = {}
        for config_name in self.config_names:
            self.config_writers[config_name] = ConfigWriter(
                config_name + '.ini',
                self.configs[config_name], run_dir)

    def available_configs(self):
        return self.config_names

    def available_events(self):
        return self.configs['data'].keys()

    def get_config_writer(self, name):
        assert(name in self.available_configs())
        return self.config_writers[name]

    def get(self, config_name, type_name=None):
        if type_name in self.configs[config_name]:
            return self.configs[config_name][type_name]
        return self.configs[config_name]

    def set(self, config_name, config):
        self.configs[config_name] = configs

    def add_data_configs(self, event_name):
        if '150914' or '170104' in event_name:
            self.configs['data'][event_name] = """\
[data]
instruments = H1 L1
trigger-time = 1126259462.43
; See the documentation at
; http://pycbc.org/pycbc/latest/html/inference.html#simulated-bbh-example
; for details on the following settings:
analysis-start-time = -6
analysis-end-time = 2
psd-estimation = median-mean
psd-start-time = -256
psd-end-time = 256
psd-inverse-length = 8
psd-segment-length = 8
psd-segment-stride = 4
; The frame files must be downloaded from GWOSC before running. Here, we
; assume that the files have been downloaded to the same directory. Adjust
; the file path as necessary if not.
frame-files = H1:H-H1_GWOSC_16KHZ_R1-1126257415-4096.gwf L1:L-L1_GWOSC_16KHZ_R1-1126257415-4096.gwf
channel-name = H1:GWOSC-16KHZ_R1_STRAIN L1:GWOSC-16KHZ_R1_STRAIN
; this will cause the data to be resampled to 2048 Hz:
sample-rate = 2048
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
####
