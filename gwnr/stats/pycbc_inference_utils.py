# Copyright (C) 2020 Prayush Kumar
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

from gwnr.stats.config_utils import ConfigBase


class InferenceConfigs(ConfigBase):
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
    def __init__(
            self,
            run_dir,
            configs={},
            # workflow opts
            n_cpus=10,
            checkpoint_interval=2000,
            # nested samplers opts
            n_live=2000,
            n_maxmcmc=8000,
            d_logz=0.1,
            # parallel mcmc opts
            n_walkers=1000,
            n_temperatures=20,
            n_maxsamps_per_walker=1000,
            n_eff_samples=4000):
        super(InferenceConfigs, self).__init__(run_dir, configs)

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
        self.add_sampler_configs(n_cpus=n_cpus,
                                 n_live=n_live,
                                 n_maxmcmc=n_maxmcmc,
                                 d_logz=d_logz,
                                 n_walkers=n_walkers,
                                 n_temperatures=n_temperatures,
                                 n_maxsamps_per_walker=n_maxsamps_per_walker,
                                 n_eff_samples=n_eff_samples,
                                 ckpt_interval=checkpoint_interval)

        # Add inference configs
        if 'inference' not in self.configs:
            self.configs['inference'] = {}
        self.add_inference_configs()

        # Initialize their config writers
        self.update_config_writers()

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
{psd_options}
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
{psd_options}
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
instruments = H1 L1 V1
trigger-time = {gpstime}
analysis-start-time = -6
analysis-end-time = 2
{psd_options}
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
""".format(numpy.random.randint(1, 1e6), numpy.random.randint(1, 1e6))
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

    def add_sampler_configs(self,
                            n_cpus=10,
                            n_live=2000,
                            n_maxmcmc=8000,
                            d_logz=0.1,
                            n_walkers=1000,
                            n_temperatures=20,
                            n_maxsamps_per_walker=1000,
                            n_eff_samples=4000,
                            ckpt_interval=2000):
        self.configs['sampler']['emcee'] = """\
[sampler]
name = emcee
nprocesses = {n_cpus}
nwalkers = {n_walkers}
effective-nsamples = {n_eff_samples}
max-samples-per-chain = {n_maxsamps_per_walker}
checkpoint-interval = {ckpt_interval}

;[sampler-burn_in]
;burn-in-test = nacl & max_posterior
""".format(n_cpus=n_cpus,
           n_walkers=n_walkers,
           n_eff_samples=n_eff_samples,
           n_maxsamps_per_walker=n_maxsamps_per_walker,
           ckpt_interval=ckpt_interval)
        self.configs['sampler']['emcee_pt'] = """\
[sampler]
name = emcee_pt
nprocesses = {n_cpus}
nwalkers = {n_walkers}
ntemps = {n_temperatures}
effective-nsamples = {n_eff_samples}
max-samples-per-chain = {n_maxsamps_per_walker}
checkpoint-interval = {ckpt_interval}

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
""".format(n_cpus=n_cpus,
           n_walkers=n_walkers,
           n_temperatures=n_temperatures,
           n_maxsamps_per_walker=n_maxsamps_per_walker,
           n_eff_samples=n_eff_samples,
           ckpt_interval=ckpt_interval)
        self.configs['sampler']['epsie'] = """\
[sampler]
name = epsie
nprocesses = {n_cpus}
checkpoint-interval = {ckpt_interval}
nchains = {n_walkers}
ntemps = {n_temperatures}
effective-nsamples = {n_eff_samples}
max-samples-per-chain = {n_maxsamps_per_walker}

;[sampler-burn_in]
;burn-in-test = nacl & max_posterior

[jump_proposal-distance]
name = normal


[jump_proposal-mass1]
name = normal


[jump_proposal-delta_tc]
name = normal


[jump_proposal-mass2]
name = normal


[jump_proposal-spin2_polar]
name = normal


[jump_proposal-spin1_polar]
name = normal


[jump_proposal-spin2_azimuthal]
name = normal


[jump_proposal-polarization]
name = normal


[jump_proposal-spin1_azimuthal]
name = normal


[jump_proposal-ra]
name = normal


[jump_proposal-inclination]
name = normal


[jump_proposal-spin1_a]
name = normal


[jump_proposal-dec]
name = normal


[jump_proposal-coa_phase]
name = normal


[jump_proposal-spin2_a]
name = normal
""".format(n_cpus=n_cpus,
           n_walkers=n_walkers,
           n_eff_samples=n_eff_samples,
           n_temperatures=n_temperatures,
           n_maxsamps_per_walker=n_maxsamps_per_walker,
           ckpt_interval=ckpt_interval)
        self.configs['sampler']['dynesty'] = """\
[sampler]
name = dynesty
nprocesses = {n_cpus}
dlogz = {d_logz}
nlive = {n_live}
sample = rwalk ; uniform, rwalk, rstagger, slice, rslice, hslice
bound = multi  ; none, single, multi, balls, cubes

; Other arguments (see Dynesty package for details).
; https://dynesty.readthedocs.io/en/latest/quickstart.html#nested-sampling-with-dynesty
; bound =
; bootstrap =
; enlarge =
; update_interval =
; loglikelihood-function = loglr
""".format(n_cpus=n_cpus, d_logz=d_logz, n_live=n_live)
        self.configs['sampler']['ultranest'] = """\
[sampler]
name = ultranest
dlogz = {d_logz}
min_num_live_points = {n_live}

;##### Other possible options (see ultranest package for useage)
; update_interval_iter_fraction, update_interval_ncall
; log_interval, show_status, dKL, frac_remain,
; Lepsilon, min_ess, max_iters, max_ncalls,
; max_num_improvement_loops, 
; cluster_num_live_points
""".format(n_live=n_live, d_logz=d_logz)
        self.configs['sampler']['multinest'] = """\
[sampler]
name = multinest
nprocesses = {n_cpus}
nlivepoints = {n_live}
checkpoint-interval = {ckpt_interval}
evidence-tolerance = {d_logz}
sampling-efficiency = 0.8
importance-nested-sampling = True
""".format(n_cpus=n_cpus,
           n_live=n_live,
           d_logz=d_logz,
           ckpt_interval=ckpt_interval)
        self.configs['sampler']['cpnest'] = """\
[sampler]
;
; WARNING: this sampler requires python3 support
;
name = cpnest
nthreads = {n_cpus}
nlive = {n_live} ;(anything between 1000 (faster) and 2000 (slower), should be good)
maxmcmc = {n_maxmcmc} ;(you should always use >= 5000)
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
""".format(n_cpus=n_cpus, n_live=n_live, n_maxmcmc=n_maxmcmc)

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
        self.configs['inference']['bbh_alignedspin'] = """\
[model]
name = gaussian_noise
low-frequency-cutoff = 20.0

[variable_params]
; waveform parameters that will vary in MCMC
delta_tc =
mass1 =
mass2 =
spin1z =
spin2z =
distance =
coa_phase =
inclination =
polarization =
ra =
dec =

[static_params]
; waveform parameters that will not change in MCMC
approximant = IMRPhenomD
f_lower = 20
f_ref = 20
spin1x = 0
spin1y = 0
spin2x = 0
spin2y = 0
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

[prior-spin1z]
name = uniform
min-spin1z = -0.99
max-spin1z = 0.99

[prior-spin2z]
name = uniform
min-spin2z = -0.99
max-spin2z = 0.99

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
