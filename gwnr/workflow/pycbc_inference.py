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
import logging
import shutil
import subprocess
import numpy

from gwnr.utils import mkdir
from gwnr.analysis import Merger
from gwnr.stats.pycbc_inference_utils import InferenceConfigs
from gwnr.workflow.inference import (OneInferenceAnalysis,
                                     BatchInferenceAnalyses)


class PycbcInferenceInjectionAnalysis(OneInferenceAnalysis):
    def __init__(self,
                 opts,
                 run_dir,
                 config_files,
                 inj_exe_name='inspinj',
                 inf_exe_name='inference',
                 plt_exe_name='plot',
                 verbose=False):
        '''
Setup and run inference on one single injection:
- setup individual analysis dir
- setup all analysis dirs
- start / stop / restart individual analysis
- check status of individual analysis

Parameters
----------
opts : ConfigParser object
    Configuration options
run_dir : string
    Path to directory in which the analysis is to be run
config_files : dict
    Dictionary with names and locations of ini files needed
inj_exe_name : string
    Name of the injection exe's options' section in opts
inf_exe_name : string
    Name of the inference exe's options' section in opts
        '''
        super(PycbcInferenceInjectionAnalysis,
              self).__init__(opts,
                             run_dir,
                             config_files,
                             inj_exe_name=inj_exe_name,
                             inf_exe_name=inf_exe_name,
                             plt_exe_name=plt_exe_name,
                             verbose=verbose)

    def setup(self):
        # Make the analysis directory
        if self.verbose:
            logging.info("Making {0} in {1}".format(self.run_dir, os.getcwd()))
        mkdir(self.run_dir)
        mkdir(os.path.join(self.run_dir, 'scripts'))
        mkdir(os.path.join(self.run_dir, 'log'))
        mkdir(os.path.join(self.run_dir, 'plots'))

        # Copy over the relevant configuration files
        if self.verbose:
            logging.info("Copying config files to {0}".format(self.run_dir))
        for conf_name in self.config_files:
            shutil.copy(
                self.config_files[conf_name],
                os.path.join(self.run_dir, '{0}.ini'.format(conf_name)))

        # Copy over executables
        if self.verbose:
            logging.info("Copying executables to {0}".format(
                os.path.join(self.run_dir, 'scripts/')))
        for _, exe in self.opts.items('executables'):
            shutil.copy(exe, os.path.join(self.run_dir, 'scripts/'))
            os.chmod(
                os.path.join(self.run_dir, 'scripts/', os.path.basename(exe)),
                0o0777)

        # Write injection creation script
        self.opts.set(self.inj_exe_name, 'ninjections', '1')
        self.opts.set(self.inj_exe_name, 'seed',
                      str(numpy.random.randint(1, 1e5)))
        self.write_run_script(
            self.opts.items(self.inj_exe_name), "scripts/{0}".format(
                os.path.basename(
                    self.opts.get('executables', self.inj_exe_name))),
            os.path.join(self.run_dir, "make_injection"))

        # Write pycbc_inference run script
        self.write_run_script(
            self.opts.items(self.inf_exe_name), "scripts/{0}".format(
                os.path.basename(
                    self.opts.get('executables', self.inf_exe_name))),
            os.path.join(self.run_dir, "run_inference"), """#!/bin/bash

# run sampler
# Running with OMP_NUM_THREADS=1 stops lalsimulation
# from spawning multiple jobs that would otherwise be used
# by pycbc_inference and cause a reduced runtime.
OMP_NUM_THREADS=1 \\\n""")

        # Write pycbc_plot_posterior run script
        if self.opts.has_option('executables', self.plt_exe_name):
            from copy import deepcopy
            plt_base_opts = self.opts.items(self.plt_exe_name)
            for ss in self.opts.get_subsections(self.plt_exe_name):
                curr_opts = deepcopy(plt_base_opts)
                curr_opts.extend(self.opts.items(self.plt_exe_name + '-' + ss))
                self.write_run_script(
                    curr_opts, "scripts/{0}".format(
                        os.path.basename(
                            self.opts.get('executables', self.plt_exe_name))),
                    self.get_plt_exe_path() + '_{0}'.format(ss))


class PycbcInferenceOnInjectionBatch(BatchInferenceAnalyses):
    def __init__(self,
                 opts,
                 run_dir,
                 inj_exe_name='inspinj',
                 inf_exe_name='inference',
                 plt_exe_name='plot',
                 verbose=False):
        '''
Thin wrapper class that encapsulates a suite of injection runs

Parameters
----------
opts : workflow.ConfigParser object
    Options for the workflow
run_dir : string
    Path to the main directory where all analyses are to be run
        '''
        super(PycbcInferenceOnInjectionBatch,
              self).__init__(opts,
                             run_dir,
                             inj_exe_name=inj_exe_name,
                             inf_exe_name=inf_exe_name,
                             plt_exe_name=plt_exe_name,
                             verbose=verbose)

        self.num_injections = int(opts.get(self.inj_exe_name, 'ninjections'))
        self.inj_config = opts.get(self.inj_exe_name, 'config-files')

        self.data_configs = opts.get('workflow', 'data').split()
        self.sampler_configs = opts.get('workflow', 'sampler').split()
        self.inf_configs = opts.get('workflow', 'inference').split()

        import itertools
        self.config_combos = list(
            itertools.product(self.data_configs, self.sampler_configs,
                              self.inf_configs))
        for inj_num in range(self.num_injections):
            for configs in self.config_combos:
                confs = {}
                confs['injection'] = self.inj_config
                confs['data'], confs['sampler'], confs['inference'] = configs
                run_tag = self.get_run_tag()
                self.runs[run_tag] = PycbcInferenceInjectionAnalysis(
                    opts,
                    self.name_run_dir(inj_num, confs),
                    confs,
                    inj_exe_name=self.inj_exe_name,
                    inf_exe_name=self.inf_exe_name,
                    plt_exe_name=self.plt_exe_name,
                    verbose=self.verbose)

    def setup_runs(self):
        for r in self.runs:
            self.runs[r].setup()

    def name_run_dir(self, inj_num, configs):
        return 'injection{0:03d}/{1}/{2}/{3}'.format(
            inj_num, configs['data'].split('.ini')[0],
            configs['sampler'].split('.ini')[0],
            configs['inference'].split('.ini')[0])


class PycbcInferenceEventAnalysis(OneInferenceAnalysis):
    def __init__(self,
                 opts,
                 run_dir,
                 data_dir,
                 config_files,
                 event_name,
                 inf_exe_name='inference',
                 plt_exe_name='plot',
                 verbose=False):
        '''
Setup and run inference on one single event:
- setup individual analysis dir
- setup all analysis dirs
- start / stop / restart individual analysis
- check status of individual analysis

Parameters
----------
opts : ConfigParser object
    Configuration options
run_dir : string
    Path to directory in which the analysis is to be run
config_files : dict
    Dictionary with names and locations of ini files needed
inf_exe_name : string
    Name of the inference exe's options' section in opts
        '''
        super(PycbcInferenceEventAnalysis,
              self).__init__(opts,
                             run_dir,
                             config_files,
                             inf_exe_name=inf_exe_name,
                             plt_exe_name=plt_exe_name,
                             verbose=verbose)
        self.data_dir = data_dir

        # Extract useful options now
        if not opts.has_option('workflow', 'data-sample-rate'):
            raise IOError(
                "When analyzing events, provide the data-sample-rate in the workflow section!"
            )

        if not opts.has_option('workflow', 'data-duration'):
            raise IOError(
                "When analyzing events, provide the data-duration in the workflow section!"
            )
        self.data_sample_rate = int(opts.get('workflow', 'data-sample-rate'))
        self.data_duration = int(opts.get('workflow', 'data-duration'))

        self.event_name = event_name
        self.merger = Merger(self.event_name)

        if opts.get('workflow', 'psd-estimation') == 'download':
            self.psd_options = '''\
psd-inverse-length = 8 ; not sure why, but we need this for now!
psd-file ='''
            for ifo in self.merger.operating_ifos():
                self.psd_options += ' {0}:{1}'.format(
                    ifo,
                    os.path.join(
                        os.path.relpath(self.get_data_dir(),
                                        self.get_analysis_dir()),
                        self.merger.psd_file_name(ifo)))
        elif opts.get('workflow', 'psd-estimation') == 'data-standard':
            self.psd_options = '''\
psd-estimation = median-mean
psd-start-time = -256
psd-end-time = 256
psd-inverse-length = 8
psd-segment-length = 8
psd-segment-stride = 4
'''
        else:
            raise IOError('''The option psd-estimation in section workflow
        is required. Currently supported values are:
        download : Download the PSD for event
        data-standard : Use 512 secs around the event with 8s segments to
                        determine PSD''')

    def get_event_name(self):
        return self.event_name

    def get_data_dir(self):
        return self.data_dir

    def fetch_all_data(self, data_dir=None):
        if self.verbose:
            logging.info('Fetching GWOSC frame data')
        if data_dir is None:
            data_dir = self.data_dir
        for ifo in self.merger.operating_ifos():
            self.merger.fetch_data(ifo, self.data_duration,
                                   self.data_sample_rate, data_dir)

    def fetch_all_psds(self, data_dir=None):
        if self.verbose:
            logging.info("Fetching PSD files")
        if data_dir is None:
            data_dir = self.data_dir
        self.merger.fetch_psds(self.data_duration, self.data_sample_rate,
                               data_dir)

    def setup(self):
        # Make the analysis directory
        if self.verbose:
            logging.info("Making {0} in {1}".format(self.run_dir, os.getcwd()))
        mkdir(self.run_dir)
        mkdir(os.path.join(self.run_dir, 'scripts'))
        mkdir(os.path.join(self.run_dir, 'log'))
        mkdir(os.path.join(self.run_dir, 'plots'))

        # Setup formatting options for data.ini for this event
        myargs = {
            'gpstime': self.merger.gpstime(),
            'sample_rate': self.sample_rate,
            'psd_options': self.psd_options
        }

        for ifo in self.merger.operating_ifos():
            myargs['{0}_frame_file'.format(ifo)] = os.path.join(
                os.path.relpath(self.get_data_dir(), self.get_analysis_dir()),
                self.merger.frame_data_name(ifo, self.data_duration,
                                            self.data_sample_rate))
            myargs['{0}_channel'.format(ifo)] = self.merger.channel_name(
                ifo, self.data_sample_rate)

        # Write data.ini configuring options in ConfigWriter.write
        InferenceConfigs(self.run_dir).get_config_writer('data').write(
            self.event_name, **myargs)
        subprocess.call('mv {0}/{1}.ini {0}/data.ini'.format(
            self.run_dir, self.event_name).split())

        # Copy over the relevant configuration files
        if self.verbose:
            logging.info("Copying config files to {0}".format(self.run_dir))
        for conf_name in self.config_files:
            shutil.copy(
                self.config_files[conf_name],
                os.path.join(self.run_dir, '{0}.ini'.format(conf_name)))

        # Copy over executables
        if self.verbose:
            logging.info("Copying executables to {0}".format(
                os.path.join(self.run_dir, 'scripts/')))
        for _, exe in self.opts.items('executables'):
            shutil.copy(exe, os.path.join(self.run_dir, 'scripts/'))
            os.chmod(
                os.path.join(self.run_dir, 'scripts/', os.path.basename(exe)),
                0o0777)

        # Write pycbc_inference run script
        self.write_run_script(self.opts.items(self.inf_exe_name),
                              "scripts/{0}".format(
                                  os.path.basename(
                                      self.opts.get('executables',
                                                    self.inf_exe_name))),
                              os.path.join(self.run_dir, "run_inference"),
                              script_base="""#!/bin/bash

# run sampler
# Running with OMP_NUM_THREADS=1 stops lalsimulation
# from spawning multiple jobs that would otherwise be used
# by pycbc_inference and cause a reduced runtime.
OMP_NUM_THREADS=1 \\\n""")

        # Write pycbc_plot_posterior run script
        if self.opts.has_option('executables', self.plt_exe_name):
            from copy import deepcopy
            plt_base_opts = self.opts.items(self.plt_exe_name)
            for ss in self.opts.get_subsections(self.plt_exe_name):
                curr_opts = deepcopy(plt_base_opts)
                curr_opts.extend(self.opts.items(self.plt_exe_name + '-' + ss))
                self.write_run_script(
                    curr_opts, "scripts/{0}".format(
                        os.path.basename(
                            self.opts.get('executables', self.plt_exe_name))),
                    self.get_plt_exe_path() + '_{0}'.format(ss))


class PycbcInferenceOnEventBatch(BatchInferenceAnalyses):
    def __init__(self,
                 opts,
                 run_dir,
                 inf_exe_name='inference',
                 plt_exe_name='plot',
                 verbose=False):
        '''
Thin wrapper class that encapsulates a suite of event analyses

Parameters
----------
opts : workflow.ConfigParser object
    Options for the workflow
run_dir : string
    Path to the main directory where all analyses are to be run
        '''
        super(PycbcInferenceOnEventBatch,
              self).__init__(opts,
                             run_dir,
                             inf_exe_name=inf_exe_name,
                             plt_exe_name=plt_exe_name,
                             verbose=verbose)
        self.events = opts.get('workflow', 'events').split()

        self.sampler_configs = opts.get('workflow', 'sampler').split()
        self.inf_configs = opts.get('workflow', 'inference').split()

        import itertools
        self.config_combos = list(
            itertools.product(self.sampler_configs, self.inf_configs))
        for event_num, event_name in enumerate(self.events):
            for configs in self.config_combos:
                confs = {}
                #confs['data'] = None
                confs['sampler'], confs['inference'] = configs
                run_tag = self.get_run_tag()
                self.runs[run_tag] = PycbcInferenceEventAnalysis(
                    opts,
                    self.name_run_dir(event_name, confs),
                    self.name_data_dir(event_name),
                    confs,
                    event_name,
                    inf_exe_name=self.inf_exe_name,
                    plt_exe_name=self.plt_exe_name,
                    verbose=self.verbose)

    def setup_runs(self):
        for r in self.runs:
            self.runs[r].setup()
            self.runs[r].fetch_all_data()

    def name_run_dir(self, event_name, configs):
        return 'event{0}/{1}/{2}'.format(event_name,
                                         configs['sampler'].split('.ini')[0],
                                         configs['inference'].split('.ini')[0])

    def name_data_dir(self, event_name):
        return 'event{0}/data/'.format(event_name)
