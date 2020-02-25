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
import logging
import shutil
import subprocess
import numpy

from GWNRTools.DataAnalysis.MiscFunctions import get_unique_hex_tag


def get_ini_opts(confs, section):
    op_str = ""
    for opt in confs.options(section):
        val = confs.get(section, opt)
        op_str += "--" + opt + " " + val + " \\" + "\n"
    return op_str


def mkdir(dir_name):
    try:
        subprocess.call(["mkdir", "-p", dir_name])
    except OSError:
        pass

####
# **`InjectionInferenceAnalysis`**:
# - setup individual analysis dir
# - setup all analysis dirs
# - start / stop / restart individual analysis
# - check status of individual analysis


class InjectionInferenceAnalysis():
    def __init__(self, opts, run_dir, config_files,
                 inj_exe_name='inspinj', inf_exe_name='inference',
                 plt_exe_name='plot',
                 verbose=False):
        '''
        Setup and run inference on one single injection

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
        self.verbose = verbose
        self.opts = opts
        self.run_dir = run_dir
        self.config_files = config_files
        self.inj_exe_name = inj_exe_name
        self.inf_exe_name = inf_exe_name
        self.plt_exe_name = plt_exe_name

    def get_run_dir(self): return self.run_dir

    def get_opts(self): return self.opts

    def get_config_files(self): return self.config_files

    def get_inj_exe_name(self): return self.inj_exe_name

    def get_inj_exe_path(self):
        return os.path.join(self.run_dir, "make_injection")

    def get_inf_exe_name(self): return self.inf_exe_name

    def get_inf_exe_path(self):
        return os.path.join(self.run_dir, "run_inference")

    def get_plt_exe_path(self):
        return os.path.join(self.run_dir, "make_plot")

    def get_log_dir(self): return os.path.join(self.run_dir, 'log')

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
            shutil.copy(self.config_files[conf_name],
                        os.path.join(self.run_dir, '{0}.ini'.format(conf_name)))

        # Copy over executables
        if self.verbose:
            logging.info("Copying executables to {0}".format(
                os.path.join(self.run_dir, 'scripts/')))
        for _, exe in self.opts.items('executables'):
            shutil.copy(exe, os.path.join(self.run_dir, 'scripts/'))
            os.chmod(os.path.join(self.run_dir, 'scripts/',
                                  os.path.basename(exe)), 0o0777)

        # Write injection creation script
        self.opts.set(self.inj_exe_name, 'ninjections', '1')
        self.opts.set(self.inj_exe_name, 'seed',
                      str(numpy.random.randint(1, 1e5)))
        self.write_run_script(self.opts.items(self.inj_exe_name),
                              "scripts/{0}".format(os.path.basename(
                                  self.opts.get('executables', self.inj_exe_name))),
                              os.path.join(
                                  self.run_dir, "make_injection"))

        # Write pycbc_inference run script
        self.write_run_script(self.opts.items(self.inf_exe_name),
                              "scripts/{0}".format(os.path.basename(
                                  self.opts.get('executables', self.inf_exe_name))),
                              os.path.join(
                                  self.run_dir, "run_inference"),
                              """#!/bin/bash

# run sampler
# Running with OMP_NUM_THREADS=1 stops lalsimulation
# from spawning multiple jobs that would otherwise be used
# by pycbc_inference and cause a reduced runtime.
OMP_NUM_THREADS=1 \\\n""")

        # Write pycbc_plot_posterior run script
        if self.opts.has_option('executables', self.plt_exe_name):
            self.write_run_script(self.opts.items(self.plt_exe_name),
                                  "scripts/{0}".format(os.path.basename(
                                      self.opts.get('executables', self.plt_exe_name))),
                                  os.path.join(self.run_dir, "make_plot"))

    def write_run_script(self, exe_opts, exe_path, script_name,
                         script_base="""#!/bin/bash\n"""):
        out_str = script_base
        out_str += "{0} \\\n".format(exe_path)
        for exe_opt_name, exe_opt in exe_opts:
            out_str += "  --" + exe_opt_name + " " + exe_opt + " \\\n"
        with open(script_name, "w") as fout:
            fout.write(out_str)
        os.chmod(script_name, 0o0777)
####

####


class InferenceOnInjectionBatch():
    def __init__(self, opts, run_dir, inj_exe_name='inspinj', inf_exe_name='inference',
                 plt_exe_name='plot', verbose=False):
        '''
        Thin wrapper class that encapsulates a suite of injection runs

        Parameters
        ----------
        opts : workflow.ConfigParser object
            Options for the workflow
        run_dir : string
            Path to the main directory where all analyses are to be run
        '''
        self.verbose = verbose
        self.opts = opts
        self.inj_exe_name = inj_exe_name
        self.inf_exe_name = inf_exe_name
        self.plt_exe_name = plt_exe_name

        self.num_injections = int(opts.get(self.inj_exe_name, 'ninjections'))
        self.inj_config = opts.get(self.inj_exe_name, 'config-files')

        self.data_configs = opts.get('workflow', 'data').split()
        self.sampler_configs = opts.get('workflow', 'sampler').split()
        self.inf_configs = opts.get('workflow', 'inference').split()

        import itertools
        self.config_combos = list(itertools.product(self.data_configs,
                                                    self.sampler_configs,
                                                    self.inf_configs))
        self.runs = {}
        for inj_num in range(self.num_injections):
            for configs in self.config_combos:
                confs = {}
                confs['injection'] = self.inj_config
                confs['data'], confs['sampler'], confs['inference'] = configs
                run_tag = self.get_run_tag()
                self.runs[run_tag] = InjectionInferenceAnalysis(
                    opts, self.name_run_dir(inj_num, confs), confs,
                    inj_exe_name=self.inj_exe_name,
                    inf_exe_name=self.inf_exe_name,
                    plt_exe_name=self.plt_exe_name,
                    verbose=self.verbose)

    def get_opts(self): return self.opts

    def get_run_dir(self): return self.run_dir

    def setup_runs(self):
        for r in self.runs:
            self.runs[r].setup()

    def get_runs(self): return self.runs

    def get_run_tag(self): return get_unique_hex_tag()

    def name_run_dir(self, inj_num, configs):
        return 'injection{0:03d}/{1}/{2}/{3}'.format(
            inj_num,
            configs['data'].split('.ini')[0],
            configs['sampler'].split('.ini')[0],
            configs['inference'].split('.ini')[0])
####
