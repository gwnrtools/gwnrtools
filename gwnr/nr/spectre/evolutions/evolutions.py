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
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
from __future__ import (absolute_import, print_function)

import os
import logging
import subprocess
import numpy
from .configurations import cluster_submission_file


class BatchEvolutions(object):
    def __init__(self,
                 exes,
                 inputs,
                 tests,
                 test_dir,
                 reduction_data_file_name='PlaneWave2DPeriodicReductions.h5',
                 volume_data_file_name='PlaneWave2DPeriodicVolume0.h5',
                 xdmf_converter=None):
        '''
Input:
------
exes     : dict, executables for all executable configs
inputs   : dict, text data for all input files
tests    : list, analysis tags for all test configs
test_dir : str, Base directory within which to run all tests

Functions:
----------

1) Setup tests
2) Run tests
3) Visualize output (**development**)
        '''
        self.exes = exes
        self.tests = tests
        self.input_files = inputs
        self.test_dirs = {x: os.path.join(test_dir, x) for x in self.tests}
        self.output_files = {
            'reduction': reduction_data_file_name,
            'volume': volume_data_file_name
        }
        self.xdmf_converter = xdmf_converter

    def available_tests(self):
        '''
Names of planned tests
        '''
        return self.tests

    def check_exes(self, tests=None):
        '''
Check whether executables for all tests planned actually exist.
Returns True if they do for all tests.
        '''
        if tests is None:
            tests = self.tests
        for test in tests:
            config_name, test_name = test.split('/')
            if config_name in self.exes.keys() and \
                    os.path.exists(self.exes[config_name]) and \
                    os.path.getsize(self.exes[config_name]) > 0:
                logging.info("Exec for test {0:s}: Found".format(config_name))
            else:
                logging.info("Exec for test {0:s}: Not Found at {1}".format(
                    config_name, self.exes[config_name]))

    def exe(self, config_name):
        '''
Full path to executable file for a given test
        '''
        assert config_name in self.exes.keys(),\
            "Config name {0} not defined above...".format(config_name)
        return self.exes[config_name]

    def exe_name(self, config_name):
        '''
Full path to executable file for a given test
        '''
        return self.exe(config_name).split('/')[-1]

    def run_dir(self, test):
        '''
Full path to cluster directory to run the given test
        '''
        assert test in list(self.test_dirs.keys()),\
            "Test name {0} not defined above...".format(test)
        return self.test_dirs[test]

    def input_file_name(self, test):
        '''
Name oft input file for a given test
        '''
        return "TestInput.yaml"

    def input_file(self, test):
        '''
Full path to test input file for a given test
        '''
        return os.path.join(self.run_dir(test), "TestInput.yaml")

    def cluster_submission_file(self, test, cluster):
        '''
Full path to cluster submission file for a given test
        '''
        return os.path.join(self.run_dir(test), "{0:s}.sh".format(cluster))

    def setup_run(self,
                  test,
                  cluster='local',
                  compiler='gcc',
                  spectre_root=None,
                  symlink_exe=True):
        '''
Setup a single test run:

 - Setup on either a computing cluster or locally;
 - If on computing cluster, additional inputs are required;

        '''
        self.check_exes([test])
        config_name, test_name = test.split('/')
        assert test_name in self.input_files,\
            "Input file for test {} not found in {}".format(
                test_name, self.input_files.keys())
        input_file = self.input_file(test)
        exe = self.exe(config_name)
        run_dir = self.run_dir(test)
        exe_dest = os.path.join(run_dir, os.path.split(exe)[-1])

        # Make directory
        logging.info("Making run dir: {0:s}".format(run_dir))
        subprocess.call(['mkdir', '-p', run_dir])

        # Copy the executable over
        logging.info("Copying over {0:s}".format(exe))
        if not os.path.exists(exe_dest):
            if symlink_exe:
                os.symlink(exe, exe_dest)
                logging.info("..exe linked to {0:s}".format(exe_dest))
            else:
                subprocess.call(['cp', '-v', exe, exe_dest])
                logging.info("..exe copied to {0:s}".format(exe_dest))
            # Make it executable
            subprocess.call(['chmod', '+x', exe_dest])

        # Write appropriate input file
        in_file_name = self.input_file(test)
        logging.info("Writing input file: {0:s}".format(in_file_name))
        with open(in_file_name, 'w') as fout:
            fout.write(self.input_files[test_name])

        # Write appropriate submission file if needed
        if cluster != 'local':
            with open(self.cluster_submission_file(test, cluster),
                      'w') as fout:
                fout.write(
                    cluster_submission_file(
                        cluster=cluster,
                        spectre_root=spectre_root,
                        compiler=compiler,
                        run_dir=run_dir,
                        input_file=self.input_file_name(test),
                        exe=self.exe_name(config_name),
                        tag=test))
        else:
            pass
        return exe_dest, run_dir

    def submit_to_cluster(self, test, cluster):
        '''
Function to submit a prepared test

WARNING: To be run on the cluster only!
        '''
        # Move to run directory
        os.chdir(self.run_dir(test))

        # Submit
        return subprocess.check_output([
            'sbatch',
            os.path.split(self.cluster_submission_file(test, cluster))[-1],
            '> sub.out'
        ],
                                       shell=True)

    def run(self, test, setup=False, ncores=4, cluster='local'):
        '''
Run a single test:

 - Either on a computing cluster or locally;
 - If on computing cluster, this function submits the test to the queue;
 - If locally, this function actually **runs** the executable (can be slow!)
'''
        if setup:
            exe, run_dir = self.setup_run(test)
        else:
            config_name, test_name = test.split('/')
            exe = self.exe(config_name)
            run_dir = self.run_dir(test)
            exe = os.path.join(run_dir, os.path.split(exe)[-1])

        # move to analysis directory
        os.chdir(run_dir)

        if cluster == 'local':
            if self.check_output(test):
                logging.warn("Output exists for this test, not overwriting")
                return
            exe = os.path.split(exe)[-1]
            input_file = os.path.split(self.input_file(test))[-1]

            # Run SpECTRE here
            logging.info(
                "Executing ./{0:s} ++ppn {1:d} --input-file {2:s}".format(
                    exe, ncores, input_file))
            return subprocess.check_output([
                './{0:s} ++ppn {1:d} --input-file {2:s}'.format(
                    exe, ncores, input_file)
            ],
                                           shell=True)
        else:
            self.submit_to_cluster(self, test, cluster)

    def output_file(self, test, which='reduction'):
        '''
Full path to reduction or volume observer file for a given test
        '''
        return os.path.join(self.run_dir(test), self.output_files[which])

    def check_output(self, test, which=['volume', 'reduction']):
        '''
Verify if reduction and / or volume observer outputs have been
written for a given test
        '''
        run_dir = self.run_dir(test)
        out_found = []
        for output in which:
            out_file = self.output_file(test, output)
            logging.info("Checking for {0:s} data...: {1:s}".format(
                output, out_file))
            if os.path.exists(out_file):
                size = os.path.getsize(out_file) / (1024 * 1024)
                unit = 'M'
                if size < 1:
                    size = os.path.getsize(out_file) / (1024 * 1)
                    unit = 'k'
                logging.info("...Found with size {0:.2f}{1:s}".format(
                    size, unit))
                out_found.append(True)
            else:
                logging.info("...Not Written.")
                out_found.append(False)
        return numpy.any(out_found)

    def read_output_file(self, test, which='reduction'):
        '''
H5Py File pointer to reduction or volume observer output
        '''
        import h5py
        out_file = self.output_file(test, which)
        if self.check_output(test, [which]):
            return h5py.File(out_file, "r")
        return None

    def convert_volume_output_to_xdmf(self, test, conversion_bin=None):
        if conversion_bin and os.path.exists(conversion_bin):
            exe = conversion_bin
        else:
            exe = self.xdmf_converter
        assert os.path.exists(exe) and os.path.getsize(exe) > 0,\
            "Xdmf converter utility {} not found!".format(exe)

        from gwnr.nr.spectre.evolutions import HandleSpectreVolumeDatum
        out_file = self.output_file(test, which='volume')
        out_handler = HandleSpectreVolumeDatum(volume_data_file=out_file,
                                               name='TMP',
                                               read_fields=[],
                                               xdmf_converter=exe)
        out_handler.convert_to_xdmf()
