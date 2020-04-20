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


class BatchEvolutions(object):
    def __init__(self, exes, tests, test_dir,
                reduction_data_file_name='PlaneWave2DPeriodicReductions.h5',
                volume_data_file_name='PlaneWave2DPeriodicVolume0.h5'):
        '''
Input:
------
exes     : dict, executables for all executable configs
tests    : list, analysis tags for all test configs
test_dir : str, Base directory within which to run all tests

Functions:
----------

1) Setup tests
2) Run tests
3) Visualize output (**development**)
        '''
        self.exes     = exes
        self.tests    = tests
        self.test_dirs = {x:os.path.join(test_dir, x) for x in self.tests}
        self.output_files = {
            'reduction' : reduction_data_file_name,
            'volume'    : volume_data_file_name
        }
    
    def available_tests(self):
        '''
Names of planned tests
        '''
        return self.tests
    
    def check_exes(self):
        '''
Check whether executables for all tests planned actually exist. 
Returns True if they do for all tests.
        '''
        for test in self.tests:
            config_name, test_name = test.split('/')
            assert(config_name in self.exes.keys(),
                  "Config name {0} not defined above...".format(config_name));
            logging.info("Exec for test {0:s}: Found".format(config_name))
            
    def exe(self, config_name):
        '''
Full path to executable file for a given test
        '''
        assert(config_name in self.exes.keys(),
              "Config name {0} not defined above...".format(config_name));
        return self.exes[config_name]
    
    def run_dir(self, test):
        '''
Full path to cluster directory to run the given test
        '''
        assert(test in self.test_dirs.keys(),
               "Test name {0} not defined above...".format(test));
        return self.test_dirs[test]
    
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
    
    def setup_run(self, test, cluster='local'):
        '''
Setup a single test run:

 - Setup on either a computing cluster or locally;
 - If on computing cluster, additional inputs are required;
 - TODO: Add option to symlink exec instead of copying it over;
        '''
        config_name, test_name = test.split('/')
        exe        = self.exe(config_name)
        input_file = self.input_file(test)
        run_dir    = self.run_dir(test)
        
        # Make directory
        logging.info("Making run dir: {0:s}".format(run_dir))
        subprocess.call(['mkdir', '-p', run_dir])
        
        # Copy the executable over
        logging.info("Copying over {0:s}".format(exe))
        if not os.path.exists(exe):
            subprocess.call(['cp', '-v', exe, run_dir])
            # Make it executable
            subprocess.call(['chmod', '+x', exe])
            logging.info("..exe copied to {0:s}".format(exe))
        
        # Write appropriate input file
        in_file_name = self.input_file(test)
        logging.info("Writing input file: {0:s}".format(in_file_name))
        with open(in_file_name, 'w') as fout:
            fout.write(input_files[test_name])
        
        # Write appropriate submission file if needed
        if cluster != 'local':
            sub_file_name = self.cluster_submission_file(test, cluster)
            fmts = cluster_submission_files_formatting[cluster]
            opts = []
            with open(sub_file_name, 'w') as fout:
                for fmt in fmts:
                    if 'EXE' in fmt:
                        opts.append(os.path.split(exe)[-1])
                        continue
                    if 'RUN_DIR' in fmt:
                        opts.append(run_dir)
                        continue
                    if 'INPUT' in fmt:
                        opts.append(os.path.split(input_file)[-1])
                        continue
                    if 'OUTPUT_PREFIX' in fmt:
                        opts.append(self.output_files['volume'])
                fout.write(cluster_submission_files[cluster].format(*fmts))
        else:
            pass
        return exe, run_dir
    
    def submit_to_cluster(self, test, cluster):
        '''
Function to submit a prepared test
        '''
        # Move to run directory
        os.chdir(self.run_dir(test))
        # Submit
        return subprocess.check_output([
                'sbatch',
                    os.path.split(self.cluster_submission_file(test, cluster))[-1],
                        '> sub.out'
            ], shell=True)
        
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
        
        
        # move to analysis directory
        os.chdir(run_dir)
        
        if cluster != 'local':
            exe = os.path.split(exe)[-1]
            input_file   = os.path.split(self.input_file(test))[-1]
            
            # Run SpECTRE here
            subprocess.check_output([
                './' + exe,
                '++ppn', '{0:d}'.format(ncores),
                '--input-file', input_file
            ], shell=True)
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
            logging.info("Checking for {0:s} data...".format(output))
            out_file = self.output_file(test, output)
            if os.path.exists(out_file):
                size = os.path.getsize(out_file) / (1024 * 1024)
                unit = 'M'
                if size < 1:
                    size = os.path.getsize(out_file) / (1024 * 1)
                    unit = 'k'
                logging.info("...Found with size {0:.2f}{1:s}".format(size, unit))
                out_found.append(True)
            else:
                logging.info("...Not Written.")
                out_found.append(False)
        return np.all(out_found)
    
    def read_output_file(self, test, which='reduction'):
        '''
H5Py File pointer to reduction or volume observer output
        '''
        out_file = self.output_file(test, which)
        if self.check_output(test, [which]):
            return h5py.File(out_file, "r")
        return None
