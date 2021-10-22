# Copyright (C) 2021 Prayush Kumar
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
import subprocess
import h5py

from gwnr.utils import mkdir
from gwnr.analysis import Merger
from gwnr.stats.bilby_utils import (BilbyScriptWriterEvent,
                                    BilbyScriptWriterInjection)
from gwnr.workflow.inference import (OneInferenceAnalysis,
                                     BatchInferenceAnalyses)
from gwnr.workflow.utils import WorkflowConfigParserBase


class BilbyInferenceConfigParser(WorkflowConfigParserBase):
    def __init__(self, *args, **kwargs) -> None:
        super(BilbyInferenceConfigParser, self).__init__(*args, **kwargs)

    def get_section_opts(self, sec, val_map=float):
        if sec not in self:
            return {}
        out = {}
        for k in self[sec]:
            try:
                v = val_map(self[sec][k])
            except:
                v = self[sec][k]
            out[k] = v
        return out

    def get_inference_opts(self):
        sec_conf = self.get_section_opts('inference')
        out = {}
        for p in ['duration', 'sample_rate', 'lower_frequency_cutoff']:
            out[p] = float(sec_conf[p])
        for p in [
                'phase_marginalization', 'distance_marginalization',
                'time_marginalization'
        ]:
            out[p] = (p in sec_conf)
        for p in ['upper_frequency_cutoff', 'reference_frequency']:
            if p in sec_conf:
                out[p] = float(sec_conf[p])
        if 'reference_frequency' not in out:
            out['reference_frequency'] = out['lower_frequency_cutoff']
        if 'upper_frequency_cutoff' not in out:
            out['upper_frequency_cutoff'] = out['sample_rate'] // 2
        return out

    def get_interferometer_list(self):
        return self.get_section_opts('data')['interferometers'].split(',')

    def get_source_type(self):
        return self.get_section_opts('data')['source_type']

    def get_prior_lines(self):
        sec_conf = self.get_section_opts('prior')
        return [k + ' = ' + sec_conf[k] for k in sec_conf]

    def get_injection_params(self, subsection='static_params'):
        if 'injection' in self:
            if subsection in self.get_subsections('injection'):
                return dict(self['injection-{}'.format(subsection)])
        return {}

    def get_injections_config(self):
        def get_injection_approximant(conf):
            return conf['injection']['approximant']

        injection_config_lines = []
        for subsec_name in self.get_subsections('injection'):
            # replace `prior_` with `prior-``
            injection_config_lines.append('[{}]'.format(
                subsec_name.replace('prior_', 'prior-')))
            if 'static_params' in subsec_name:
                injection_config_lines.append('approximant = {}'.format(
                    get_injection_approximant(self)))

            sec_name = 'injection-{}'.format(subsec_name)
            for f in self[sec_name]:
                injection_config_lines.append('{} = {}'.format(
                    f, self[sec_name][f]))

            injection_config_lines.append('')
        return injection_config_lines

    def write_injections_config(self, fname='injection.ini'):
        with open(fname, 'w') as f:
            lines = self.get_injections_config()
            for line in lines:
                f.write(line + "\n")


class BilbyInferenceInjectionAnalysis(OneInferenceAnalysis):
    def __init__(self,
                 opts,
                 script_writer,
                 run_dir,
                 config_files,
                 verbose=False):
        '''
Setup and run inference on one single injection:
- setup individual analysis dir
- setup all analysis dirs
- start / stop / restart individual analysis
- check status of individual analysis

Parameters
----------
opts : WorkflowConfigParserBase object
    Configuration options
run_dir : string
    Path to directory in which the analysis is to be run
config_files : dict
    Dictionary with names and locations of ini files needed
        '''
        self.script_writer_obj = script_writer
        super(BilbyInferenceInjectionAnalysis, self).__init__(opts,
                                                              run_dir,
                                                              config_files,
                                                              inj_exe_name='',
                                                              inf_exe_name='',
                                                              plt_exe_name='',
                                                              verbose=verbose)

    def write_prior(self):
        script_name = self.get_inf_exe_path()
        temp = self.script_writer_obj.priors_filename
        self.script_writer_obj.priors_filename = os.path.join(
            os.path.dirname(script_name),
            os.path.basename(self.script_writer_obj.priors_filename))
        self.script_writer_obj.write_prior_file()
        self.script_writer_obj.priors_filename = temp

    def write_run_script(self):
        script_name = self.get_inf_exe_path()
        self.script_writer_obj.write_script(script_name)
        os.chmod(script_name, 0o0777)

    def setup(self):
        # Make the analysis directory
        if self.verbose:
            logging.info("Making {0} in {1}".format(self.get_analysis_dir(),
                                                    os.getcwd()))
        mkdir(self.get_analysis_dir())
        mkdir(os.path.join(self.get_analysis_dir(), 'log'))

        # Write bilby priors file
        self.write_prior()

        # Write bilby run script
        self.write_run_script()


class BilbyInferenceEventAnalysis(OneInferenceAnalysis):
    def __init__(self,
                 opts,
                 script_writer,
                 run_dir,
                 config_files,
                 verbose=False):
        '''
Setup and run inference on one single GW event using public data:
- setup individual analysis dir
- setup all analysis dirs
- start / stop / restart individual analysis
- check status of individual analysis

Parameters
----------
opts : WorkflowConfigParserBase object
    Configuration options
run_dir : string
    Path to directory in which the analysis is to be run
config_files : dict
    Dictionary with names and locations of ini files needed
        '''
        self.script_writer_obj = script_writer
        super(BilbyInferenceEventAnalysis, self).__init__(opts,
                                                          run_dir,
                                                          config_files,
                                                          inj_exe_name='',
                                                          inf_exe_name='',
                                                          plt_exe_name='',
                                                          verbose=verbose)

    def write_prior(self):
        script_name = self.get_inf_exe_path()
        temp = self.script_writer_obj.priors_filename
        self.script_writer_obj.priors_filename = os.path.join(
            os.path.dirname(script_name),
            os.path.basename(self.script_writer_obj.priors_filename))
        self.script_writer_obj.write_prior_file()
        self.script_writer_obj.priors_filename = temp

    def write_run_script(self):
        script_name = self.get_inf_exe_path()
        self.script_writer_obj.write_script(script_name)
        os.chmod(script_name, 0o0777)

    def setup(self):
        # Make the analysis directory
        if self.verbose:
            logging.info("Making {0} in {1}".format(self.get_analysis_dir(),
                                                    os.getcwd()))
        mkdir(self.get_analysis_dir())
        mkdir(os.path.join(self.get_analysis_dir(), 'log'))

        # Write bilby priors file
        self.write_prior()

        # Write bilby run script
        self.write_run_script()


class BilbyOnInjectionBatch(BatchInferenceAnalyses):
    def __init__(self, opts, run_dir, config_files={}, verbose=False):
        '''
Thin wrapper class that encapsulates a suite of injection runs

Parameters
----------
opts : WorkflowConfigParserBase object
    Options for the workflow
run_dir : string
    Path to the main directory where all analyses are to be run
        '''
        super(BilbyOnInjectionBatch, self).__init__(opts,
                                                    run_dir,
                                                    inj_exe_name='',
                                                    inf_exe_name='',
                                                    plt_exe_name='',
                                                    verbose=verbose)

        logging.info("--- verifying injection params config")
        static_inj_params = opts.get_injection_params('static_params')
        var_inj_params = opts.get_injection_params('variable_params')

        for p in var_inj_params:
            if p in static_inj_params:
                raise IOError(
                    "Inconsistent set of injection params specified. Check {}".
                    format(p))
        logging.info("--- injection params config verified")

        self.num_injections = opts.get_section_opts('injection',
                                                    int)['num_injections']

        logging.info("--- sampling {} injection params".format(
            self.num_injections))
        opts.write_injections_config()
        subprocess.check_output("rm -f injection.hdf", shell=True)
        subprocess.check_output(
            "pycbc_create_injections --config-files injection.ini --ninjections {} --output-file injection.hdf --force"
            .format(self.num_injections),
            shell=True)
        logging.info("--- injection params sampled")

        logging.info("--- reading injection params")
        inj_params = []
        with h5py.File('injection.hdf') as injs:
            for i in range(self.num_injections):
                inj_p = static_inj_params
                for p in var_inj_params:
                    inj_p[p] = injs[p][()][i]
                # Make actual copies of the parameters, instead of passing pointers around
                inj_params.append({k: inj_p[k] for k in inj_p})
        logging.info("--- {} injection params read".format(len(inj_params)))

        logging.info("--- creating script writer objects for injections")
        template_opts = opts.get_section_opts('template')
        injection_opts = opts.get_section_opts('injection')
        for p in ['source_model', 'approximant']:
            if p not in injection_opts:
                logging.info(
                    "----- borrowing {} for injections from [template]".format(
                        p))
                injection_opts[p] = template_opts[p]
            if p not in template_opts:
                logging.info(
                    "--- borrowing {} for templates from [injection]".format(
                        p))
                template_opts[p] = injection_opts[p]

        objs = []
        for i in range(self.num_injections):
            # Make a copy instead of moving points
            inj_opts = {k: injection_opts[k] for k in injection_opts}
            # Assign injection parameters
            inj_opts['parameters'] = inj_params[i]

            # Create script writer
            obj = BilbyScriptWriterInjection(
                'injection_{0:03d}'.format(i),
                opts.get_source_type(),
                injection_opts=inj_opts,
                interferometer_list=opts.get_interferometer_list(),
                inference_opts=opts.get_inference_opts(),
                template_opts=template_opts,
                sampler_opts=opts.get_section_opts('sampler', int),
                priors=opts.get_prior_lines(),
                priors_file="priors.prior",
                verbosity=1)
            logging.info(
                "--- script writer object created for injection {}".format(i))

            run_tag = self.get_run_tag()
            self.runs[run_tag] = BilbyInferenceInjectionAnalysis(
                opts,
                obj,
                self.name_run_dir(i),
                config_files,
                verbose=self.verbose)
            logging.info(
                "--- analysis objects created for injection {}".format(i))

            objs.append(obj)
        self.script_writers = objs

    def setup_runs(self):
        for r in self.runs:
            self.runs[r].setup()

    def name_run_dir(self, inj_num):
        return os.path.join(self.get_analyses_dir(),
                            'injection{0:03d}'.format(inj_num))


class BilbyOnEventBatch(BatchInferenceAnalyses):
    def __init__(self, opts, run_dir, config_files={}, verbose=False):
        '''
Thin wrapper class that encapsulates a suite of event runs

Parameters
----------
opts : WorkflowConfigParserBase object
    Options for the workflow
run_dir : string
    Path to the main directory where all analyses are to be run
        '''
        super(BilbyOnEventBatch, self).__init__(opts,
                                                run_dir,
                                                inj_exe_name='',
                                                inf_exe_name='',
                                                plt_exe_name='',
                                                verbose=verbose)
        self.event_names = opts.get_section_opts('data')['event_names'].split(
            ',')

        logging.info("--- creating script writer objects for events")
        template_opts = opts.get_section_opts('template')

        objs = []
        for i, event_name in enumerate(self.event_names):

            # Create script writer
            obj = BilbyScriptWriterEvent(
                event_name,
                opts.get_source_type(),
                interferometer_list=opts.get_interferometer_list(),
                inference_opts=opts.get_inference_opts(),
                template_opts=template_opts,
                sampler_opts=opts.get_section_opts('sampler', int),
                priors=opts.get_prior_lines(),
                priors_file="priors.prior",
                verbosity=1)
            logging.info(
                "--- script writer object created for event {}".format(
                    event_name))

            run_tag = self.get_run_tag()
            self.runs[run_tag] = BilbyInferenceEventAnalysis(
                opts,
                obj,
                self.name_run_dir(event_name),
                config_files,
                verbose=self.verbose)
            logging.info("--- analysis objects created for event {}".format(i))

            objs.append(obj)
        self.script_writers = objs

    def setup_runs(self):
        for r in self.runs:
            self.runs[r].setup()

    def name_run_dir(self, event_name):
        return os.path.join(self.get_analyses_dir(), event_name)
