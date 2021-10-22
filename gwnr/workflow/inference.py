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

from gwnr.utils import get_unique_hex_tag


def get_ini_opts(confs, section):
    op_str = ""
    for opt in confs.options(section):
        val = confs.get(section, opt)
        op_str += "--" + opt + " " + val + " \\" + "\n"
    return op_str


class OneInferenceAnalysis(object):
    def __init__(self,
                 opts,
                 run_dir,
                 config_files,
                 inj_exe_name=None,
                 inf_exe_name=None,
                 plt_exe_name=None,
                 verbose=False):
        '''
Base class for inference on one single source

Use `PycbcInferenceInjectionAnalysis` or `PycbcInferenceEventAnalysis` instead
        '''
        self.verbose = verbose
        self.opts = opts

        # Extract useful options now
        if opts.has_option('workflow', 'sample-rate'):
            self.sample_rate = int(opts.get('workflow', 'sample-rate'))
        else:
            self.sample_rate = 2048

        self.run_dir = run_dir
        self.config_files = config_files
        self.inj_exe_name = inj_exe_name
        self.inf_exe_name = inf_exe_name
        self.plt_exe_name = plt_exe_name

    def get_analysis_dir(self):
        return self.run_dir

    def get_opts(self):
        return self.opts

    def get_config_files(self):
        return self.config_files

    def get_inj_exe_name(self):
        return self.inj_exe_name

    def get_inj_exe_path(self):
        return os.path.join(self.run_dir, "make_injection")

    def get_inf_exe_name(self):
        return self.inf_exe_name

    def get_inf_exe_path(self):
        return os.path.join(self.run_dir, "run_inference")

    def get_plt_exe_name(self):
        return self.plt_exe_name

    def get_plt_exe_path(self):
        return os.path.join(self.run_dir, "make_plot")

    def get_log_dir(self):
        return os.path.join(self.run_dir, 'log')

    def setup(self):
        raise NotImplementedError()

    def write_run_script(self,
                         exe_opts,
                         exe_path,
                         script_name,
                         script_base="""#!/bin/bash\n"""):
        out_str = script_base
        out_str += "{0} \\\n".format(exe_path)
        for exe_opt_name, exe_opt in exe_opts:
            out_str += "  --" + exe_opt_name + " " + exe_opt + " \\\n"
        with open(script_name, "w") as fout:
            fout.write(out_str)
        os.chmod(script_name, 0o0777)


class BatchInferenceAnalyses(object):
    def __init__(self,
                 opts,
                 run_dir,
                 inj_exe_name=None,
                 inf_exe_name=None,
                 plt_exe_name=None,
                 verbose=False):
        '''
Base class for inference on a batch of sources

Use `PycbcInferenceOnInjectionBatch` or `PycbcInferenceOnEventBatch` instead
        '''
        self.verbose = verbose
        self.opts = opts
        self.run_dir = run_dir
        self.inj_exe_name = inj_exe_name
        self.inf_exe_name = inf_exe_name
        self.plt_exe_name = plt_exe_name
        self.runs = {}

    def get_opts(self):
        return self.opts

    def get_analyses_dir(self):
        return self.run_dir

    def setup_runs(self):
        raise NotImplementedError()

    def get_runs(self):
        return self.runs

    def get_run_tag(self):
        return get_unique_hex_tag()

    def name_run_dir(self, inj_num, configs):
        raise NotImplementedError()
