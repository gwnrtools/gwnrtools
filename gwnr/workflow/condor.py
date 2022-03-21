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
from glue.pipeline import CondorDAGJob, CondorDAGNode, CondorJob


class BaseJob(CondorDAGJob, CondorJob):
    def __init__(self,
                 log_dir,
                 executable,
                 cp,
                 section,
                 gpu=False,
                 accounting_group=None,
                 request_memory=None):
        CondorDAGJob.__init__(self, "vanilla", executable)

        if gpu:
            CondorJob.__init__(self, "vanilla", executable, 2)
        # These are all python jobs so need to pull in the env
        self.add_condor_cmd('getenv', 'True')
        log_base = os.path.join(
            log_dir,
            os.path.basename(executable) + '-$(cluster)-$(process)')
        self.set_stderr_file(log_base + '.err')
        self.set_stdout_file(log_base + '.out')
        self.set_sub_file(os.path.basename(executable) + '.sub')

        if cp is not None:
            self.add_ini_opts(cp, section)

        if accounting_group:
            self.add_condor_cmd('accounting_group', accounting_group)

        if request_memory:
            self.add_condor_cmd('RequestMemory', request_memory)


class BanksimNode(CondorDAGNode):
    def __init__(self,
                 job,
                 inj_file,
                 tmplt_file,
                 match_file,
                 gpu=True,
                 gpu_postscript=False,
                 inj_per_job=None):
        CondorDAGNode.__init__(self, job)

        self.add_file_opt("signal-file", inj_file)
        self.add_file_opt("template-file", tmplt_file)

        if gpu:
            self.add_var_opt("processing-scheme", 'cuda')

        if gpu and gpu_postscript:
            self.set_retry(5)
            mf = match_file + ".$(Process)"
            mf1 = match_file + ".0"
            mf2 = match_file + ".1"
            self.add_file_opt("match-file",
                              match_file + ".$(Process)",
                              file_is_output_file=True)
            self.job().__queue = 2

            # Needed to satisfy the requirements for both running on atlas and spice
            job.add_condor_cmd('+WantsGPU', 'true')
            job.add_condor_cmd('+WantGPU', 'true')
            job.add_condor_cmd(
                'Requirements',
                '(GPU_PRESENT =?= true) || (HasGPU =?= "gtx580")')

            self.set_post_script(gpu_postscript)
            self.add_post_script_arg(mf1)
            self.add_post_script_arg(mf2)
            self.add_post_script_arg(".0001")
            self.add_post_script_arg(match_file)
            self.add_post_script_arg(str(inj_per_job))
        else:
            self.add_file_opt("match-file",
                              match_file,
                              file_is_output_file=True)


class BanksimCombineNode(CondorDAGNode):
    def __init__(self, job, inj_num):
        CondorDAGNode.__init__(self, job)

        self.add_var_opt("inj-num", inj_num)

        outf = "match/match" + str(inj_num) + ".dat"

        self.add_file_opt("output-file", outf)


class FaithsimNode(CondorDAGNode):
    def __init__(self, job, tmplt_file, match_file, inj_per_job=None):
        CondorDAGNode.__init__(self, job)
        self.add_file_opt("param-file", tmplt_file)
        self.add_file_opt("match-file", match_file, file_is_output_file=True)


class InferenceJob(CondorDAGJob, CondorJob):
    def __init__(self,
                 log_dir,
                 executable,
                 cp,
                 section,
                 gpu=False,
                 accounting_group=None,
                 request_memory=None,
                 request_cpus=10):
        CondorDAGJob.__init__(self, "vanilla", os.path.abspath(executable))

        self.add_condor_cmd('initialdir',
                            os.path.dirname(os.path.abspath(executable)))

        if gpu:
            CondorJob.__init__(self, "vanilla", os.path.basename(executable),
                               2)
        # These are all python jobs so need to pull in the env
        self.add_condor_cmd('getenv', 'True')
        log_base = os.path.join(
            log_dir,
            os.path.basename(executable) + '-$(cluster)-$(process)')
        self.set_stderr_file(log_base + '.err')
        self.set_stdout_file(log_base + '.out')
        self.set_sub_file(executable + '.sub')

        if cp is not None:
            self.add_ini_opts(cp, section)

        if accounting_group:
            self.add_condor_cmd('accounting_group', accounting_group)

        if request_memory:
            self.add_condor_cmd('request_memory', request_memory)

        if request_cpus:
            self.add_condor_cmd('request_cpus', request_cpus)
