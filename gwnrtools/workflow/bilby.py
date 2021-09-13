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
import numpy

from gwnrtools.utils import mkdir
from gwnrtools.analysis import (get_unique_hex_tag, Merger)
# from gwnrtools.stats.bilby_inference_utils import ??

from gwnrtools.utils import mkdir
from gwnrtools.analysis import Merger
# from gwnrtools.stats.bilby_inference_utils import InferenceConfigs
from gwnrtools.workflow.inference import (OneInferenceAnalysis,
                                          BatchInferenceAnalyses)


class BilbyInferenceInjectionAnalysis(OneInferenceAnalysis):
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
        super(BilbyInferenceInjectionAnalysis,
              self).__init__(opts,
                             run_dir,
                             config_files,
                             inj_exe_name=inj_exe_name,
                             inf_exe_name=inf_exe_name,
                             plt_exe_name=plt_exe_name,
                             verbose=verbose)