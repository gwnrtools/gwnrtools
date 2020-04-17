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

'''
Collections of configuration files for various 
evolution systems as Python dictionaries for 
their text and formatting
'''


__header__ = '''\
# Distributed under the MIT License.
# See LICENSE.txt for details.
'''

# Output files
reduction_data_file_name = 'EvolutionReductions'
volume_data_file_name = 'EvolutionVolume'


input_files = {}
input_files['bc_on_x_periodic_on_y'] = '''\
# Distributed under the MIT License.
# See LICENSE.txt for details.

# Executable: EvolvePlaneWave2D

AnalyticSolution:
  PlaneWave:
    WaveVector: [1.0, 0.0]
    Center: [1.0, 0.0]
    Profile:
      Gaussian:
        Amplitude: 1.0
        Center: 0.0
        Width: 0.5

Evolution:
  InitialTime: 0.0
  InitialTimeStep: 0.002
  TimeStepper:
    AdamsBashforthN:
      Order: 3

# 6.283185307179586
# [18.84955592153876, 10]
DomainCreator:
  Rectangle:
    LowerBound: [0.0, 0.0]
    UpperBound: [10, 6]
    IsPeriodicIn: [false, true]
    InitialRefinement: [4, 3]
    InitialGridPoints: [5, 5]
#  Disk:
#    InitialGridPoints: [7, 5]
#    InitialRefinement: 3
#    InnerRadius: 2.0
#    OuterRadius: 8.0
#    UseEquiangularMap: true

# Filtering is being tested by the 2D executable (see EvolveScalarWave.hpp)
# Filtering:
#   ExpFilter0:
#     Alpha: 12
#     HalfPower: 32

NumericalFlux:
  Upwind:

EventsAndTriggers:
  ? EveryNSlabs:
        N: 6
        Offset: 0
  : - ObserveErrorNorms
  ? EveryNSlabs:
        N: 6
        Offset: 0
  : - ObserveFields:
        VariablesToObserve: ["Psi", "Phi", "Pi", "dt(Psi)", "dt(Phi)", "dt(Pi)"]
  ? PastTime: 15
  : - Completion

Observers:
  VolumeFileName: "{0:s}"
  ReductionFileName: "{1:s}"
'''.format(volume_data_file_name, reduction_data_file_name)

input_files['bc_on_xy'] = '''\
# Distributed under the MIT License.
# See LICENSE.txt for details.

# Executable: EvolvePlaneWave2D

AnalyticSolution:
  PlaneWave:
    WaveVector: [1.0, 0.0]
    Center: [1.0, 0.0]
    Profile:
      Gaussian:
        Amplitude: 1.0
        Center: 0.0
        Width: 0.5

Evolution:
  InitialTime: 0.0
  InitialTimeStep: 0.002
  TimeStepper:
    AdamsBashforthN:
      Order: 3

# 6.283185307179586
# [18.84955592153876, 10]
DomainCreator:
  Rectangle:
    LowerBound: [0.0, 0.0]
    UpperBound: [10, 6]
    IsPeriodicIn: [false, false]
    InitialRefinement: [4, 3]
    InitialGridPoints: [5, 5]
#  Disk:
#    InitialGridPoints: [7, 5]
#    InitialRefinement: 3
#    InnerRadius: 2.0
#    OuterRadius: 8.0
#    UseEquiangularMap: true

# Filtering is being tested by the 2D executable (see EvolveScalarWave.hpp)
# Filtering:
#   ExpFilter0:
#     Alpha: 12
#     HalfPower: 32

NumericalFlux:
  Upwind:

EventsAndTriggers:
  ? EveryNSlabs:
        N: 6
        Offset: 0
  : - ObserveErrorNorms
  ? EveryNSlabs:
        N: 6
        Offset: 0
  : - ObserveFields:
        VariablesToObserve: ["Psi", "Phi", "Pi", "dt(Psi)", "dt(Phi)", "dt(Pi)"]
  ? PastTime: 15
  : - Completion

Observers:
  VolumeFileName: "{0:s}"
  ReductionFileName: "{1:s}"
'''.format(volume_data_file_name, reduction_data_file_name)

cluster_submission_files = {}
cluster_submission_files['Wheeler'] = '''#!/bin/bash -
#SBATCH -o spectre.out
#SBATCH -e spectre.out
#SBATCH --ntasks-per-node 24
#SBATCH -A sxs
#SBATCH --no-requeue
#SBATCH -J csw_exe_3d
#SBATCH --nodes 1
#SBATCH -t 23:59:00
                                                                                
# Distributed under the MIT License.                                            
# See LICENSE.txt for details.                                                  
                                                                                
# To run a job on Wheeler:                                                      
# - Set the -J, --nodes, and -t options above, which correspond to job name,    
#   number of nodes, and wall time limit in HH:MM:SS, respectively.             
# - Set the build directory, run directory, executable name,                    
#   and input file below. The input file path is relative to ${RUN_DIR}.        
#                                                                               
# NOTE: The executable will not be copied from the build directory, so if you   
#       update your build directory this file will use the updated executable.  
#                                                                               
# Optionally, if you need more control over how SpECTRE is launched on          
# Wheeler you can edit the launch command at the end of this file directly.     
#                                                                               
# To submit the script to the queue run:                                        
#   sbatch Wheeler.sh                                                           
############################################################################    
# Set paths
export RUN_DIR={0}
#
export SPECTRE_BUILD_DIR=/home/prayush/prayush/src/spectre/spectre_csw_exe_2/   
#
export SPECTRE_EXECUTABLE=./{1}
export SPECTRE_INPUT_FILE={2}


############################################################################    
# Set desired permissions for files created with this script                    
umask 0022

source ${SPECTRE_BUILD_DIR}/support/Environments/wheeler_clang.sh
spectre_load_modules && module load jemalloc && module swap blaze/3.6

cd ${RUN_DIR}

# The 23 is there because Charm++ uses one thread per node for communication
srun -n ${SLURM_JOB_NUM_NODES} -c 24 \
     ${SPECTRE_EXECUTABLE} ++ppn 23 --input-file ${SPECTRE_INPUT_FILE}

module swap python/3.6.5
python3 ${SPECTRE_BUILD_DIR}/src/Visualization/Python/GenerateXdmf.py \
  --file-prefix {3}\
  --output {3}
'''

cluster_submission_files_formatting = {}
cluster_submission_files_formatting['Wheeler'] = [
    'RUN_DIR', 'EXE', 'INPUT', 'OUTPUT_PREFIX'
]
