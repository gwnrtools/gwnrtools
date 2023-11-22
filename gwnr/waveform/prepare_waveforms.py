''' 
MIT License

Copyright (c) 2023 Vaishak Prasad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Script to 

    a) join h5 files
    b) extrapolate to infinity
    c) correct for CoM drift
    
    from SXS simulations
    
'''

import os
import scri
import numpy as np
from pathlib import Path


class PrepareSXSWaveform:
    ''' Prepare waveforms from a particular Lev and ECC of 
    an SXS NR runs by
    
    1) Joining all segments
    2) Extrapolating to infinity
    3) Transforming to CoM frame 
    4) Upload to cloud 
    
    
    Attributes
    ----------
    sim_dir : str/POSIXPath
              The root directory containing 
              the simulation directory
    sim_name : 
    out_dir
    joined_outfile_dir
    joined_waveform_outfile_name
    joined_waveform_outfile_path
    joined_horizons_outfile_name
    joined_waveform_outfile_path
    lev
    ecc
    extrap_out_dir
    
    
    
    Methods
    -------
    join_waveform_h5_files
    extrapolate
    join_horizons
    transform_to_CoM_frame
    upload_output_dir
    prepare_waveform
    
    
    
    '''
    
    def __init__(self,
                 sim_name,
                 sim_dir=Path('./'),
                 out_dir=None,
                 joined_waveform_outfile_name=None,
                 joined_horizons_outfile_name=None,
                 lev=2,
                 ecc=0):
        
        
        if not os.path.isabs(sim_dir):
            raise ValueError("Please provide the full sim path!")
            
        self._sim_dir = Path(sim_dir)
        
        self._sim_name = sim_name
        self._lev = lev
        self._ecc = ecc
        
        if joined_waveform_outfile_name is None:
            print("Choosing default directory for output...")
            joined_waveform_outfile_name = sim_name +\
                    f"Lev{self.lev}JoinedWaveform.h5"
            
        self._joined_waveform_outfile_name = joined_waveform_outfile_name
        
        if out_dir is None:
            self._out_dir = os.path.join(os.getcwd(), 
                                Path(f'{sim_name}_waveforms_Lev{self.lev}'))
            #print(f"Creating out directory ({sim_name}_waveforms) in cwd...")
            
        else:
            print(f"Out directory is set to {self.out_dir}")
            
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
        
        joined_outfile_dir = os.path.join(self.out_dir, Path(f"joined"))
        
        self._joined_outfile_dir = joined_outfile_dir
        
        if not os.path.isdir(joined_outfile_dir):
            os.mkdir(joined_outfile_dir)
            
        self._joined_outfile_path = os.path.join(joined_outfile_dir, 
                                            self.joined_waveform_outfile_name)
        
        if joined_horizons_outfile_name is None:
            self._joined_horizons_outfile_name = sim_name + \
                                    f"Lev{self.lev}JoinedHorizons.h5"
            
    @property
    def sim_dir(self):
        ''' The full path to the directory containing the 
        simulation '''
        return self._sim_dir
    
    @property
    def sim_name(self):
        return self._sim_name
    
    @property
    def out_dir(self):
        return self._out_dir
    
    @property
    def joined_outfile_dir(self):
        ''' The directory containing the 
        joined files '''
        return self._joined_outfile_dir
    
    @property
    def joined_waveform_outfile_name(self):
        return self._joined_waveform_outfile_name
    
    @property
    def joined_waveform_outfile_path(self):
        return os.path.join(self.joined_outfile_dir, 
                            Path(f"{self.joined_waveform_outfile_name}"))
    
    @property
    def joined_horizons_outfile_name(self):
        return self._joined_horizons_outfile_name
    
    @property
    def joined_horizons_outfile_path(self):
        return os.path.join(self.joined_outfile_dir, 
                            Path(f"{self.joined_horizons_outfile_name}"))
    
    @property
    def lev(self):
        return self._lev
    
    @property
    def ecc(self):
        return self._ecc
    
    @property
    def extrap_out_dir(self):
        return os.path.join(self.out_dir, Path("extrapolated"))
    
    def join_waveform_h5_files(self, verbose=False):
        ''' Join the waveform h5 files '''

        if Path(self.joined_horizons_outfile_path).exists():
            print("File already exists. Skipping operation.")

        else:
            print('Joining waveform h5 files...')
            
            data_paths_insp = os.path.join(self.sim_dir, 
                                      Path(f"{self.sim_name}/Ecc{self.ecc}"
                                           f"/Ev/Lev{self.lev}*/Run/GW2/"
                                           "rh_FiniteRadii_CodeUnits.h5"))
            
            data_paths_rdown = os.path.join(self.sim_dir, 
                                      Path(f"{self.sim_name}/Ecc{self.ecc}"
                                           f"/Ev/Lev{self.lev}_Ringdown/"
                                           f"Lev{self.lev}*/Run/GW2/"
                                           "rh_FiniteRadii_CodeUnits.h5"))
            
            
            if verbose:
                run_cmd = "JoinH5 -v"
                
            else:
                run_cmd = "JoinH5"
                
            run_cmd += f" -o {self.joined_waveform_outfile_path}"\
                            f" -l {data_paths_insp} {data_paths_rdown}"
            
            
                
            print(f"Running command\n {run_cmd}")
            
            #with open('join_waveforms_output.txt', "wb") as fout:
            #subprocess.check_call('dir',stdout=f)
                #cmd_out = subprocess.Popen(run_cmd, stdout=subprocess.PIPE)
                
                #for cline in iter(lambda: process.stdout.read(1), b""):
                    #sys.stdout.buffer.write(cline)
                    #f.buffer.write(cline)
            
            
            
            cmd = os.popen(run_cmd)
        
            out = cmd.read()
        
            print('Command output \n', out)
            
            
            print("Command completed. Please check Errors.txt for details")
        
        
    def extrapolate(self, 
                    ChMass = 1.0,
                    UseStupidNRARFormat = True):
        ''' Extrapolate the waveform '''
        
        try:
            files = os.listdir(self.extrap_out_dir)
        
            exists = np.array([item for item in files if 'Extrapolated' in item])
        #print(exists)
        except Exception as excep:

            print(excep)
            print("No extrapolated files from previous run found")
            exists = []

        if len(exists)>0:
            print('Skipping extrapolation')

        else:
            print('Extrapolating...')
        
            wf = scri.extrapolate(InputDirectory = self.joined_outfile_dir,
                                  OutputDirectory = self.extrap_out_dir,
                                  DataFile = self.joined_waveform_outfile_name,
                                  ChMass = ChMass, 
                                  UseStupidNRARFormat = UseStupidNRARFormat,
                                  DifferenceFiles = '',
                                  PlotFormat = '',)
        
    def join_horizons(self, verbose=False):
        ''' Join horizons file and save to the joined
        file dir '''
        
        if Path(self.joined_horizons_outfile_path).exists():
            print("File already exists. Skipping join horizons operation.")

        else:

            print('Joining Horizon h5 files...')
            
            input_insp_dat_rel_loc = Path(f"{self.sim_name}/Ecc{self.ecc}"
                                     f"/Ev/Lev{self.lev}*/Run/"
                                     "ApparentHorizons/Horizons.h5")
            
            input_rdown_dat_rel_loc = Path(f"{self.sim_name}/Ecc{self.ecc}"
                                     f"/Ev/Lev{self.lev}_Ringdown/"
                                     f"Lev{self.lev}*/Run/"
                                     "ApparentHorizons/Horizons.h5")
            
            data_paths_insp = os.path.join(self.sim_dir, 
                                      input_insp_dat_rel_loc)
            
            data_paths_rdown = os.path.join(self.sim_dir, 
                                      input_rdown_dat_rel_loc)
            if verbose:
                run_cmd = "JoinH5 -v"
                
            else:
                run_cmd = "JoinH5"
                
                
            run_cmd += f" -o {self.joined_horizons_outfile_path}"\
                                f" -l {data_paths_insp} {data_paths_rdown}"
            
            print(f"Running command\n {run_cmd}")
            
            
            cmd = os.popen(run_cmd)
        
            out = cmd.read()
        
            print('Command output \n', out)
            
            
            print("Command completed. Please check Errors.txt for details.")
        
    def transform_to_CoM_frame(self, 
                               skip_beginning_fraction=0.01,
                               skip_ending_fraction=0.10,
                               file_format="NRAR",
                               extrap_enn_list=[-1, 2, 3, 4, 5, 6]):
        
        from scri.SpEC.com_motion import remove_avg_com_motion
        
        try:
            files = os.listdir(self.extrap_out_dir)
        
            exists = np.array([item for item in files if 'CoM' in item])

        except Exception as excep:
            print(excep)
            print("Continuing with transformation")
            exists=[]

        if len(exists)>0:
            print('Skipping CoM transformation')
        else:

            print("Transforming to CoM frame...")
            
            for extrap_enn in extrap_enn_list:
                
                print(f"Working on Extrapolated N_{extrap_enn}")
                
                path_to_waveform_h5=os.path.join(self.extrap_out_dir, 
                                    f"rhOverM_Extrapolated_N{extrap_enn}.h5")

                path_to_horizons_h5 = self.joined_horizons_outfile_path

                remove_avg_com_motion(
                                  w_m=None,
                                  path_to_waveform_h5=path_to_waveform_h5,
                                  path_to_horizons_h5=path_to_horizons_h5,
                                  skip_beginning_fraction=skip_beginning_fraction,
                                  skip_ending_fraction=skip_ending_fraction,
                                  file_write_mode="w",
                                  m_A=None,
                                  m_B=None,
                                  file_format=file_format,
                                  write_corrected_file=True
                                  )
            
            
    def upload_output_dir(self):
        raise NotImplementedError
        
    def prepare_waveform(self, 
                         verbose=False, 
                         ChMass = 1.0,
                         UseStupidNRARFormat = True, 
                         skip_beginning_fraction=0.01,
                         skip_ending_fraction=0.10,
                         file_format="NRAR",
                         extrap_enn_list=[-1, 2, 3, 4, 5, 6],
                         upload=False):
        
        
        self.join_waveform_h5_files(verbose=verbose)
        
        self.extrapolate(ChMass=ChMass, 
                         UseStupidNRARFormat=UseStupidNRARFormat)
        
        self.join_horizons(verbose=verbose)
        
        self.transform_to_CoM_frame(skip_beginning_fraction=skip_beginning_fraction, 
                                    skip_ending_fraction=skip_ending_fraction,
                                    file_format=file_format,
                                    extrap_enn_list=extrap_enn_list)

        
        if upload:
            self.upload_output_dir()
            
            pass

        print('\n--------------------------------------------------------\n')

        return True
