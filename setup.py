#!/usr/bin/env python

# Copyright (c) 2018, Prayush Kumar
# See LICENSE file for details: <https://github.com/prayush/gwnrtools/blob/master/LICENSE>

# Construct the version number from the date and time this python version was created.
from os import environ
if "package_version" in environ:
    version = environ["package_version"]
    print("Setup.py using environment version='{0}'".format(version))
else:
    print("The variable 'package_version' was not present in the environment")
    try:
        from subprocess import check_output
        version = check_output(
            """git log -1 --format=%cd --date=format:'%Y.%m.%d.%H.%M.%S'""", shell=use_shell).decode('ascii').rstrip()
        print("Setup.py using git log version='{0}'".format(version))
    except:
        from time import strftime, gmtime
        version = strftime("%Y.%m.%d.%H.%M.%S", gmtime())
        print("Setup.py using strftime version='{0}'".format(version))
with open('_version.py', 'w') as f:
    f.write('__version__ = "{0}"'.format(version))


if __name__ == "__main__":
    from setuptools import setup, find_packages
    setup(name='gwnrtools',
          version=version,
          description='A collection of tools for academic research in gravitational-wave astronomy & astrophysics',
          license="GPL",
          url='https://github.com/prayush/gwnrtools',
          author='Prayush Kumar',
          author_email='prayush.kumar@gmail.com',
          package_dir={'gwnrtools': 'gwnrtools'},
          packages=find_packages(),
          install_requires=[
              'astropy>=4.0',
              'celluloid>=0.2.0',
              'h5py>=2.10.0',
              'lalsuite>=6.63',
              'lscsoft_glue==2.0.0',
              'matplotlib==3.1.2',
              'numexpr',
              'numpy==1.16.5',
              'pandas==0.24.2',
              'pycbc>=1.15.4',
              'pyswarm==0.6',
              'pytest',
              'romspline==1.1.6',
              'scikit-learn==0.20.4',
              'scipy==1.4.1',
              'seaborn==0.9.1',
              'six==1.10.0',
              'statsmodels>=0.10.2',
              'utils==0.9.0'
          ],
          scripts=['bin/utils/makepdf',
                   'bin/banksim_generic.py',
                   'bin/choose_testpoints.py',
                   'bin/choose_best_testpoints.py',
                   'bin/remove_eliminated_testpoints.py',
                   'bin/split_table_geometrically.py',
                   'bin/gwnrtools_create_bank_workflow',
                   'bin/gwnrtools_create_banksim_workflow',
                   'bin/gwnrtools_create_faithsim_workflow',
                   'bin/gwnrtools_write_inference_configs',
                   'bin/gwnrtools_create_injections_inference_workflow',
                   'bin/gwnrtools_create_public_events_inference_workflow',
                   'bin/gwnrtools_banksim',
                   'bin/gwnrtools_faithsim',
                   'bin/gwnrtools_force_success_from_condor_sub',
                   'bin/gwnrtools_sample_parameter_space',
                   'bin/utils/toggle_lsctable_type',
                   'bin/utils/ConvertHTMLToIpynb',
                   'bin/analysis/ComputeOptimalSNRForGWSignals.py',
                   'bin/nr/JoinDatainHDF',
                   'bin/stats/ComputeInferredParametersFromLIPosterior.py'],
          )
