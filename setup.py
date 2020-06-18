#!/usr/bin/env python

# Copyright (c) 2018, Prayush Kumar
# See LICENSE file for details: <https://github.com/prayush/GWNRTools/blob/master/LICENSE>

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
    setup(name='GWNRTools',
          version=version,
          description='Manipulating GW data in the form of time-dependent functions of spin-weighted spherical harmonics from NR simulations',
          license="GPL",
          url='https://github.com/prayush/GWNRTools',
          author='Prayush Kumar',
          author_email='prayush.kumar@gmail.com',
          package_dir={'GWNRTools': 'GWNRTools'},
          packages=find_packages(),
          requires=['numpy', 'scipy', 'pycbc', 'lal', 'matplotlib'],
          scripts=['bin/Utils/makepdf',
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
                   'bin/Utils/toggle_lsctable_type',
                   'bin/Utils/ConvertHTMLToIpynb',
                   'bin/DataAnalysis/ComputeOptimalSNRForGWSignals.py',
                   'bin/NR/JoinDatainHDF',
                   'bin/Stats/ComputeInferredParametersFromLIPosterior.py'],
          )
