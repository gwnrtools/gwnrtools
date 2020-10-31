#!/usr/bin/env python

# Copyright (c) 2018, Prayush Kumar
# See LICENSE file for details: <https://github.com/prayush/gwnrtools/blob/master/LICENSE>

from __future__ import print_function

from os import environ, path
import subprocess
from pathlib import Path

NAME = 'gwnrtools'
VERSION = 'v2020.10.30'


def write_version_file(version):
    """ Writes a file with version information to be used at run time

    Parameters
    ----------
    version: str
        A string containing the current version information

    Returns
    -------
    version_file: str
        A path to the version file (relative to the src package directory)
    """
    version_file = Path(NAME) / ".version"

    try:
        git_log = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%h %ai"]).decode("utf-8")
        git_diff = (subprocess.check_output(["git", "diff", "."]) +
                    subprocess.check_output(["git", "diff", "--cached", "."
                                             ])).decode("utf-8")
    except subprocess.CalledProcessError as exc:  # git calls failed
        # we already have a version file, let's use it
        if version_file.is_file():
            return version_file.name
        # otherwise error out
        exc.args = ("unable to obtain git version information, and {} doesn't "
                    "exist, cannot continue ({})".format(
                        version_file, str(exc)), )
        raise
    else:
        git_version = "{}: ({}) {}".format(version,
                                           "UNCLEAN" if git_diff else "CLEAN",
                                           git_log.rstrip())
        print("parsed git version info as: {!r}".format(git_version))

    try:
        with open(version_file, "w") as f:
            print(git_version, file=f)
            print("created {}".format(version_file))
    except:
        with open(str(version_file), "w") as f:
            print(git_version, file=f)
            print("created {}".format(version_file))

    return version_file.name


def get_long_description():
    """ Finds the README and reads in the description """
    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, "README.md")) as f:
        long_description = f.read()
    return long_description


if "package_version" in environ:
    version = environ["package_version"]
    print("Setup.py using environment version='{0}'".format(version))
else:
    print("The variable 'package_version' was not present in the environment")
    try:
        from subprocess import check_output
        version = check_output(
            """git log -1 --format=%cd --date=format:'%Y.%m.%d.%H.%M.%S'""",
            shell=use_shell).decode('ascii').rstrip()
        print("Setup.py using git log version='{0}'".format(version))
    except:
        from time import strftime, gmtime
        version = strftime("%Y.%m.%d.%H.%M.%S", gmtime())
        print("Setup.py using strftime version='{0}'".format(version))

if __name__ == "__main__":
    from setuptools import setup, find_packages
    setup(
        name=NAME,
        version=VERSION,
        description=
        'A collection of tools for academic research in gravitational-wave astronomy & astrophysics',
        long_description=get_long_description(),
        license="GPL",
        url='https://github.com/prayush/gwnrtools',
        author='Prayush Kumar',
        author_email='prayush.kumar@gmail.com',
        packages=find_packages(),
        package_dir={NAME: NAME},
        package_data={
            # version info
            NAME: [write_version_file(VERSION)],
            'gwnrtools.data':
            ['gw_noise_curves/*.txt', 'gw_noise_curves/*.dat']
        },
        install_requires=[
            'astropy==2.0.16',
            'h5py>=2.10.0',
            # 'lalsuite>=6.63',
            'lscsoft_glue==2.0.0',
            'matplotlib==2.1',
            'numexpr',
            'numpy==1.16.5',
            'pandas==0.24.2',
            'pathlib',
            # 'pycbc>=1.15.4',
            'pyswarm==0.6',
            'pytest',
            'romspline==1.1.6',
            'scikit-learn==0.20.4',
            'scipy>=1.2.3',
            'seaborn==0.9.1',
            'six==1.10.0',
            'statsmodels>=0.10.2',
            'utils==0.9.0'
        ],
        scripts=[
            'bin/utils/makepdf', 'bin/banksim_generic.py',
            'bin/choose_testpoints.py', 'bin/choose_best_testpoints.py',
            'bin/remove_eliminated_testpoints.py',
            'bin/split_table_geometrically.py',
            'bin/gwnrtools_create_bank_workflow',
            'bin/gwnrtools_create_banksim_workflow',
            'bin/gwnrtools_create_faithsim_workflow',
            'bin/gwnrtools_write_inference_configs',
            'bin/gwnrtools_create_injections_inference_workflow',
            'bin/gwnrtools_create_public_events_inference_workflow',
            'bin/gwnrtools_banksim', 'bin/gwnrtools_faithsim',
            'bin/gwnrtools_force_success_from_condor_sub',
            'bin/gwnrtools_sample_parameter_space',
            'bin/gwnrtools_enigma_plan_calib_grid_and_make_dag',
            'bin/gwnrtools_enigma_sample_calib_parameters',
            'bin/utils/toggle_lsctable_type', 'bin/utils/ConvertHTMLToIpynb',
            'bin/analysis/ComputeOptimalSNRForGWSignals.py',
            'bin/nr/JoinDatainHDF',
            'bin/stats/ComputeInferredParametersFromLIPosterior.py'
        ],
    )
