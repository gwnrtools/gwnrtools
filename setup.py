#!/usr/bin/env python

# Copyright (c) 2018, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnrtools/gwnrtools/blob/master/LICENSE>
"""Build shim for gwnr.

All package metadata lives in pyproject.toml. This file exists only to
stamp git version information into gwnr/.version at build time, which
gwnr.get_version_information() reads back at run time.
"""

import subprocess
from pathlib import Path

VERSION = "2021.9.20"


def write_version_file(version):
    """Write a file with version information for use at run time

    Parameters
    ----------
    version: str
        A string containing the current version information

    Returns
    -------
    version_file: pathlib.Path
        Path to the version file (relative to the package directory)
    """
    version_file = Path("gwnr") / ".version"

    try:
        git_log = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%h %ai"]
        ).decode("utf-8")
        git_diff = (
            subprocess.check_output(["git", "diff", "."])
            + subprocess.check_output(["git", "diff", "--cached", "."])
        ).decode("utf-8")
    except (subprocess.CalledProcessError, OSError):
        # not building from a git checkout; keep any existing version file
        if version_file.is_file():
            return version_file
        git_version = version
    else:
        git_version = "{}: ({}) {}".format(
            version, "UNCLEAN" if git_diff else "CLEAN", git_log.rstrip()
        )

    with open(version_file, "w") as f:
        print(git_version, file=f)

    return version_file


if __name__ == "__main__":
    from setuptools import setup

    write_version_file(VERSION)
    setup()
