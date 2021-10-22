# Copyright (C) 2018 Prayush Kumar
#
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
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
from __future__ import print_function

import os
import h5py
import glob
import numpy as np
import re
from gwnr.utils import find_nearest

verbose = False


def GetQuantitySdNameFromKey(name):
    name = name.strip()
    quantity, sdname = name.split('[')
    sdname = sdname.strip().strip(']')
    return quantity, sdname


def ParseHeaderLineForSpECTabularOutputASCII(line):
    """
Can deal only with lines indexed in 4 or fewer digits
    """
    line = line.strip().strip('#').strip(' ')
    mm = re.search('\[[0-9]\]', line)
    if mm is None:
        mm = re.search('\[[0-9][0-9]\]', line)
        if mm is None:
            mm = re.search('\[[0-9][0-9][0-9]\]', line)
            if mm is None:
                mm = re.search('\[[0-9][0-9][0-9][0-9]\]', line)
                if mm is None:
                    # PASS, this line must be a generic comment
                    pass
                    #raise RuntimeError("Could not locate column number in %s" % line)
                else:
                    pass
            else:
                pass
        else:
            pass
    else:
        pass
    if mm is None:
        return None, None
    # subtract 1 to start col# from 0
    col = int(mm.group(0).strip(']').strip('[')) - 1
    full_key = line[mm.end():].strip().strip('=').strip()
    return col, full_key


def ParseHeaderForSpECTabularOutputASCII(
        filename,
        first_column_is_global=True,  # First column is time
        separate_quantities_from_subdomains=True,
        verbose=False):
    """
Parses the header of a SpEC subdomain-by-subdomain output file that
outputs the same quantitiy separately for each subdomain. Subdomains
don't all exist all the time, and there are periods for which some
subdomains are not used (because the same region has been re-gridded
more finely, usually).

This function returns the following:

"header_strings"   : a dictionary that takes one from the column
        number in the ASCII file to its full header string
"header_quantities": a dictionary that takes one from the quantity
        needed to each of the subdomains it exists in, PLUS the
        column its data is stored in.
    """
    # {{{
    header_strings = {}
    header_quantities = {}
    with open(filename) as fp:
        _header = fp.readlines()
        idx = 0
        for i in range(len(_header)):
            if '#' not in _header[i]:
                break
            else:
                if verbose:
                    print("\n\nChecking out line %d : %s" % (i, _header[i]))
                _header_string = _header[i].strip().strip('#').strip(' ')
                _header_col, _header_key =\
                    ParseHeaderLineForSpECTabularOutputASCII(_header_string)
                if verbose:
                    print("%s parsed to %s and %s" %
                          (_header_string, _header_col, _header_key))
                # NOW checking for comments in headers
                # They should be parsed to None and None
                if _header_col == None and _header_key == None:
                    continue
                header_strings[_header_col] = _header_key
                ###
                if separate_quantities_from_subdomains:
                    if idx == 0 and first_column_is_global:
                        idx += 1
                        continue
                    if verbose:
                        print("Trying header line : ", _header_key)
                    _header_quantity, _header_sdname = GetQuantitySdNameFromKey(
                        _header_key)
                    if _header_quantity not in list(header_quantities.keys()):
                        header_quantities[_header_quantity] = {}
                    header_quantities[_header_quantity][
                        _header_sdname] = _header_col
                else:
                    header_quantities[_header_key] = _header_col
                idx += 1
    #_header = _header[:i]
    return header_strings, header_quantities
    # }}}


def ReadH5DatasetWithLegend(tdset, downsample_by=1):
    retval = {}
    for idx, leg in enumerate(tdset.attrs['Legend']):
        if len(tdset.value[:, idx]) < downsample_by:
            retval[leg] = tdset.value[0, idx]
        else:
            retval[leg] = tdset.value[::downsample_by, idx]
    return retval


def GetSegmentDirectories(DIR,
                          LEV,
                          use_non_standard_segments=False,
                          non_standard_prefix='./',
                          verbose=False):
    if use_non_standard_segments:
        return GetNonStandardSegmentDirectories(
            DIR, LEV, non_standard_prefix=non_standard_prefix, verbose=verbose)
    inspiral_dirs = sorted(glob.glob(os.path.join(DIR, 'Lev%d_??/') % LEV))
    ringdown_dirs = sorted(
        glob.glob(os.path.join(DIR, 'Lev%d_Ringdown/Lev%d_??/' % (LEV, LEV))))
    if len(ringdown_dirs) >= 1:
        inspiral_dirs.extend(ringdown_dirs)
    if verbose:
        print("Reading from following dirs:\n", inspiral_dirs)
    return inspiral_dirs


def GetSegmentLettersFromName(NAME):
    retval = NAME.strip('/').split('_')[-1]
    if '_Ringdown' in NAME:
        retval = 'RD' + retval
    return retval


def GetNonStandardSegmentDirectories(
        DIR,
        LEV,
        non_standard_prefix='LATE_RINGDOWN_START_SEGMENTS/',
        verbose=False,
        debug=False):
    # {{{
    standard_inspiral_dirs = sorted(
        glob.glob(os.path.join(DIR, 'Lev%d_??/') % LEV))
    standard_ringdown_dirs = sorted(
        glob.glob(os.path.join(DIR, 'Lev%d_Ringdown/Lev%d_??/' % (LEV, LEV))))
    ##
    inspiral_dirs = sorted(
        glob.glob(os.path.join(DIR, non_standard_prefix + 'Lev%d_??/') % LEV))
    ringdown_dirs = sorted(
        glob.glob(
            os.path.join(
                DIR, non_standard_prefix + 'Lev%d_Ringdown/Lev%d_??/' %
                (LEV, LEV))))
    ##
    if debug:
        for kk in standard_inspiral_dirs:
            print(kk)
        print("\n")
        for kk in inspiral_dirs:
            print(kk)
        print("\n")
        for kk in standard_ringdown_dirs:
            print(kk)
        print("\n")
        for kk in ringdown_dirs:
            print(kk)
    ##
    final_dirs = []
    for idx, d in enumerate(standard_inspiral_dirs):
        lett_standard = GetSegmentLettersFromName(d)
        for jdx, dd in enumerate(inspiral_dirs):
            lett_non_standard = GetSegmentLettersFromName(dd)
            if lett_non_standard == lett_standard:
                break
        if jdx < (len(inspiral_dirs) - 1):
            # We must have found the same lettered segment in the non-standard
            # inspiral dirs as in the standard ones. Append the non-standard one.
            final_dirs.append(dd)
        else:
            # We must NOT have found the same lettered segment in the non-standard
            # ones as in the standard ones. Append the standard one.
            final_dirs.append(d)
    ##
    for idx, d in enumerate(inspiral_dirs):
        if d in final_dirs:
            continue
        final_dirs.append(d)
    ##
    for idx, d in enumerate(standard_ringdown_dirs):
        lett_standard = GetSegmentLettersFromName(d)
        for jdx, dd in enumerate(ringdown_dirs):
            lett_non_standard = GetSegmentLettersFromName(dd)
            if lett_non_standard == lett_standard:
                break
        if jdx < (len(inspiral_dirs) - 1):
            final_dirs.append(dd)
        else:
            final_dirs.append(d)
    ##
    for idx, d in enumerate(ringdown_dirs):
        if d in final_dirs:
            continue
        final_dirs.append(d)
    ##
    if verbose:
        print("Reading from following dirs:\n", final_dirs)
    return final_dirs
    # }}}


########################################
# USER-FACING FUNCTIONS
########################################

if verbose:
    print(""">>>>>>>>>>>>>>>>>>>>>>>>>
Name: ReadSpECTabularOutputFromASCII
Function: Read in SpEC's .dat or .txt (ASCII) output files from all available
segments and combine them. Provide file name with respect to Lev?_??/
directory. Returns combined data in a numpy.array.
""")


def ReadSpECTabularOutputFromASCII(DIR,
                                   LEV,
                                   FILE,
                                   use_non_standard_segments=False,
                                   non_standard_prefix='./',
                                   verbose=False,
                                   debug=False):
    """
  Read in SpEC's .dat or .txt (ASCII) output files from all available
  segments and combine them. Provide file name with respect to Lev?_??/
  directory. Returns combined data in a numpy.array.
    """
    # {{{
    if not os.path.exists(DIR):
        raise IOError("%s does not exist" % DIR)
    #
    inspiral_dirs = GetSegmentDirectories(
        DIR,
        LEV,
        use_non_standard_segments=use_non_standard_segments,
        non_standard_prefix=non_standard_prefix,
        verbose=verbose)
    #
    MAX_NROW = 20000
    MAX_NROW_DELTA = 10000
    MAX_NCOL = 200
    data = np.zeros([MAX_NROW, MAX_NCOL])
    NROW = 0  # Running counter
    NCOL = MAX_NCOL  # NCOL is fixed
    #
    get_NCOL = True
    for jdx, _dir in enumerate(inspiral_dirs):
        filename = os.path.join(_dir, FILE)
        if not os.path.exists(filename):
            if verbose:
                print("Skipping: %s" % filename)
            continue
        if debug:
            print("READING %s" % filename)
        _data = np.loadtxt(filename)
        #
        if len(np.shape(_data)) == 1:
            ncol = len(_data)
            nrow = 1
        else:
            ncol = np.shape(_data)[1]
            nrow = np.shape(_data)[0]
        if debug:
            print("shape of data = (%d,%d)" % (nrow, ncol))
        ncol = min(NCOL, ncol)
        if get_NCOL:
            NCOL = ncol
            get_NCOL = False
        if debug:
            print("nco, NCOL, MAX_NCOL = (%d, %d,%d)" % (ncol, NCOL, MAX_NCOL))
        #
        # Ensure new data can be accommodated
        while (NROW + nrow) > MAX_NROW:
            tmp_z2 = np.zeros([MAX_NROW_DELTA, MAX_NCOL])
            data = np.append(data, tmp_z2, axis=0)
            MAX_NROW += MAX_NROW_DELTA
        #
        if len(np.shape(_data)) == 1:
            data[NROW, :ncol] = _data[:ncol]
            NROW += 1
        else:
            data[NROW:NROW + nrow, :ncol] = _data[:, :ncol]
            NROW += nrow
    #
    return data[:NROW, :NCOL]
    # }}}


if verbose:
    print(""">>>>>>>>>>>>>>>>>>>>>>>>>
Name: ReadSpECTabularOutputWithColsFromASCII
Function: Reads in data from SpEC output files that contains variables written
 **subdomain-by-subdomain**.  The same variable needs to be stored in separate
  columns for different sub-domains.

Note1 : This function will takes ASCII output files from all available
segments and combine them. Provide file name with respect to Lev?_??/
directory. Each column in data file is stored in a dictionary as per the
header, and then the data is combined.

This makes sure that if some segments are missing certain columns,
those are smoothly glossed over. E.g. subdomain X may exist between t = 0-1000M
but not between 1000-1500M and then again from 1500-\infty M.
""")


def ReadSpECTabularOutputWithColsFromASCII(DIR,
                                           LEV,
                                           FILE,
                                           downsample_by=1,
                                           use_non_standard_segments=False,
                                           non_standard_prefix='./',
                                           verbose=False,
                                           debug=False):
    """
  Reads in data from SpEC output files that contains variables written
   **subdomain-by-subdomain**.  The same variable needs to be stored in separate
    columns for different sub-domains.

  Note1 : This function will takes ASCII output files from all available
  segments and combine them. Provide file name with respect to Lev?_??/
  directory. Each column in data file is stored in a dictionary as per the
  header, and then the data is combined.

  This makes sure that if some segments are missing certain columns,
  those are smoothly glossed over. E.g. subdomain X may exist between t = 0-1000M
  but not between 1000-1500M and then again from 1500-\infty M.
    """
    # {{{
    if not os.path.exists(DIR):
        raise IOError("%s does not exist" % DIR)
    #
    inspiral_dirs = GetSegmentDirectories(
        DIR,
        LEV,
        use_non_standard_segments=use_non_standard_segments,
        non_standard_prefix=non_standard_prefix,
        verbose=verbose)
    data = {}
    #
    for idx, _dir in enumerate(inspiral_dirs):
        filename = os.path.join(_dir, FILE)
        if not os.path.exists(filename):
            if verbose:
                print("Skipping: %s" % filename)
            continue
        if debug:
            print("READING %s" % filename)
        #
        # Read in data
        _data = np.loadtxt(filename)
        if np.shape(_data) == (0, ):
            if debug:
                print("No data found for %s" % filename)
            continue
        header_string, header_quantities = ParseHeaderForSpECTabularOutputASCII(
            filename)
        #
        # Now, parse columns as per column header strings
        for jdx, qty in enumerate(header_quantities):
            if qty not in list(data.keys()):
                data[qty] = {}
            for kdx, sdn in enumerate(header_quantities[qty]):
                col_num = header_quantities[qty][sdn]
                try:
                    tmpd = np.array(
                        list(
                            zip(_data[::downsample_by, 0],
                                _data[::downsample_by, col_num])))
                except:
                    tmpd = np.array([[_data[0], _data[col_num]]])
                #
                if sdn not in list(data[qty].keys()):
                    data[qty][sdn] = tmpd
                else:
                    data[qty][sdn] = np.append(data[qty][sdn], tmpd, axis=0)
        # Delete temp data
        del _data
    return data
    # }}}


if verbose:
    print(""">>>>>>>>>>>>>>>>>>>>>>>>>
Name: ReadSpECGlobalOutputWithColsFromASCII
Function: Reads in data from SpEC output files that contains global variables
dumped as a function of time.

Note 1: This function will takes ASCII output files from all available
segments and combine them. Provide file name with respect to Lev?_??/
directory. Each column in data file is stored in a dictionary as per the
header, and then the data is combined.

This makes sure that if some segments are missing certain columns,
those are smoothly glossed over. E.g. subdomain X may exist between t = 0-1000M
but not between 1000-1500M and then again from 1500-\infty M.

Note 2: SEE SIMILAR FUNCTION ReadSpECTabularOutputFromASCII.
  """)


def ReadSpECGlobalOutputWithColsFromASCII(DIR,
                                          LEV,
                                          FILE,
                                          downsample_by=1,
                                          use_non_standard_segments=False,
                                          non_standard_prefix='./',
                                          verbose=False,
                                          debug=False):
    """
  Reads in data from SpEC output files that contains global variables
  dumped as a function of time.

  Note 1: This function will takes ASCII output files from all available
  segments and combine them. Provide file name with respect to Lev?_??/
  directory. Each column in data file is stored in a dictionary as per the
  header, and then the data is combined.

  This makes sure that if some segments are missing certain columns,
  those are smoothly glossed over. E.g. subdomain X may exist between t = 0-1000M
  but not between 1000-1500M and then again from 1500-\infty M.

  Note 2: SEE SIMILAR FUNCTION ReadSpECTabularOutputFromASCII.
    """
    # {{{
    if not os.path.exists(DIR):
        raise IOError("%s does not exist" % DIR)
    #
    inspiral_dirs = GetSegmentDirectories(
        DIR,
        LEV,
        use_non_standard_segments=use_non_standard_segments,
        non_standard_prefix=non_standard_prefix,
        verbose=verbose)
    data = {}
    #
    for idx, _dir in enumerate(inspiral_dirs):
        filename = os.path.join(_dir, FILE)
        if not os.path.exists(filename):
            if verbose:
                print("Skipping: %s" % filename)
            continue
        if debug:
            print("READING %s" % filename)
        #
        # Read in data
        _data = np.loadtxt(filename)
        if np.shape(_data) == (0, ):
            if debug:
                print("No data found for %s" % filename)
            continue
        _, header_quantities = ParseHeaderForSpECTabularOutputASCII(
            filename, separate_quantities_from_subdomains=False)
        #
        # Now, parse columns as per column header strings
        for jdx, qty in enumerate(header_quantities):
            col_num = header_quantities[qty]
            try:
                tmpd = np.array(
                    list(
                        zip(_data[::downsample_by, 0], _data[::downsample_by,
                                                             col_num])))
            except:
                tmpd = np.array([[_data[0], _data[col_num]]])
            #
            if qty not in list(data.keys()):
                data[qty] = tmpd
            else:
                data[qty] = np.append(data[qty], tmpd, axis=0)
    return data
    # }}}


def GetSpECRDStartTime(DIR, LEV):
    """
Get the time at which ringdown segments are started for a given simulation.
Inputs needed are the main run directory (path to "Ev"), and Lev number (int)
    """
    if not os.path.exists(os.path.join(DIR, 'Lev%s_Ringdown/' % LEV)):
        print("  Warning: Run in %s has NOT STARTED RINGDOWN AT LEV%d" %
              (DIR, LEV))
        return -1
    filename = os.path.join(
        DIR, 'Lev%s_Ringdown/Lev%d_AA/Run/ConstraintNorms/GhCe_Norms.dat' %
        (LEV, LEV))
    if not os.path.exists(filename):
        print("PATH %s does not exist" % filename)
        raise IOError("Could not determine ringdown start time")
    d = np.loadtxt(filename)
    if len(np.shape(d)) == 1:
        return d[0]
    elif len(np.shape(d)) == 2:
        return d[0, 0]


def GetSpECAhCAppearanceTime(DIR, LEV):
    """
Get the time at which ringdown segments are started for a given simulation.
Inputs needed are the main run directory (path to "Ev"), and Lev number (int)
    """
    ahc_glob = glob.glob(
        os.path.join(DIR, "Lev%d_??/Run/ForContinuation/AhC.dat" % LEV))
    ahc_glob = sorted(ahc_glob)
    if len(ahc_glob) == 0:
        print(
            "  Warning: AhC as not been found even once @ Run IN %s AT LEV%d" %
            (DIR, LEV))
        return -1
    ahc_file = ahc_glob[0]
    if not os.path.exists(ahc_file):
        print("PATH %s does not exist" % ahc_file)
        raise IOError("Could not determine the time when AhC was found.")
    d = np.loadtxt(ahc_file)
    if len(np.shape(d)) == 1:
        return d[0]
    elif len(np.shape(d)) == 2:
        return d[0, 0]


if verbose:
    print(""">>>>>>>>>>>>>>>>>>>>>>>>>
Name: ReadSpECTabularOutputFromH5
Function:  Read in SpEC's HDF5 output files from all available segments and
 combine them. Provide file name with respect to Lev?_?? directory.
  """)


def ReadSpECTabularOutputFromH5(DIR,
                                LEV,
                                FILE,
                                GROUP='AhA.dir',
                                DATASET='',
                                use_non_standard_segments=False,
                                non_standard_prefix='./',
                                verbose=False,
                                debug=False):
    """
  Read in SpEC's HDF5 output files from all available segments and combine them.
  Provide file name with respect to Lev?_?? directory.
  """
    # {{{
    if not os.path.exists(DIR):
        raise IOError("%s does not exist" % DIR)
    #
    inspiral_dirs = GetSegmentDirectories(
        DIR,
        LEV,
        use_non_standard_segments=use_non_standard_segments,
        non_standard_prefix=non_standard_prefix,
        verbose=verbose)
    #
    MAX_NROW = 20000
    MAX_NROW_DELTA = 10000
    MAX_NCOL = 200
    data = np.zeros([MAX_NROW, MAX_NCOL])
    NROW = 0  # Running counter
    NCOL = 0  # NCOL is fixed
    #
    get_NCOL = True
    for jdx, _dir in enumerate(inspiral_dirs):
        filename = os.path.join(_dir, FILE)
        if not os.path.exists(filename):
            if verbose:
                print("Skipping: %s" % filename)
            continue
        if debug:
            print("READING %s" % filename)
        with h5py.File(filename, 'r') as fp:
            try:
                if GROUP is not '':
                    _data = fp[GROUP]
                else:
                    _data = fp
            except:
                continue
            if DATASET is not '':
                _data = _data[DATASET]
            else:
                raise IOError("Please provide name of dataset to read")
            #
            if debug:
                print("Shape of dataset = ", np.shape(_data))
            if len(np.shape(_data)) == 1:
                ncol = len(_data)
                nrow = 1
            else:
                ncol = np.shape(_data)[1]
                nrow = np.shape(_data)[0]
            NCOL = max(NCOL, ncol)
            if get_NCOL:
                NCOL = ncol
                get_NCOL = False
            #
            # Ensure new data can be accommodated
            while (NROW + nrow) > MAX_NROW:
                tmp_z2 = np.zeros([MAX_NROW_DELTA, MAX_NCOL])
                data = np.append(data, tmp_z2, axis=0)
                MAX_NROW += MAX_NROW_DELTA
            #
            if len(np.shape(_data)) == 1:
                data[NROW, :ncol] = _data[:]
                NROW += 1
            else:
                data[NROW:NROW + nrow, :ncol] = _data[:, :]
                NROW += nrow
    #
    return data[:NROW, :NCOL]
    # }}}


if verbose:
    print(""">>>>>>>>>>>>>>>>>>>>>>>>>
Read HDF5 files completely, recursively. Returns a dictionary with structure
of input file preserved.

NOTE: First input needed is the dictionary output has to be appended to.
 If reading the first time, pass "{}"
    """)


def ReadH5Dir(out_dict,
              in_dir,
              downsample_by=1,
              read_dirs_matching_0level='',
              read_dirs_matching_alllevels='',
              verbose=False,
              debug=False):
    """
Read HDF5 files completely, recursively. Returns a dictionary with structure
of input HDF5 file preserved.

NOTE: First input needed is the dictionary output has to be appended to.
 If reading the first time, pass "{}"
    """
    # {{{
    retval = out_dict
    for kk in list(in_dir.keys()):
        if type(in_dir[kk]) == h5py._hl.group.Group:
            if read_dirs_matching_0level != '':
                if kk.find(read_dirs_matching_0level) < 0:
                    if verbose:
                        print("Skipping directory: ", kk)
                    continue
            if verbose:
                print("Reading directory: ", kk)
            if kk not in list(retval.keys()):
                retval[kk] = {}
            retval[kk] = ReadH5Dir(
                retval[kk],
                in_dir[kk],
                read_dirs_matching_0level=read_dirs_matching_alllevels,
                verbose=debug)
        elif type(in_dir[kk]) == h5py._hl.dataset.Dataset:
            if kk not in list(retval.keys()):
                if verbose:
                    print("Reading dataset: ", kk)
                retval[kk] = ReadH5DatasetWithLegend(
                    in_dir[kk], downsample_by=downsample_by)
            else:
                if verbose:
                    print("Appending dataset: ", kk)
                # TO CHANGE DATATYPE TO PANDAS, CHANGE THE FOLLOWING
                _data = ReadH5DatasetWithLegend(in_dir[kk],
                                                downsample_by=downsample_by)
                for col_num, col_name in enumerate(_data.keys()):
                    if col_name not in list(retval[kk].keys()):
                        retval[kk][col_name] = _data[col_name]
                    else:
                        retval[kk][col_name] = np.append(retval[kk][col_name],
                                                         _data[col_name],
                                                         axis=0)
    return retval
    # }}}


if verbose:
    print(""">>>>>>>>>>>>>>>>>>>>>>>>>
Name: ReadSpECTabularOutputWithColsFromH5
Function: Reads in SpEC output stored in HDF5 format. The structure of HDF5 file is
preserved in output dictionary. And output from different segments is combined.

Note 1: Provide file name with respect to Lev?_??/
directory. Each column in data file is stored in a dictionary as per the
header, and then the data is combined.

This makes sure that if some segments are missing certain columns,
those are smoothly glossed over. E.g. subdomain X may exist between t = 0-1000M
but not between 1000-1500M and then again from 1500-\infty M.
    """)


def ReadSpECTabularOutputWithColsFromH5(DIR,
                                        LEV,
                                        FILE,
                                        downsample_by=1,
                                        read_dirs_matching_0level='',
                                        read_dirs_matching_alllevels='',
                                        use_non_standard_segments=False,
                                        non_standard_prefix='./',
                                        verbose=False,
                                        debug=False):
    """
Reads in SpEC output stored in HDF5 format. The structure of HDF5 file is
preserved in output dictionary. And output from different segments is combined.

Note 1: Provide file name with respect to Lev?_??/
directory. Each column in data file is stored in a dictionary as per the
header, and then the data is combined.

This makes sure that if some segments are missing certain columns,
those are smoothly glossed over. E.g. subdomain X may exist between t = 0-1000M
but not between 1000-1500M and then again from 1500-\infty M.
    """
    # {{{
    if not os.path.exists(DIR):
        raise IOError("%s does not exist" % DIR)
    #
    inspiral_dirs = GetSegmentDirectories(
        DIR,
        LEV,
        use_non_standard_segments=use_non_standard_segments,
        non_standard_prefix=non_standard_prefix,
        verbose=verbose)
    data = {}
    #
    for idx, _dir in enumerate(inspiral_dirs):
        filename = os.path.join(_dir, FILE)
        if not os.path.exists(filename):
            if verbose:
                print("Skipping: %s" % filename)
            continue
        if debug:
            print("READING %s" % filename)
        # Read in data
        fin = h5py.File(filename, 'r')
        data = ReadH5Dir(
            data,
            fin,
            downsample_by=downsample_by,
            read_dirs_matching_0level=read_dirs_matching_0level,
            read_dirs_matching_alllevels=read_dirs_matching_alllevels,
            verbose=debug)
        fin.close()
    return data
    # }}}


if verbose:
    print(""">>>>>>>>>>>>>>>>>>>>>>>>>
Name: GetOpOfQuantityOverDomain
Function: Wrapper function that takes in a dataset that contains some quantity
Q_i == Q_i(t) over each subdomain, as a function of time; and maps them all
{Q_i} with op_func: {Q_i} --> Q_o, that operates on the complete set {Q_i}
from all subdomains and returns a single value Q_o. Q_o is returned.

INPUTS:

data_dict : (dictionary, with keys "SUBDOMAIN-NAME.dir")
op_func   : (function) It takes in all {Q_i} together and maps
            them to a single value Q_o
    """)


def DummyMin(vals, sds):
    return np.min(vals)


def GetOpOfQuantityOverDomain(data_dict,
                              op_func=DummyMin,
                              verbose=True,
                              debug=False):
    """
Wrapper function that takes in a dataset that contains some quantity
Q_i == Q_i(t) over each subdomain, as a function of time; and maps them all
{Q_i} with op_func: {Q_i} --> Q_o, that operates on the complete set {Q_i}
from all subdomains and returns a single value Q_o. Q_o is returned.

INPUTS:

data_dict : (dictionary, with keys "SUBDOMAIN-NAME.dir")
op_func   : (function) It takes in all {Q_i} together and maps
            them to a single value Q_o
    """
    all_subdomains = list(data_dict.keys())
    if len(all_subdomains) == 0:
        raise RuntimeError("No subdomains found in input dictionary..")
    if debug:
        print("All subdomains:- ", all_subdomains)

    # Get time series from data
    if verbose:
        print("Extracting global times .. ")

    for idx, sd in enumerate(all_subdomains):
        if idx != 0:
            sd_tseries = data_dict[sd][:, 0]
            if sd_tseries.min() < all_tseries.min():
                mask = sd_tseries < all_tseries.min()
                all_tseries = np.append(sd_tseries[mask], all_tseries)
            if sd_tseries.max() > all_tseries.max():
                mask = sd_tseries > all_tseries.max()
                all_tseries = np.append(all_tseries, sd_tseries[mask])
        else:
            all_tseries = data_dict[sd][:, 0]

    # Get min/max over *available* subdomains at all times
    if verbose:
        print("Computing min/max data at those times .. ")

    yseries = np.array([])
    subdomainseries = []
    for idx, tvalue in enumerate(all_tseries):
        avail_values = np.array([])
        avail_subdomains = []
        for sd in all_subdomains:
            sd_tseries = data_dict[sd][:, 0]
            if tvalue not in sd_tseries:
                continue
            sd_yseries = data_dict[sd][:, 1]
            idx_tvalue, _ = find_nearest(sd_tseries, tvalue)
            avail_values = np.append(avail_values, sd_yseries[idx_tvalue])
            avail_subdomains.append(sd)
        yseries = np.append(yseries, op_func(avail_values, avail_subdomains))
        subdomainseries.append(avail_subdomains[np.where(
            op_func(avail_values, avail_subdomains) == yseries[-1])[0][0]])
    return all_tseries, yseries, subdomainseries
