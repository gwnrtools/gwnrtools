#!/usr/bin/env python
# Copyright (C) 2015  Prayush Kumar
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

import os, sys, time
import commands as cmd
import h5py
from numpy import where, ceil, log2
import numpy as np

import lal
from pycbc.types import TimeSeries
from utils import amplitude_from_polarizations

from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils

sys.path.append(cmd.getoutput('pwd -P'))
import UseNRinDA


@lsctables.use_in
class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
    pass


__author__ = "Prayush Kumar <prkumar@cita.utoronto.ca>"

PROGRAM_NAME = os.path.abspath(sys.argv[0])

__itime__ = time.time()

################################################################################
################################################################################
### Constants Definitions ###
################################################################################
################################################################################
MAX_NR_LENGTH = 100000

EPS_Params = {'eta' : 1.e-7,\
              'spin1x' : 1.e-7, 'spin1y' : 1.e-7, 'spin1z' : 1.e-7,\
              'spin2x' : 1.e-7, 'spin2y' : 1.e-7, 'spin2z' : 1.e-7\
              }
EPS_Default = 1.e-7
params_tested = [
    'eta', 'spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z'
]

################################################################################
################################################################################
### Function Definitions ###
################################################################################
################################################################################


def nextpow2(x):
    return int(2**ceil(log2(x)))


################################################################################
# Matches the pameter 'param' in templates p and c, and checks if they agrene
# within the tolerance specified above as a constant
################################################################################
def does_this_map(p, c, param='eta', return_err=False, verbose=False):
    """
Judge if the given template 'p' matches the given catalog row 'c' for the param
  """
    if type(p) == dict: val1 = p[param]
    elif type(p) == lsctables.SnglInspiral or type(p) == lsctables.SimInspiral:
        val1 = getattr(p, param)
    else:
        raise IOError("This type %s not supported by does_this_map" %
                      str(type(p)))

    if type(c) == dict: val2 = c[param]
    elif type(c) == lsctables.SnglInspiral or type(c) == lsctables.SimInspiral:
        val2 = getattr(c, param)
    else:
        raise IOError("This type %s not supported by does_this_map" %
                      str(type(c)))

    if param in EPS_Params.keys(): eps = EPS_Params[param]
    else: eps = EPS_Default

    num_err = np.abs(val1 - val2)
    den_err = np.abs((val1 + val2) / 2.)
    if den_err == 0 or np.abs(val1) <= eps or np.abs(val2) <= eps:
        frac_err = num_err
    else:
        frac_err = num_err / den_err

    if verbose:
        print("Param error for %s: %.2e, Threshold: %.2e" %
              (param, frac_err, eps))

    if frac_err <= eps:
        if return_err: return True, frac_err
        else: return True
    else:
        if return_err: return False, frac_err
        else: return False


################################################################################
# Return the data location for a given NR template
################################################################################
def get_nr_data_location(p,\
                      catalog_var='NR_CATALOG_PATH',\
                      file_var='NR_CATALOG_FILE',\
                      map_var='TemplateBankToNRMappings-DataLocation',\
                      method='tmplt_bank_map',
                      use_longest_simulation=True,\
                      verbose=False):
    """
 This function returns the location of the NR data corresponding to the
 parameters input by the user. There are a few ways in which the location
 information is provided through:

 (a) The field 'numrel_data' in p

 (b) A template bank map file (HDF5), which stores the location of the NR data
     labelled by the 'event_id' of each of the templates in the input sngl
     inspiral bank. This is to be used when using a normal template bank in a
     search, and we want to fix the mappings a priori.

 (c) A catalog file (LIGOLW XML) which contains the parameters of all available
     simulations and their location on accessible disk. This is more general,
     and has the flexibility that the matching of data is done against the
     entire catalog allowing the code to pick the longest of available
     simulations with matching parameters.

 The location of the catalog/map file is provided through the ENVIRONMENT
 VARIABLES:

NR_CATALOG_PATH: catalog/mapping-file location
NR_CATALOG_FILE: catalog/mapping file's name

 NOTE: the code will basically find ANY file pointed to by these two variables,
       so the 'method' argument is the real restriction.

 INPUT ARGUMENTS:
 1. p : DICTIONARY object being passed by the get_td_waveform function. this
        contains the masses, spins, and other parameters required to create
        the template. This might have a 'numrel_data' field that contains the
        location of NR data file.

 2. catalog_var : STRING-Name of the environment variable which contains the
        directory(ies), ONE of which HAS to contain the catalog/map file.

 3. file_var : STRING-Name of the environment variable which contains the name
        of the catalog/map file

 4. map_var : STRING-['TemplateBankToNRMappings-DataLocation', 'TemplateBankToNRMappings']
        THIS IS REQUIRED IFF method == 'tmplt_bank_map'..
        It is the name of the group in the HDF5 file which contains
        'event_id' --> 'data location' maps

 5. method : STRING-['tmplt_bank_map', 'catalog']
        This input controls the switch between the two, although this function
        intelligently deciphers which method to use from
        "NR_CATALOG_FILE"'s file-extension.

 6. use_longest_simulation : BOOL-If multiple simulations are caught as matching
        the input parameters, use the one with the lowest starting frequency.

 7. verbose : BOOL-print out informative messages during code execution.
    """
    error_msg = """\
\n\n
################################################################################

**NR waveform catalog not found in environment **

Please add the path to to your NR catalog to NR_CATALOG_PATH,
 and NR catalog file-name to NR_CATALOG_FILE with:

export NR_CATALOG_PATH={}:$NR_CATALOG_PATH
export NR_CATALOG_FILE={}:$NR_CATALOG_FILE

################################################################################
\n"""
    if verbose: print("\n @INPUTS: ", p)
    #################################################################
    # O: Check INPUTs for one of ["numrel_data", "event_id"]
    #################################################################
    pkeys = [str(kk) for kk in p.keys()]
    if 'numrel_data' in pkeys:
        if verbose: print(" Found numrel_data column : %s" % p['numrel_data'])
        return str(p['numrel_data'])

    if 'catalog_var' in pkeys: catalog_var = p['catalog_var']
    if 'file_var' in pkeys: file_var = p['file_var']
    if 'map_var' in pkeys: map_var = p['map_var']
    if 'method' in pkeys: method = p['method']
    if 'use_longest_simulation' in pkeys:
        use_longest_simulation = p['use_longest_simulation']

    #################################################################
    # 1. LOCATE the NR MAPPING / CATALOG FILE
    #################################################################
    catalog_dirs = os.environ[catalog_var].split(':')
    catalog_dirs.append('.')  # Add PWD -- just in case --
    catalog_file = os.environ[file_var]
    found_flag = False
    for d in catalog_dirs:
        fp = os.path.abspath(os.path.join(d, catalog_file))
        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            found_flag = True
            break
    if not found_flag: raise RuntimeError(error_msg)

    if verbose: print("Mapping/Catalog file located. Reading : %s " % fp)

    #################################################################
    # 2. UPDATE method for mapping template to NR data
    #################################################################
    if fp.endswith('.h5'):
        method = 'tmplt_bank_map'
    elif fp.endswith('.xml') or fp.endswith('.xml.gz'):
        method = 'catalog'
    elif 'tmplt_bank_map' not in method and 'catalog' not in method:
        raise IOError("Method %s is not supported yet..\n" % method)

    # If 'numrel_data' is not present in p, *and* 'event_id' is also not
    # present, there is no way to locate NR data with method == 'tmplt_bank_map'
    if 'event_id' not in pkeys and 'tmplt_bank_map' in method:
        raise IOError("No input event_id")

    #################################################################
    # 3. Read in the MAPPING / CATALOG FILE
    # 4. Return the location of NR data
    #################################################################
    if 'tmplt_bank_map' in method:
        #################################################################
        # 3. READ the MAPPING FILE
        try:
            fin = h5py.File(fp, 'r')
        except:
            raise IOError(error_msg +
                          (" ** Could not open catalog file %s **" % fp))
        #
        gin = fin[map_var]
        #
        # Now get the event_id field of the template and use it to get NR data's
        # location from template bank mapping file
        tmplt_tag = str(p['event_id'])
        catalog_tags = [str(kk) for kk in gin.keys()]
        if tmplt_tag not in catalog_tags:
            raise IOError(\
                "Template %s not found in %s. Is this the correct catalog?" %\
                (tmplt_tag, fp))
        #
        if verbose:
            print("Template event_id found : %s " % tmplt_tag)
            print("Location of NR data     : %s " % str(gin[tmplt_tag].value))
        #
        ################################################################
        # 4. Return the LOCATION of template's NR DATA
        return str(gin[tmplt_tag].value)
        #
    elif 'catalog' in method:
        #################################################################
        # 3. READ IN CATALOG XML
        indoc = ligolw_utils.load_filename(fp, \
              contenthandler=LIGOLWContentHandler, verbose=verbose)
        #
        try:
            fin = lsctables.SimInspiralTable.get_table(indoc)
        except:
            raise IOError("Catalog file %s must have a SimInspiral table" % fp)
        #################################################################
        # 4.1 Find the LOCATION of template's NR DATA
        #    This is done by matching the parameters of the template with
        #    those of all NR simulations in the catalog. The acceptable
        #    difference thresholds are defined at the top of this script.
        errs, mflower, rows = [], [], []
        for idx, row in enumerate(fin):
            if verbose: print("\n .. checking row %d in catalog" % idx)
            tmp_errs = []
            for param in params_tested:
                is_map = True
                tval, err = does_this_map(p, row, param=param, \
                                            return_err=True, verbose=verbose)
                # Store the fractional error for this parameter, for this row
                errs.append(err)
                # Now, even if *one* parameter differs, we reject this row and
                # get out of the loop over remaining parameters
                if not tval:
                    is_map = False
                    break
            #
            # Store the maximum of (fractional errors over all tested parameters)
            # for this particular row in the catalog table
            errs.append(np.array(errs).max())
            #
            # If the template is mapped with at least one row in the catalog table,
            # 'is_map' would be True. If so, print out information about the NR data
            # and break out of the loop over the catalog
            if is_map:
                #
                # Also store the initial orbital frequency for this simulation
                mflower.append(row.f_lower)
                rows.append(row)
                #
                if verbose:
                    print("Template found in catalog: %s " % fp,
                          file=sys.stdout)
                    print("Location of NR data     : %s " %
                          getattr(row, 'numrel_data'),
                          file=sys.stdout)
                #
                # Get out of the loop over the catalog if the 'longest' simulation is
                # not required
                if not use_longest_simulation: break

        if len(mflower) > 0 and len(rows) > 0:
            if verbose:
                print(" .. template matches %s simulations" % len(rows),
                      file=sys.stderr)
            is_map = True

        #################################################################
        # 4.2.1 If LOCATION is found, return it
        if is_map:
            # Multiple templates might have been found matching the input parameters
            # and we print out the location of each here
            if use_longest_simulation:
                if verbose:
                    print(
                        "Found the following simulations with same parameters as the template:\n",
                        file=sys.stderr)
                    for rr in rows:
                        print("\t", str(rr.numrel_data), file=sys.stderr)

            #################################################################
            # 4.2.1 Find the longest of all found matching simulations
            mflower = np.array(mflower)
            return_row = rows[np.where(mflower == mflower.min())[0][0]]
            return getattr(return_row, 'numrel_data')
        else:
            #################################################################
            # 4.2.2 If LOCATION is NOT found, throw an error!
            # print out the maximum difference between the template's and catalog's
            # parameters
            if verbose:
                print(
                    " .. template does not match any sim in the NR catalog table.\nParameters: ",
                    p,
                    file=sys.stdout)
            print("MIN ERROR for rejection..: %e\n" % np.array(errs).min())
            raise RuntimeError("NR data not found")

    #################################################################
    # 5. If none of the supported METHOD is used, throw an error!
    else:
        raise IOError("Method %s not supported.." % method)


################################################################################
# Return the re-scaled NR waveform
################################################################################
def get_hplus_hcross_from_sxs(hdf5_file_name, template_params, delta_t,\
                              modeLmin = 2, modeLmax = 8,\
                              taper=True, verbose=False, debug=False):
    if verbose:
        print(" \n\n\nIn get_hplus_hcross_from_sxs..", file=sys.stdout)
        sys.stdout.flush()
    #
    def get_param(value):
        try:
            if value == 'end_time':
                # FIXME: Imprecise!
                return float(template_params.get_end())
            else:
                return getattr(template_params, value)
        except:
            return template_params[value]

    #
    # Get relevant binary parameters
    #
    total_mass = get_param('mtotal')
    theta = get_param('inclination')
    try:
        phi = get_param('coa_phase')
    except:
        phi = 0
    distance = get_param('distance')
    end_time = get_param('end_time')  # FIXME
    f_lower = get_param('f_lower')

    try:
        taper = get_param('taper')
    except:
        pass
    try:
        verbose = get_param('verbose')
    except:
        pass

    if debug:
        print("mass, theta, phi , distance, end_time = ", \
                          total_mass, theta, phi, distance, end_time, file=sys.stdout)
        try:
            print("end_time = 0 (could be %f)" % template_params['end_time'],
                  file=sys.stdout)
        except:
            pass

    ###########################################################################
    # Read in the waveform from file & rescale it
    ###########################################################################
    #
    # Figure out how much memory to allocate
    estimated_length_pow2 = nextpow2(MAX_NR_LENGTH * total_mass * lal.MTSUN_SI)
    if debug:
        print("estimated length = ", estimated_length_pow2)

    idx = 0
    while True:
        # this loop is to make sure that an under-estimation of the waveform's
        # length does not lead to failure, and we scale up the length allocation
        # till the whole NR waveform fits in it (in powers of 2)
        try:
            print("ONLY RESCALING")
            if isinstance(hdf5_file_name, UseNRinDA.nr_wave):
                nrwav = hdf5_file_name
                if nrwav.dt != delta_t:
                    raise RuntimeError("sample_rate asked: %f, should be: %f" %\
                                          (1./delta_t, 1./nrwav.dt))
                nrwav.rescale_wave(total_mass, theta, phi, distance)
            elif os.path.exists(str(hdf5_file_name)):
                if debug:
                    print("\n \t>>try %d at reading waveform" % (idx + 1))
                    print(
                        "\n \t[More than ONE try is required if estimated NR length is too short]"
                    )
                idx += 1
                group_name = get_param("group_name")  # GROUP NAME
                nrwav = UseNRinDA.nr_wave(filename=hdf5_file_name,
                          sample_rate=1./delta_t, time_length=estimated_length_pow2, \
                          totalmass=total_mass, inclination=theta, phi=phi, \
                          modeLmin=modeLmin, modeLmax=modeLmax,\
                          distance=distance*1e6,\
                          ex_order=3, group_name=group_name,\
                          verbose=debug)
            else:
                print(hdf5_file_name)
                raise IOError("""
            ^^ First argument passed is neither a nr_wave class
            nor a file path. It must be one of two.""")
            break
        except RuntimeError as rerr:
            estimated_length_pow2 *= 2

    if debug and type(hdf5_file_name) == str:
        print("\t Waveform read from %s" % hdf5_file_name, file=sys.stdout)
        sys.stdout.flush()

    ###########################################################################
    ## Condition the waveform
    ##  This is done with a Planck taper window which smoothly joins the start
    ##  and the end of the waveform to zero. The width of such a window was
    ##  investigated in Chu, Fong, Kumar et al (2015). We hard-code their
    ##  pre-set tapering configuration.
    ###########################################################################

    upwin_t_start = 100
    upwin_t_width = 1000
    downwin_amp_frac = 0.01
    downwin_t_width = 50
    t_filter = upwin_t_start + upwin_t_width
    m_lower = nrwav.get_lowest_binary_mass(t_filter, f_lower)

    # Check if re-scaling to input total mass is allowed by the tapering choice
    if m_lower > total_mass:
        raise IOError("Can rescale down to %f Msun at %fHz, asked for %f Msun" % \
                              (m_lower, f_lower, total_mass))

    # Taper the waveform polarizations
    if taper:
        hp, hc = nrwav.taper_filter_waveform(ttaper1=upwin_t_start,\
                                              ttaper2=upwin_t_width,\
                                              ftaper3=downwin_amp_frac,\
                                              ttaper4=downwin_t_width,\
                                              verbose=debug)
        if debug:
            print("Tapering window [start1-2], [end-amplfraction - t-width]: [%.1f - %.1f], [%.5f - %.1f]" %\
                    (upwin_t_start, upwin_t_width+upwin_t_start, downwin_amp_frac, downwin_t_width))
    else:
        hp, hc = [nrwav.rescaled_hp, nrwav.rescaled_hc]

    #
    # Chop off trailing zeros in the waveform. These include zeros at the
    # END of the waveform ONLY. The START does NOT change.
    #
    i_filter = int(ceil(t_filter * total_mass * lal.MTSUN_SI / nrwav.dt))
    hpExtraIdx = where(hp.data[i_filter:] == 0)[0]
    hcExtraIdx = where(hc.data[i_filter:] == 0)[0]
    idx = i_filter + hpExtraIdx[where(hpExtraIdx == hcExtraIdx)[0][0]]
    #
    if debug:
        print(" \tIndex = %d where waveform ends" % idx, file=sys.stdout)
        print("\t Length of waveform BEFORE removing zeros = %d, %d" %
              (len(hp), len(hc)))
        sys.stdout.flush()

    # Actually remove the trailing zeros
    hp = hp[:idx]
    hc = hc[:idx]

    if debug:
        print("\t Length of waveform AFTER removing zeros = %d, %d" %
              (len(hp), len(hc)))
    #
    # Find the time at which the (2,2) mode's amplitude peaks, and use it to
    # set the epoch of the waveform being returned. Because trailing zeros have
    # been chopped off, that might alter the length of the simulation,
    # including spare zeros at the beginning. Therefore, we get the peak
    # amplitude AFTER this trimming.
    time_start_s = -nrwav.get_peak_amplitude(
        amplitude_from_polarizations(hp, hc))[0] * nrwav.dt

    if debug:
        print(" \t time_start_s = %f" % time_start_s, file=sys.stdout)
        sys.stdout.flush()

    #
    # Prepare the output polarization vectors, with the correct epoch set
    #
    hp = TimeSeries(hp.data, delta_t=delta_t,\
                    epoch=lal.LIGOTimeGPS(end_time+time_start_s))
    hc = TimeSeries(hc.data, delta_t=delta_t,\
                    epoch=lal.LIGOTimeGPS(end_time+time_start_s))
    #
    if debug:
        try:
            print("  Length of rescaled waveform = %f.." % idx * hp.delta_t,
                  file=sys.stdout)
            print(" hp.epoch = ", hp._epoch, file=sys.stdout)
            sys.stdout.flush()
        except:
            print(type(idx), type(hp))
    #
    if debug: print("Returning hp, hc")
    #
    return hp, hc, nrwav


################################################################################
# Wrapper function between 'get_td_waveform' and 'get_hplus_hcross_from_sxs'
################################################################################
def get_hplus_hcross_from_get_td_waveform(**p):
    """
    Interface between get_td_waveform and get_hplus_hcross_from_directory above
    """
    delta_t = float(p['delta_t'])
    p['end_time'] = 0.

    # Re-direct to sxs-format strain reading code
    if 'verbose' in p.keys(): verbose = p['verbose']
    else: verbose = False
    nr_data_location = get_nr_data_location(p,\
                              map_var='TemplateBankToNRMappings',\
                              method='tmplt_bank_map',\
                              verbose=verbose)
    fp = h5py.File(nr_data_location, 'r')
    # For now, if all groups in the hdf file are directories consider that as
    # sufficient evidence that this is a strain file
    if np.all(['.dir' in kk for kk in fp]):
        hp, hc = get_hplus_hcross_from_sxs(nr_data_location, p, delta_t)
        fp.close()
        return hp, hc
    else:
        raise IOError("This data file is not in SXS format..Abort..!")

    # Assign correct reference frequency for consistency:
    Mflower = fp.attrs['f_lower_at_1MSUN']
    fp.close()
    mass1 = p['mass1']
    mass2 = p['mass2']
    total_mass = mass1 + mass2
    p['f_ref'] = Mflower / (total_mass)
    print("The reference frequency has been set to %1.5f" % p['f_ref'])

    hp, hc = get_hplus_hcross_from_directory(p['numrel_data'], p, delta_t)
    return hp, hc
