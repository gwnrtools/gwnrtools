#!/usr/bin/env python
import CosmoUtilities
import SupportFunctions
import LALInferenceUtilities
from optparse import OptionParser
import numpy as np
import os
import sys
import time
_itime = time.time()

sys.path.append('/home/prayush/src/nsbh_tidal/emcee_PE/src/')
sys.path.append('/home/prayush/src/nsbh_tidal/emcee_PE/scripts/')
sys.path.append('/home/prayush/src/UseNRinDA/Scripts/')
sys.path.append('/home/prayush/src/UseNRinDA/Scripts/Utils/')
sys.path.append('/home/prayush/src/UseNRinDA/Scripts/SpEC/')
sys.path.append('/home/prayush/src/UseNRinDA/Scripts/Bayesian/')
sys.path.append('/home/prayush/src/UseNRinDA/Scripts/DataAnalysis/')
sys.path.append('/home/prayush/src/UseNRinDA/Scripts/Waveforms/')
sys.path.append('/home/prayush/src/UseNRinDA/Scripts/Plotting/')
sys.path.append('/home/prayush/src/UseNRinDA/Scripts/Plotting/Bayesian/')

LU = LALInferenceUtilities
SF = SupportFunctions
CU = CosmoUtilities

__author__ = "Prayush Kumar <prayush@astro.cornell.edu>"

PROGRAM_NAME = os.path.abspath(sys.argv[0])

##################################################
### option parsing ###
##################################################
parser = OptionParser(
    usage="%prog [OPTIONS]",
    description=
    "Takes posterior_samples.dat from LI and adds (redshift, mass_source) columns."
)

parser.add_option("-p",
                  "--input-posterior",
                  metavar='FILE',
                  help='posterior samples to process',
                  default='posterior_samples.dat')
parser.add_option("-o",
                  "--output-posterior",
                  metavar='FILE',
                  help="Output file to be written")
parser.add_option("-V",
                  "--verbose",
                  action="store_true",
                  help="print extra debugging information",
                  default=False)

options, argv_frame_files = parser.parse_args()
if not options.input_posterior or not options.output_posterior:
    raise IOError("Both INPUT and OUTPUT file names are REQUIRED.")
print("Processing ", options.input_posterior, " to ", options.output_posterior)


##################################################
##################################################
def AddRedshiftAndSourceFrameParamsToPosterior(in_file, out_file):
    if not os.path.exists(in_file):
        raise IOError("File %s not found.." % in_file)
    Headers, Data = \
        LU.get_header_data_from_posterior_samples_file(
            in_file, no_of_samples=-1)

    ##

    def AppendQuantityToPosterior(Data, Headers, newQ, newQName):
        Headers.append(newQName)
        if options.verbose:
            print("Extending Data, shape: ", np.shape(Data))
        Data = np.append(Data, np.array([newQ]).T, axis=1)
        if options.verbose:
            print("New Data, shape: ", np.shape(Data))
        return Data, Headers

    # redshift
    d_array = Data[:, LU.get_param_idx('dist', Headers)]
    z_array = CU.calculate_redshift(d_array)
    Data, Headers = AppendQuantityToPosterior(Data, Headers, z_array,
                                              'redshift')
    # Append source frame mass parameters
    for param in ['m1', 'm2', 'mc', 'mtotal', 'mf']:
        if options.verbose:
            print("Computing and appending: %s" % param)
        m_array = Data[:, LU.get_param_idx(param, Headers)]
        ms_array = CU.detector_to_source_frame(m_array, z_array)
        Data, Headers = AppendQuantityToPosterior(Data, Headers, ms_array,
                                                  '%s_source' % param)
    # Write File
    LU.write_posterior_samples_file(out_file, Headers, Data)
    return


##################################################
in_filename = options.input_posterior
out_filename = options.output_posterior


def main():
    AddRedshiftAndSourceFrameParamsToPosterior(in_filename, out_filename)


if __name__ == "__main__":
    main()
    print("Total time taken: %.2f seconds" % (time.time() - _itime))
