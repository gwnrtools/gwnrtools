# Copyright (C) 2020 Prayush Kumar
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
import subprocess
import pycbc.catalog

gwtc1_psd_url_template = "https://dcc.ligo.org/public/0158/P1900011/001/GWTC1_{0}_PSDs.dat"


def get_psd_url(source, name):
    """Get the source psd for a particular GW catalog
    """
    if source == 'gwtc-1':
        fname = gwtc1_psd_url_template.format(name)
    else:
        raise ValueError('Unkown catalog source {}'.format(source))
    return fname


def mkdir(dir_name):
    try:
        subprocess.call(["mkdir", "-p", dir_name])
    except OSError:
        pass


def rmfile(filename):
    try:
        subprocess.call(["rm", "-rf", filename])
    except OSError:
        pass


class Merger(pycbc.catalog.Merger):
    """Informaton about a specific compact binary merger"""
    def __init__(self, name, source='gwtc-1', **kwargs):
        super(Merger, self).__init__(name, source, **kwargs)
        self.psd_url = get_psd_url(source, name)

    def operating_ifos(self, ignore_ifos=['G1']):
        ifos = self.data['files']['OperatingIFOs'].split()
        # Explicitly ignore IFOs
        for ifo in ignore_ifos:
            if ifo in ifos:
                ifos.remove(ifo)
        return ifos

    def gpstime(self):
        return self.data['tc']['best']

    def frame_data_url(self, ifo, duration=32, sample_rate=4096):
        length = "{}sec".format(duration)
        if sample_rate == 4096:
            sampling = "4KHz"
        elif sample_rate == 16384:
            sampling = "16KHz"
        else:
            raise IOError(
                "Data is not available at sample rate {}Hz. Resampling is not supported yet."
                .format(sample_rate))

        return self.data['files'][ifo][length][sampling]['GWF']

    def frame_data_name(self, ifo, duration, sample_rate):
        """ Get the name of frame data file using pycbc.catalog API
        """
        return self.frame_data_url(ifo,
                                   duration=duration,
                                   sample_rate=sample_rate).split('/')[-1]

    def fetch_data(self, ifo, duration=32, sample_rate=4096, save_dir=''):
        """ Download strain file around the event

        By default this will return the strain around the event in the smallest
        format available. Selection of other data requires options to be set.

        Parameters
        ----------
        ifo: str
            The name of the observatory you want strain for. Ex. H1, L1, V1
        duration: int
            Number of seconds for which data is to be downloaded. E.g. 32, 4096
        sample_rate: int
            Sampling rate at which data is to be downloaded. E.g. 4096, 16384

        Returns
        -------
        filename: str
            Name of file in current directory containing strain around
            the event.
        """
        import os
        import subprocess
        from astropy.utils.data import download_file

        length = "{}sec".format(duration)
        if sample_rate == 4096:
            sampling = "4KHz"
        elif sample_rate == 16384:
            sampling = "16KHz"
        else:
            raise IOError(
                "Data is not available at sample rate {}Hz. Resampling is not supported yet."
                .format(sample_rate))

        url = self.frame_data_url(ifo,
                                  duration=duration,
                                  sample_rate=sample_rate)
        filename = download_file(url, cache=False)
        local_filename = self.frame_data_name(ifo,
                                              duration=duration,
                                              sample_rate=sample_rate)

        if save_dir != '.' and save_dir != '':
            mkdir(save_dir)
            local_filename = os.path.join(save_dir, local_filename)

        subprocess.call('mv {0} {1}'.format(filename, local_filename).split())

        return local_filename

    def channel_name(self, ifo, sample_rate):
        """ Get the channel name in data

        Currently this is hard-coded to the regex followed at
        https://www.gw-openscience.org/catalog/GWTC-1-confident/

        and does not use the catalog 
        """
        if sample_rate == 4096:
            sampling = "4KHz"
        elif sample_rate == 16384:
            sampling = "16KHz"
        else:
            raise IOError(
                "Data is not available at sample rate {}Hz. Resampling is not supported yet."
                .format(sample_rate))
        return "{}:GWOSC-{}_R1_STRAIN".format(ifo, sampling.upper())

    def psd_file_name(self, ifo):
        return 'psd_{0}.dat'.format(ifo)

    def fetch_psds(self, duration=32, sample_rate=4096, save_dir=''):
        """Download PSD file around the event

        This will return the PSD around the event and interpolate
        to delta_f = 1 / duration provided.

        By default it returns PSDs for all operating IFOs
        (since LIGO DCC provides a single PSD file for all IFOs)

        By default it uses a linear spline, which performs best under
        high-frequency noise in data-based PSDs

        Parameters
        ----------
        duration: int
            Number of seconds of data being analyzed
        sample_rate: int
            Sampling rate at which data is to be processed using this PSD
            This sets the Nyquist frequency for extrapolation

        Returns
        -------
        filenames: dict
            Name of PSD file in save directory for each operating IFO.
        """
        import os
        import numpy
        from astropy.utils.data import download_file
        from gwnr.analysis.psd import resample_and_extrapolate_psd

        filename = download_file(self.psd_url, cache=False)
        all_psds = numpy.loadtxt(filename)
        rmfile(filename)

        # extract frequency samples
        freq_vals = all_psds[:, 0]

        filenames = {}
        for idx, ifo in enumerate(self.operating_ifos()):
            # get file name for the PSD
            local_filename = self.psd_file_name(ifo)
            if save_dir != '.' and save_dir != '':
                mkdir(save_dir)
                local_filename = os.path.join(save_dir, local_filename)
            filenames[ifo] = local_filename

            # We shamelessly assume that the operating_ifos() returns
            # IFO names in the same ORDER as the columns in the PSD file
            psd_vals = all_psds[:, idx + 1]

            # Interpolate to desired delta_f
            resampled_psd = resample_and_extrapolate_psd(
                freq_vals, psd_vals, 1. / duration, sample_rate / 2)

            # Write interpolated PSD to disk
            numpy.savetxt(local_filename,
                          numpy.column_stack([
                              resampled_psd.sample_frequencies,
                              resampled_psd.data
                          ]),
                          fmt='%.12e',
                          header='Frequency(Hz)  PSD')

        return filenames


class Catalog(pycbc.catalog.Catalog):
    def __init__(self, **kwargs):
        super(Catalog, self).__init__(**kwargs)


# Names of available events
c = pycbc.catalog.Catalog()
catalog_events = c.names
