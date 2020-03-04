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
import subprocess
import pycbc.catalog


def mkdir(dir_name):
    try:
        subprocess.call(["mkdir", "-p", dir_name])
    except OSError:
        pass


class Merger(pycbc.catalog.Merger):
    """Informaton about a specific compact binary merger"""

    def __init__(self, name, **kwargs):
        super(Merger, self).__init__(name, **kwargs)

    def operating_ifos(self):
        return self.data['files']['OperatingIFOs'].split()

    def gpstime(self):
        return self.data['tc']['best']

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
                "Data is not available at sample rate {}Hz. Resampling is not supported yet.".format(sample_rate))

        url = self.data['files'][ifo][length][sampling]['GWF']
        filename = download_file(url, cache=False)
        local_filename = url.split('/')[-1]

        if save_dir != '.' and save_dir != '':
            mkdir(save_dir)
            local_filename = os.path.join(save_dir, local_filename)

        subprocess.call('mv {0} {1}'.format(filename, local_filename).split())

        return local_filename

    def frame_data_name(self, ifo, duration, sample_rate):
        """ Get the name of frame data file using pycbc.catalog API
        """
        length = "{}sec".format(duration)
        if sample_rate == 4096:
            sampling = "4KHz"
        elif sample_rate == 16384:
            sampling = "16KHz"
        else:
            raise IOError(
                "Data is not available at sample rate {}Hz. Resampling is not supported yet.".format(sample_rate))
        return self.data['files'][ifo][length][sampling]['GWF'].split('/')[-1]

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
                "Data is not available at sample rate {}Hz. Resampling is not supported yet.".format(sample_rate))
        return "{}:GWOSC-{}_R1_STRAIN".format(ifo, sampling.upper())


class Catalog(pycbc.catalog.Catalog):
    def __init__(self, **kwargs):
        super(Catalog, self).__init__(**kwargs)


# Names of available events
c = pycbc.catalog.Catalog()
catalog_events = c.names
