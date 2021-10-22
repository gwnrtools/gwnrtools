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

import os
import subprocess
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py

try:
    import celluloid
except ImportError:
    pass

############################################################################


class HandleSpectreVolumeDatum(object):
    def __init__(self,
                 volume_data_file='',
                 name='',
                 dt=0.5,
                 read_fields=['Psi'],
                 xdmf_converter=None,
                 verbose=True):
        assert os.path.exists(volume_data_file),\
            "Cannot find data file: {0:s}".format(volume_data_file)
        self.verbose = verbose
        self.name = name
        self.volume_data_file = volume_data_file
        self.xdmf_converter = xdmf_converter
        self.plotting_funcs = {
            'linlin': 'plot',
            'loglin': 'semilogx',
            'loglog': 'loglog',
            'linlog': 'semilogy'
        }
        self.linestyles = ['-', '--', '--', '-.', ':']
        self.linecolors = ['r', 'g', 'b', 'k', 'm', 'y']

        self.read_fields = read_fields
        self.data = self.read_data()

        for f in self.data[list(self.data.keys())[0]]:
            if 'InertialCoordinate' in f:
                self.read_fields.append(f)

        self.times, self.fields = self.get_data(read_fields)
        self.times, self.fields = self.downsample_fields_data(self.times,
                                                              self.fields,
                                                              dt=dt)

    def read_data(self):
        logging.info("Reading in: {0:s}".format(self.volume_data_file))
        fp = h5py.File(self.volume_data_file, 'r')
        self.data = fp['element_data.vol']
        logging.info(".. read in a dataset with shape: {}".format(
            np.shape(self.data)))
        return self.data

    def available_fields(self):
        return list(self.data[list(self.data.keys())[0]].keys())

    def get_data(self, fields=['Psi']):
        vol_dataset = self.data
        times = []
        field_data = {f: {} for f in fields}
        for obs_id in vol_dataset:
            obs_data = vol_dataset[obs_id]
            t = obs_data.attrs['observation_value']
            times.append(t)
            for f in fields:
                field_data[f][t] = obs_data[f][()]
        return np.sort(times), field_data

    def get_dt(self, times):
        dt_vals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        return dt_vals

    def downsample_fields_data(self, times, fields_data, dt=0.01):
        # Select times in sel_times from times
        dt_vals = self.get_dt(self.times)
        current_dt = np.median(dt_vals)
        assert dt >= current_dt,\
            "Requested dt = {0:.4e} is not possible (MIN: {1:.4e})".format(
                dt, current_dt)
        downsample_ratio = int(np.round(dt / current_dt))
        logging.info("Downsample by {}x".format(downsample_ratio))
        if downsample_ratio <= 1:
            return times, fields_data
        sel_times = [times[i] for i in range(0, len(times), downsample_ratio)]
        # Get data at sel_times
        new_fields_data = {f: {} for f in fields_data}
        for t in sel_times:
            for f in fields_data:
                new_fields_data[f][t] = fields_data[f][t]
        return sel_times, new_fields_data

    def fields_from_spectre_data(self, field_names):
        '''
Takes in a set of fields as a dictionary, as well as a list of times,
and separates out requested fields as a function of time.

Input:
------
times       : list, series of evolution times
f_fields    : dict, maps spectre-names of coords to lists of the same
                   coordinate as a function of evolution time
field_names : list, list of spectre-names of desired fields

Output:
--------
result : list, set of arrays of all spatial fields as a function of
         input times
        '''
        times = self.times
        f_fields = self.fields

        if len(field_names) == 1:
            fname = field_names[0]
            return np.array([f_fields[fname][t] for t in times])

        return [
            np.array([f_fields[fname][t] for t in times])
            for fname in field_names
        ]

    def coords_from_spectre_data(self,
                                 dim_to_coord_map=[
                                     'InertialCoordinates_x',
                                     'InertialCoordinates_y'
                                 ]):
        '''
Takes in a set of fields as a dictionary, as well as a list of times,
and separates out spatial coordinates as a function of time.

Input:
------
times            : list, series of evolution times
f_fields         : dict, maps spectre-names of coords to lists of the same
                   coordinate as a function of evolution time
dim_to_coord_map : list, ordered list of spectre-names for spatial coords

Output:
--------
result : list, set of arrays of all spatial coords as a function of
         input times
        '''
        return self.fields_from_spectre_data(dim_to_coord_map)

    def convert_to_xdmf(self, conversion_bin=None):
        if conversion_bin and os.path.exists(conversion_bin):
            exe = conversion_bin
        else:
            exe = self.xdmf_converter
        assert os.path.exists(exe) and os.path.getsize(exe) > 0,\
            "Xdmf converter utility {} not found!".format(exe)

        os.chdir(os.path.dirname(self.volume_data_file))
        o = subprocess.getoutput(
            'python3 {} --file-prefix {} --output {}'.format(
                exe,
                os.path.split(self.volume_data_file)[-1].strip('.h5'),
                os.path.split(self.volume_data_file)[-1].strip('.h5')))

        if self.verbose:
            logging.info(o)

    def make_movie(self,
                   field_name,
                   dt=None,
                   cmin=-1.,
                   cmax=1.,
                   ncolors=10,
                   name="movie.mp4",
                   **kwargs):
        '''
Make a movie

Input:
------
times      : list, series of evolution times
all_fields : dict, maps spectre-names of coords to lists of the same
                   coordinate as a function of evolution time
field_name : str, name of field to plot
dt         : time step
kwargs     : keyword arguments that are passed to `ArtistAnimation`

Output:
-------
result : tuple, (camera object, animation object) that can be used to
                display or save movie animation

        '''
        times = self.times
        all_fields = self.fields

        # Data for frames
        x_of_t, y_of_t = self.coords_from_spectre_data()
        z_of_t = self.fields_from_spectre_data([field_name])

        # Prepare figure and initialize Camera
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])
        ax2 = fig.add_axes([0.8, 0.1, 0.03, 0.8])
        camera = celluloid.Camera(fig)

        # Prepare a colorbar
        cmap = matplotlib.cm.jet
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'CustomCmap', cmaplist, cmap.N)

        cbar_bounds = np.linspace(cmin, cmax, ncolors + 1)
        norm = matplotlib.colors.BoundaryNorm(cbar_bounds, cmap.N)

        # Make all frames
        for idx in range(len(times)):
            if idx % 5 == 0:
                logging.info(" ... making frame {0:d}".format(idx + 1))

            t = times[idx]
            x, y, z = x_of_t[idx, :], y_of_t[idx, :], z_of_t[idx, :]

            # Draw a frame: FIXME: Use `contourf`
            sc = ax.scatter(x,
                            y,
                            c=z,
                            s=50,
                            alpha=0.97,
                            marker="s",
                            cmap=cmap,
                            norm=norm,
                            edgecolors='none',
                            vmin=cmin,
                            vmax=cmax)
            tx = ax.text(min(x),
                         max(y) + 0.05 * (max(y) - min(y)),
                         'Time: {0:06.03f}'.format(t))

            if idx >= 0:
                cb = matplotlib.colorbar.ColorbarBase(ax2,
                                                      cmap=cmap,
                                                      norm=norm,
                                                      spacing='proportional',
                                                      ticks=cbar_bounds,
                                                      boundaries=cbar_bounds,
                                                      format='%3.1f')
                ax2.set_ylabel(field_name)
                cb.set_clim(cmin, cmax)

            camera.snap()

        # Animate
        anim = camera.animate(**kwargs)
        anim.save(name)


class HandleSpectreVolumeData(object):
    def __init__(self, reduction_data_files=[]):
        assert len(reduction_data_files) > 0,\
            "Please provide at least one reduction data file!"
        for f in reduction_data_files:
            assert os.path.exists(f) and os.path.getsize(f) > 0,\
                "Data file: {0:s} not found / is empty!".format(f)
        self.handler = {}
        for f in reduction_data_files:
            self.handler[f] = HandleSpectreVolumeDatum(f)

    def handlers(self):
        return self.handler

    def make_movie(self, **kwargs):
        for f in self.handler:
            self.handler[f].make_movie(**kwargs)
