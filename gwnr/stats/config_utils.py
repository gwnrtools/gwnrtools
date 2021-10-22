# Copyright (C) 2021 Prayush Kumar
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


class ConfigWriter():
    def __init__(self, configs, run_dir):
        '''
Writer class for configuration files

Parameters
----------

name : string
    The name that configuration file will be written to. 
    This does not depend on the available options 
configs : dict
    Has key:value pairs for different ini file texts
run_dir : string
    Run directory where the configuration files are to be written
        '''
        self.configs = configs
        self.run_dir = run_dir

    def write(self, name, **formatting_kwargs):
        '''
        Config file string may have some blanks that need to 
        be filled, especially for data configs for GW events.
        '''
        if name not in self.configs:
            print("Provided name {} not available. Available: {}".format(
                name, self.types()))
            return

        out_str = self.configs[name]
        with open(os.path.join(self.run_dir, name + '.ini'), 'w') as fout:
            if len(formatting_kwargs) > 0:
                fout.write(out_str.format(**formatting_kwargs))
            else:
                fout.write(out_str)

    def types(self):
        return list(self.configs.keys())


class ConfigBase():
    '''
    Class to store config file samples for categories, as dict of dicts
    '''
    def __init__(self, run_dir, configs={}) -> None:
        self.run_dir = run_dir
        assert (isinstance(configs, dict))
        self.configs = configs

    def get_run_dir(self):
        return self.run_dir

    @property
    def config_names(self):
        return list(self.configs.keys())

    def available_configs(self):
        return self.config_names

    def update_config_writers(self):
        # Initialize their config writers
        if not hasattr(self, '_config_writers'):
            self._config_writers = {}
        for config_name in self.config_names:
            self._config_writers[config_name] = ConfigWriter(
                self.configs[config_name], self.get_run_dir())

    @property
    def config_writers(self):
        # Initialize their config writers
        if not hasattr(self, '_config_writers'):
            self.update_config_writers()
        return self._config_writers

    def get_config_writer(self, name):
        assert (name in self.available_configs())
        return self.config_writers[name]

    def get(self, config_name, type_name=None):
        if type_name in self.configs[config_name]:
            return self.configs[config_name][type_name]
        return self.configs[config_name]

    def set(self, config_name, type_name, config):
        if config_name not in self.configs:
            self.configs[config_name] = {}
        self.configs[config_name][type_name] = config
