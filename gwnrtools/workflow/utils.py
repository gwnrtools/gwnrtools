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

from pycbc.workflow import configuration


def get_ini_opts(confs, section):
    op_str = ""
    for opt in confs.options(section):
        val = confs.get(section, opt)
        op_str += "--" + opt + " " + val + " \\" + "\n"
    return op_str


class WorkflowConfigParserBase(configuration.InterpolatingConfigParser):
    def __init__(self, *args, **kwargs) -> None:
        super(WorkflowConfigParserBase, self).__init__(args, **kwargs)
