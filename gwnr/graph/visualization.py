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

import os

try:
    pass
except ImportError:
    pass

############################################################################


def play_movie(m):
    import subprocess
    subprocess.call(['mpv', os.path.join(base_dir, m)])


def embed_video(fname, mimetype):
    from IPython.display import HTML
    from codecs import encode
    video_encoded = encode(open(fname, "rb").read(), "base64")
    video_tag = '<video controls alt="test" src="data:video/{0};base64,{1}">'.format(
        mimetype, video_encoded)
    return HTML(data=video_tag)
