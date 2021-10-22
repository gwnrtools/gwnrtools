from __future__ import absolute_import

from .memory import *
from .support import *
from .types import *


def get_unique_hex_tag(N=1, num_digits=10):
    import random
    if N == 1:
        return '%0{}x'.format(num_digits) % random.randrange(16**num_digits)
    else:
        return [
            '%0{}x'.format(num_digits) % random.randrange(16**num_digits)
            for i in range(N)
        ]


def get_sim_hash(N=1, num_digits=10):
    return ilwd.ilwdchar(":{}:0".format(
        get_unique_hex_tag(N=N, num_digits=num_digits)))
