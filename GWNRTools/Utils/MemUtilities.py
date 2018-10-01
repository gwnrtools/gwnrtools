#!/usr/bin/env python

########################################
## IMPORTS
########################################
import os, sys
from collections import Mapping, Container
from sys import getsizeof
import glob, commands as cmd
#import numpy as np
#import re
sys.path.append('/home/prayush/src/UseNRinDA/NR/SpEC/src/')
#from SpECUtilities import ParseHeaderForSpECTabularOutputASCII


########################################
## Other FUNCTIONS
########################################


########################################
## USER-FACING FUNCTIONS
########################################
def TotalMemoryUsage(o, ids=set()):
    """Find the memory footprint of a Python object

    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.

    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.

    :param o: the object
    :param ids:
    :return:
    """
    #{{{
    d = TotalMemoryUsage
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, unicode):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r
    #}}}

def ShowMemoryUsage(objs=[], prefac=1e-6, prefac_name='Mb'):
    """
This is a wrapper around TotalMemoryUsage that takes in a list
of arbitrary objects, and prints their total size
    """
    #{{{
    if type(objs) is not list: objs = [objs]

    mem = 0.0
    for obj in objs: mem += TotalMemoryUsage(obj, set())

    print >>sys.stdout,\
          "Memory added: %.3f %s" % (mem * prefac, prefac_name)

    sys.stdout.flush()
    return
    #}}}

show_memory_increase = ShowMemoryUsage
