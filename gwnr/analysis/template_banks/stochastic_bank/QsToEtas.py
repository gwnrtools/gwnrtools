#!/usr/bin/env python
from numpy import *
import sys

file = open(sys.argv[1], 'r')
qs = loadtxt(file)

out = open('FinalEtas.dat', 'w+')
for q in qs:
    out.write('%12.12f\n' % (q / (1. + q)**2))

out.close()
