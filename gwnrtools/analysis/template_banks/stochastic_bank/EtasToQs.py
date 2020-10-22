#!/usr/bin/env python
from numpy import *
import sys

file = open(sys.argv[1], 'r')
etas = loadtxt(file)

out = open('FinalQs.dat', 'w+')
for eta in etas:
    onebyeta = 1. / eta
    q = (onebyeta - 2) + sqrt(onebyeta**2 - 4. * onebyeta)
    q /= 2.
    out.write('%f\n' % q)

out.close()
