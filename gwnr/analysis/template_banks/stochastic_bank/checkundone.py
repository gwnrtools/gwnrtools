#!/usr/bin/env python
import os

dagfile = open('do_failed_combine_dag.dag', 'w+')

for i in range(4001):
    if os.path.exists('match-part/match%dpart0.dat' %
                      i) and not os.path.exists('match/match%d.dat' % i):
        dagfile.write('Job RECOM%d match_combine.submit\n' % i)
        dagfile.write('VARS RECOM%d JOB_NUM=\"%d\"\n' % (i, i))
        dagfile.write('PRIORITY RECOM%d 10000\n\n' % i)

dagfile.close()
