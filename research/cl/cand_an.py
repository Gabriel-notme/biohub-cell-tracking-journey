import json, glob, sys
import numpy as np
from collections import Counter
tag = sys.argv[1] if len(sys.argv) > 1 else 'cl2'
R = []
for f in glob.glob('/workspace/cl/%s/*/*.json' % tag):
    s = f.split('/')[-2]
    for r in json.load(open(f)): r['set'] = s; R.append(r)
print('rows', len(R))
for typ in ['start', 'stolen']:
    Q = [r for r in R if r['typ'] == typ and r['lab'] != 'U']
    print(typ, Counter(r['lab'] for r in Q), 'U', sum(1 for r in R if r['typ'] == typ and r['lab'] == 'U'))
Q = [r for r in R if r['typ'] == 'stolen' and r['lab'] != 'U']
print('--- stolen: q_hist bins')
for lo, hi in [(1, 1), (2, 3), (4, 6), (7, 10), (11, 20), (21, 40)]:
    k = [r for r in Q if lo <= r['q_hist'] <= hi]
    c = Counter(r['lab'] for r in k); print('  q_hist %2d-%2d P %3d D %3d N %5d' % (lo, hi, c['P'], c['D'], c['N']))
print('--- stolen with q_hist<=10: qstart_dp bins')
for lo, hi in [(0, 3), (3, 5), (5, 7), (7, 10), (10, 99)]:
    k = [r for r in Q if r['q_hist'] <= 10 and lo <= r['qstart_dp'] < hi]
    c = Counter(r['lab'] for r in k); print('  qstart_dp %2d-%2d P %3d D %3d N %5d' % (lo, hi, c['P'], c['D'], c['N']))
print('--- stolen q_root (q-track began without parent)')
for v in [0, 1]:
    c = Counter(r['lab'] for r in Q if r['q_root'] == v); print('  q_root', v, dict(c))
S = [r for r in R if r['typ'] == 'start' and r['lab'] != 'U']
print('--- start: b_fut bins')
for lo, hi in [(2, 3), (4, 9), (10, 19), (20, 40)]:
    c = Counter(r['lab'] for r in S if lo <= r['b_fut'] <= hi); print('  b_fut %2d-%2d P %3d D %3d N %5d' % (lo, hi, c['P'], c['D'], c['N']))
print('--- start: d_pb bins')
for lo, hi in [(0, 5), (5, 7), (7, 9), (9, 11), (11, 13)]:
    c = Counter(r['lab'] for r in S if lo <= r['d_pb'] < hi); print('  d_pb %2d-%2d P %3d D %3d N %5d' % (lo, hi, c['P'], c['D'], c['N']))
