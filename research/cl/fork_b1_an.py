import json, glob
from collections import Counter
R = []
for f in glob.glob('/workspace/cl/forkb1/*.json'):
    s = f.split('/')[-1].split('__')[0]
    for r in json.load(open(f)): r['set'] = s; R.append(r)
print('forks', len(R), Counter((r['prov'], r['lab']) for r in R))
BINS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01]
for grp, sets in [('b1-clean (hold36+prev4)', ('hold36', 'prev4')), ('b1-train (t127+audit32)', ('t127a', 't127b', 'audit32'))]:
    for prov in ['b5', 'dc']:
        line = '%-24s %s:' % (grp, prov)
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            c = Counter(r['lab'] for r in R if r['set'] in sets and r['prov'] == prov and lo <= r['b1'] < hi)
            line += ' [%.2f,%.2f) TP%d FP%d U%d |' % (lo, hi, c['TP'], c['FP'], c['U'])
        print(line)
