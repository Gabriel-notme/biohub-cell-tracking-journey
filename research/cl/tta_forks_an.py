import json, glob
from collections import Counter
R = []
for f in glob.glob('/workspace/cl/ttaf/*.json'):
    s = f.split('/')[-1].split('__')[0]
    for r in json.load(open(f)): r['set'] = s; R.append(r)
print('forks', len(R), Counter((r['dc'], r['lab']) for r in R))
for grp, sets in [('clean', ('hold36', 'prev4')), ('train', ('t127a', 't127b', 'audit32'))]:
    for dc in [True, False]:
        Q = [r for r in R if r['set'] in sets and r['dc'] == dc]
        for key, th in [('tta', 0.9), ('tta', 0.8), ('tta', 0.5), ('tmin', 0.5), ('tmin', 0.2)]:
            c = Counter(r['lab'] for r in Q if r[key] < th)
            print('%-5s %-9s %s<%.1f would drop: TP %d FP %d U %d  (of TP %d FP %d U %d)' % (grp, 'dc-added' if dc else 'b5', key, th, c['TP'], c['FP'], c['U'],
                  sum(r['lab'] == 'TP' for r in Q), sum(r['lab'] == 'FP' for r in Q), sum(r['lab'] == 'U' for r in Q)))
