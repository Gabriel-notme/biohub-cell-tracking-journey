import json, sys
from collections import Counter, defaultdict
import numpy as np
R = json.load(open('/workspace/cl/ideas/up_rows.json'))
dev = np.array([r['dev'] for r in R])
print('id-coord deviation (um) mean-of-means %.3f max %.3f; matched-id frac %.4f' % (dev[:, 0].mean(), dev[:, 1].max(), dev[:, 2].sum() / dev[:, 3].sum()))
EMB = ['44b6', '6bba']
def grp(r): return r['movie'][:4]
def clean(r): return r['set'] in ('hold36', 'prev4')
def tab(key, filt=lambda x: True, title=''):
    print('==', title)
    C = defaultdict(Counter)
    for r in R:
        for x in r['rows']:
            if not filt(x): continue
            k = key(x)
            C[k][(grp(r), x['lab'])] += 1
            C[k][('clean', x['lab'])] += int(clean(r))
    for k in sorted(C, key=lambda k: -sum(C[k].values())):
        c = C[k]; line = '%-40s' % str(k)
        for g in EMB + ['clean']:
            tp, fp, u = c[(g, 'TP')], c[(g, 'FP')], c[(g, 'U')]
            line += ' | %s TP %6d FP %5d U %6d P %.3f' % (g, tp, fp, u, tp / max(1, tp + fp))
        print(line)
tab(lambda x: x['cat'], title='edge candidate status')
tab(lambda x: (x['cat'], x['et']), title='status x edge type')
def pb(p):
    if p is None: return 'None'
    return '%.1f' % (np.floor(p * 10) / 10)
tab(lambda x: ('agree', pb(x['fp'])), lambda x: x['cat'] == 'agree', title='agree by fullgraph prob')
tab(lambda x: (x['cat'], 'alt_par_prob', pb(x['fp_alt_par']), 'cp_end', x['cp_end']), lambda x: x['cat'] in ('other_in_p13',), title='other_in_p13 by alt parent prob / alt parent is a track end')
tab(lambda x: (x['cat'], 'u_alt_free', min(x['u_alt_ch_free'], 1), 'cp_end', x['cp_end']), lambda x: x['cat'] in ('other_in_p13', 'nocand', 'other_dropped'), title='non-agree x u has a free alt cand child')
print('== forks')
F = defaultdict(Counter)
for r in R:
    for f in r['forks']:
        F[tuple(f['st'])][(grp(r), f['lab'])] += 1
        F[tuple(f['st'])][('clean', f['lab'])] += int(clean(r))
for k in sorted(F, key=lambda k: -sum(F[k].values())):
    c = F[k]; line = '%-40s' % str(k)
    for g in EMB + ['clean']:
        line += ' | %s TP %3d FP %3d U %4d' % (g, c[(g, 'TP')], c[(g, 'FP')], c[(g, 'U')])
    print(line)
F = defaultdict(Counter)
for r in R:
    for f in r['forks']:
        F[tuple(f['et'])][(grp(r), f['lab'])] += 1
for k in sorted(F, key=lambda k: -sum(F[k].values())):
    c = F[k]; print('%-50s' % str(k), ' '.join('%s TP %3d FP %3d U %4d |' % (g, c[(g, 'TP')], c[(g, 'FP')], c[(g, 'U')]) for g in EMB))
