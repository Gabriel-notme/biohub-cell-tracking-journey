import json
from collections import Counter, defaultdict
import numpy as np
D = json.load(open('/workspace/cl/ideas/an/p13_cfg.json'))
rows = D['rows']; forks = D['forks']
def tab(name, key, rows=rows):
    C = defaultdict(Counter)
    for r in rows: C[key(r)][r['lab']] += 1
    print(name)
    for k in sorted(C, key=str):
        c = C[k]; print('   %-14s TP %6d FP %5d NE~ %7d  fp%% %.1f' % (k, c['TP'], c['FP'], c['NE'] * 33, 100 * c['FP'] / max(1, c['TP'] + c['FP'])))
tab('motion residual (um)', lambda r: -1 if r['res'] < 0 else min(int(r['res']), 14))
tab('spike of target (um), ab<4', lambda r: -1 if r['sp'] < 0 else min(int(r['sp']), 10), [r for r in rows if 0 <= r['ab'] < 4])
for emb in ['44b6', '6bba']:
    tab('motion residual >=6 by emb ' + emb, lambda r: min(int(r['res']), 14) // 2 * 2, [r for r in rows if r['m'].startswith(emb) and r['res'] >= 6])
print('forks', Counter(f['lab'] for f in forks))
def stub(f):
    a, b = sorted(f['bl'], key=lambda z: z[0])
    return ('min%d' % min(a[0], 6), 'endfork' if a[1] == 2 else 'end')
C = defaultdict(Counter)
for f in forks: C[stub(f)][f['lab']] += 1
for k in sorted(C): print('  ', k, dict(C[k]))
C = defaultdict(Counter)
for f in forks: C[min(int(f['cd']), 16) // 2 * 2][f['lab']] += 1
print('child distance'); [print('  ', k, dict(C[k])) for k in sorted(C)]
