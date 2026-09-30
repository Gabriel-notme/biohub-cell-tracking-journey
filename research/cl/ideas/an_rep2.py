import json, sys
from collections import Counter, defaultdict
import numpy as np
R = json.load(open('/workspace/cl/ideas/an/p13_edges.json'))
rows = [r for M in R for r in M['rows']]; fns = [f for M in R for f in M['fns']]
fp = [r for r in rows if r['lab'] == 'FP' and r['fp'] == 'src_ok:tgt_unm']
c = Counter()
for r in fp:
    sd = min(r['succ_dist']); sp = r['succ_pred']
    k1 = 'sd<7' if sd < 7 else ('sd7-9' if sd < 9 else ('sd9-12' if sd < 12 else 'sd>12'))
    if all(q is None for q in sp): k2 = 'succ_unmatched'
    else:
        q = [q for q in sp if q is not None][0]
        k2 = 'succ_matched_' + ('orphan' if q[2] else 'hasparent')
    c[(k1, k2)] += 1
for k in sorted(c): print(k, c[k])
print('dend for these', Counter(r['dend'] for r in fp), 'dst type', Counter(r['dnt'] for r in fp))
print('--- FN unmatched endpoints: nearest pred node dist and whether it is matched elsewhere')
c = Counter()
for f in fns:
    for tag in 'uv':
        if tag + '_nd' in f:
            d = f[tag + '_nd']; k = 'd<7' if d < 7 else ('d7-8' if d < 8 else ('d8-10' if d < 10 else ('d10-14' if d < 14 else 'd>14')))
            c[(k, f[tag + '_nn_matched'])] += 1
for k in sorted(c): print(k, c[k])
