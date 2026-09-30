import json, sys
from collections import Counter, defaultdict
import numpy as np
R = json.load(open('/workspace/cl/ideas/an/p13_edges.json'))
rows = [r for M in R for r in M['rows']]; fns = [f for M in R for f in M['fns']]
print('TP', sum(r['lab']=='TP' for r in rows), 'FP', sum(r['lab']=='FP' for r in rows), 'FN', len(fns))
allc = Counter()
for M in R: allc.update(M['allc'])
def tab(key, f=lambda r: r):
    C = defaultdict(Counter)
    for r in rows: C[key(r)][r['lab']] += 1
    for k in sorted(C, key=lambda k: str(k)):
        c = C[k]; n = c['TP'] + c['FP']
        print('   %-28s TP %6d FP %5d  fp%% %.1f' % (str(k), c['TP'], c['FP'], 100 * c['FP'] / max(1, n)))
print('by edge type (all edges count)'); 
C = defaultdict(Counter)
for r in rows: C[r['et']][r['lab']] += 1
for k in sorted(C): print('   %-20s all %8d  TP %6d FP %5d fp%% %.1f' % (k, allc[k], C[k]['TP'], C[k]['FP'], 100*C[k]['FP']/max(1,C[k]['TP']+C[k]['FP'])))
print('src fork / dst fork'); tab(lambda r: (r['sfork'], r['dfork']))
print('src start / dst end'); tab(lambda r: (r['sstart'], r['dend']))
print('node types'); tab(lambda r: (r['snt'], r['dnt']))
print('run len'); tab(lambda r: min(r.get('rl', 0) // 5 * 5, 60))
print('nn dist min(x,y)'); tab(lambda r: min(int(min(r['nnx'], r['nny'])), 12))
print('disp'); tab(lambda r: min(int(r['disp']), 12))
print('prob'); tab(lambda r: round(r['p'], 1))
print('FP subtypes', Counter(r['fp'] for r in rows if r['lab']=='FP'))
print('FN cats', Counter(f['c'] for f in fns))
print('FN by emb', {e: Counter(f['c'] for f in fns if f['m'].startswith(e)) for e in ['44b6','6bba']})
