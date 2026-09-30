"""Pooled evaluation across sets. usage: pool_eval.py <rows_dir>  (files named <set>__<config>.json)"""
import sys, json, glob, os
from pathlib import Path
sys.path.insert(0, '/workspace/official/src')
import numpy as np
from tracking_cellmot.metrics import summarise
d = sys.argv[1]
rows = {}
for f in glob.glob(d + '/*__*.json'):
    s, c = Path(f).stem.split('__'); rows.setdefault(c, {})[s] = json.load(open(f))
sets = sorted({s for c in rows for s in rows[c]})
print('sets', sets)
hdr = '%-6s' % 'cfg' + ''.join('%12s' % s for s in sets) + '%12s %14s' % ('pooled', 'div TP/FP/FN')
print(hdr)
res = {}
for c in sorted(rows):
    if set(rows[c]) != set(sets): continue
    line = '%-6s' % c
    allr = []
    for s in sets:
        sm = summarise(rows[c][s]); line += '%12.6f' % sm['score']; allr += rows[c][s]
    sp = summarise(allr); res[c] = sp['score']
    line += '%12.6f %14s' % (sp['score'], '%d/%d/%d' % (sp['division_tp'], sp['division_fp'], sp['division_fn']))
    print(line)
# paired bootstrap vs B5 and vs P1 on pooled movies
rng = np.random.default_rng(0)
def boot(a, b, n=2000):
    A = [r for s in sets for r in rows[a][s]]; B = [r for s in sets for r in rows[b][s]]
    idx = np.arange(len(A)); out = []
    for _ in range(n):
        k = rng.choice(idx, len(idx)); out.append(summarise([B[i] for i in k])['score'] - summarise([A[i] for i in k])['score'])
    out = np.array(out); return out.mean(), np.quantile(out, .025), np.quantile(out, .975), (out > 0).mean()
for base in ['b5', 'p1']:
    if base not in res: continue
    for c in sorted(res):
        if c == base: continue
        m, lo, hi, p = boot(base, c)
        print('%s -> %s pooled delta %+.5f  95%% CI [%+.5f, %+.5f]  P>0 %.3f' % (base, c, m, lo, hi, p))
