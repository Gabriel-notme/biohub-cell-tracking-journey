"""All-model rigorous review from cached per-movie official rows (/workspace/cl/rev/<name>_<set>.json)."""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
REV = '/workspace/cl/rev/'
MODELS = ['b3', 'b5', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6'] + [a for a in sys.argv[1:]]
SRC = {('p5', 'prev4'): 'p5kaggle', ('p6', 'prev4'): 'p6kaggle'}
SETS = ['hold36', 'prev4', 'audit32']
R = {}
for m in MODELS:
    for s in SETS:
        try: R[(m, s)] = {r['movie']: r for r in json.load(open(REV + '%s_%s.json' % (SRC.get((m, s), m), s)))}
        except FileNotFoundError: pass


def summ(rows):
    s = summarise(rows)
    return s['score'], s['adj_edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn']


print('%-5s | %-34s | %-34s | %-34s | %-9s | %-9s' % ('model', 'hold36 score (adjE, div TP/FP/FN)', 'prev4', 'audit32', 'CLEAN40', 'ALL72'))
avail = [m for m in MODELS if all((m, s) in R for s in SETS)]
for m in MODELS:
    line = '%-5s' % m
    for s in SETS:
        if (m, s) in R:
            sc, ae, tp, fp, fn = summ(list(R[(m, s)].values())); line += ' | %.5f (%.5f, %2d/%2d/%2d)     ' % (sc, ae, tp, fp, fn)
        else: line += ' | %-34s' % 'n/a'
    if m in avail:
        line += ' | %.5f  ' % summ([r for s in ['hold36', 'prev4'] for r in R[(m, s)].values()])[0]
        line += ' | %.5f' % summ([r for s in SETS for r in R[(m, s)].values()])[0]
    print(line)
# pairwise bootstrap on the clean 40 movies against the best clean model
clean = {m: summ([r for s in ['hold36', 'prev4'] for r in R[(m, s)].values()])[0] for m in avail}
best = max(clean, key=clean.get)
print('\nbest on clean 40 movies:', best, '%.5f' % clean[best])
keys = [(s, mv) for s in ['hold36', 'prev4'] for mv in R[(best, s)]]
rng = np.random.default_rng(0)
idx = [rng.integers(0, len(keys), len(keys)) for _ in range(2000)]
B = [R[(best, s)][mv] for s, mv in keys]
for m in avail:
    if m == best: continue
    A = [R[(m, s)][mv] for s, mv in keys]
    d = np.array([summarise([B[i] for i in k])['score'] - summarise([A[i] for i in k])['score'] for k in idx])
    de = np.array([summarise([B[i] for i in k])['adj_edge_jaccard'] - summarise([A[i] for i in k])['adj_edge_jaccard'] for k in idx])
    print('%s - %s: d %+.5f  95%% CI [%+.5f, %+.5f]  P(best better) %.3f | edge part d %+.5f CI [%+.5f, %+.5f]' % (
        best, m, clean[best] - clean[m], np.quantile(d, .025), np.quantile(d, .975), (d > 0).mean(), de.mean(), np.quantile(de, .025), np.quantile(de, .975)))
