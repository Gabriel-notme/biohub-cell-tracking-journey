"""check2/holdout (b): selection optimism of 'pick the best sibling' within the plausible neighbourhood (one step around the default,
incl. family off), estimated by (i) embryo-stratified split halves and (ii) Efron bootstrap optimism; additive-in-counts combos."""
import json, glob
import numpy as np
K = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes']
cfgs = json.load(open('/workspace/cl/p16/check2/holdout/cfgs1.json'))
X = []; MS = []; ST = []
for f in sorted(glob.glob('/workspace/cl/p16/check2/holdout/rows1/*.json')):
    d = json.load(open(f)); rows = sorted(d['rows'], key=lambda r: r['vi']); MS.append(d['movie']); ST.append(d['set'])
    X.append([[r[k] for k in K] + [r['n_total']] for r in rows])
X = np.array(X, float); M = len(MS); EMB = np.array([m[:4] for m in MS])
DEF = cfgs[0]; P = ['cd', 'ff', 'st', 'tt', 'par', 'bm', 'bl']
oat = {(p, DEF[p]): 1 for p in P}
for i, c in enumerate(cfgs, 1):
    diff = [p for p in P if c[p] != DEF[p]]
    if len(diff) == 1: oat[(diff[0], c[diff[0]])] = i
NB = {'cd': [None, 2.6, 3.2, 3.8], 'ff': [None, 5, 6, 7], 'st': [None, 2.0, 2.5, 3.0], 'tt': [None, 3.0, 3.5, 4.0],
      'par': [None, 3.0, 3.5, 4.0], 'bm': [None, 1.0, 2.0, 3.0], 'bl': [5, 6, 7]}


def sc(C, w, nodes_from=None):
    tp, fp, fn, dtp, dfp, dfn, nod, nt = C.T
    if nodes_from is not None: nod = nodes_from[:, 6]
    ww = tp + fp + fn; J = np.where(ww > 0, tp / np.maximum(ww, 1), 0)
    adj = np.maximum(0, J * (1 - 0.1 * (nod - nt) / nt)); den = w @ (dtp + dfp + dfn)
    return (w @ (ww * adj)) / (w @ ww) + 0.1 * np.where(den > 0, (w @ dtp) / np.maximum(den, 1e-9), 0)


def combo(sel):
    C = X[:, 1].copy()
    for p, v in sel.items():
        if v != DEF[p]: C[:, :7] += X[:, oat[(p, v)], :7] - X[:, 1, :7]
    return C


def pick(w):
    sel = {}
    for p in P:
        vals = NB[p]; ds = [(sc(X[:, oat[(p, v)]], w[None]) - sc(X[:, 0], w[None]))[0] for v in vals]
        sel[p] = vals[int(np.argmax(ds))]
    return sel


one = np.ones(M)
D = lambda C, w: (sc(C, w[None]) - sc(X[:, 0], w[None]))[0]
full_sel = pick(one); print('argmax in the plausible neighbourhood on all 199:', full_sel, 'delta %+.6f vs default %+.6f' % (D(combo(full_sel), one), D(X[:, 1], one)))
rng = np.random.default_rng(11)
# (i) split halves
sh = []
for k in range(400):
    T = np.zeros(M, bool)
    for e in ['44b6', '6bba']:
        idx = np.flatnonzero(EMB == e); T[rng.choice(idx, len(idx) // 2, replace=False)] = True
    s = pick(T.astype(float)); C = combo(s)
    sh.append((D(C, T.astype(float)), D(C, (~T).astype(float)), D(X[:, 1], T.astype(float)), D(X[:, 1], (~T).astype(float))))
sh = np.array(sh)
print('split halves (400): tuned T %+.6f -> H %+.6f (optimism %+.6f, P(H>0) %.3f) | default T %+.6f H %+.6f | P(tuned beats default on H) %.3f' % (
    sh[:, 0].mean(), sh[:, 1].mean(), (sh[:, 0] - sh[:, 1]).mean(), (sh[:, 1] > 0).mean(), sh[:, 2].mean(), sh[:, 3].mean(), (sh[:, 1] > sh[:, 3]).mean()))
# (ii) Efron bootstrap optimism of the pick procedure
op = []
for b in range(400):
    w = np.bincount(rng.integers(0, M, M), minlength=M).astype(float)
    s = pick(w); C = combo(s)
    op.append(D(C, w) - D(C, one))
op = np.array(op)
app = D(combo(full_sel), one)
print('Efron bootstrap: apparent %+.6f, optimism %+.6f (sd %.6f) -> optimism-corrected %+.6f' % (app, op.mean(), op.std(), app - op.mean()))
# default's rank among all neighbourhood combos? (fraction of the 4^6*3 combos with in-sample delta below the default)
import itertools
vals = [NB[p] for p in P]; ds = []
for tup in itertools.product(*vals):
    ds.append(D(combo(dict(zip(P, tup))), one))
ds = np.array(ds)
print('neighbourhood combos: %d | in-sample delta min %+.6f median %+.6f max %+.6f | default percentile %.1f%%' % (len(ds), ds.min(), np.median(ds), ds.max(), 100 * (ds < D(X[:, 1], one)).mean()))
