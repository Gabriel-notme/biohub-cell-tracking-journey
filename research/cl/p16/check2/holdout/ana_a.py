"""check2/holdout (a): threshold/family selection on one subset, evaluation on the held-out subset.
Selection: one-at-a-time around the default over each parameter's grid (incl. family off), combined config = per-parameter argmax.
Combined-config counts approximated additively in per-movie COUNTS (exact runs of the selected combos are done separately).
usage: ana_a.py <rowsdir> <cfgs.json> [nsplit]"""
import json, glob, sys, os
import numpy as np
from collections import defaultdict

rowsdir, cfgp = sys.argv[1], sys.argv[2]; NSPLIT = int(sys.argv[3]) if len(sys.argv) > 3 else 500
cfgs = json.load(open(cfgp))
K = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes']
files = sorted(glob.glob(rowsdir + '/*.json'))
MS = []; SET = []; X = []  # X[m, vi, :] = counts + n_total
for f in files:
    d = json.load(open(f)); rows = sorted(d['rows'], key=lambda r: r['vi'])
    assert [r['vi'] for r in rows] == list(range(len(cfgs) + 1)), f
    MS.append(d['movie']); SET.append(d['set'])
    X.append([[r[k] for k in K] + [r['n_total']] for r in rows])
X = np.array(X, dtype=float); MS = np.array(MS); SET = np.array(SET)
EMB = np.array([m[:4] for m in MS])
print('movies', len(MS), 'variants', X.shape[1] - 1)


def score(C, mask=None):
    """official summarise from counts; C: (M, 8) counts+n_total"""
    if mask is not None: C = C[mask]
    tp, fp, fn, dtp, dfp, dfn, nod, nt = C.T
    w = tp + fp + fn
    J = np.where(w > 0, tp / np.maximum(w, 1), 0)
    adj = np.maximum(0, J * (1 - 0.1 * (nod - nt) / nt))
    adjE = (w * adj).sum() / w.sum()
    den = dtp.sum() + dfp.sum() + dfn.sum()
    return adjE + 0.1 * (dtp.sum() / den if den > 0 else 0)


def delta(C, mask): return score(C, mask) - score(X[:, 0], mask)


DEF = cfgs[0]
PARAMS = ['cd', 'ff', 'st', 'tt', 'par', 'bm', 'bl']
# one-at-a-time index: (param, value) -> vi
OAT = defaultdict(dict)
for i, c in enumerate(cfgs, 1):
    diff = [p for p in PARAMS if c[p] != DEF[p]]
    if i == 1: [OAT[p].__setitem__(DEF[p], 1) for p in PARAMS]
    elif len(diff) == 1: OAT[diff[0]][c[diff[0]]] = i
for p in PARAMS: print('grid', p, sorted(OAT[p].items(), key=lambda kv: (kv[0] is not None, kv[0] if kv[0] is not None else 0)))


def combo_counts(sel):
    """additive-in-counts approximation of the combined config sel {p: v}"""
    C = X[:, 1, :].copy()
    for p, v in sel.items():
        vi = OAT[p][v]
        if vi != 1: C[:, :7] += X[:, vi, :7] - X[:, 1, :7]
    return C


def select(mask):
    sel = {}; tab = {}
    for p in PARAMS:
        ds = {v: delta(X[:, vi], mask) for v, vi in OAT[p].items()}
        best = max(ds.values())
        # ties: prefer the default, then the first in grid order
        cands = [v for v, dv in ds.items() if dv >= best - 1e-12]
        sel[p] = DEF[p] if DEF[p] in cands else cands[0]; tab[p] = ds
    return sel, tab


def fmt(sel):
    return ' '.join('%s=%s' % (p, sel[p]) for p in PARAMS)


SPLITS = [('44b6 -> 6bba', EMB == '44b6', EMB == '6bba'), ('6bba -> 44b6', EMB == '6bba', EMB == '44b6'),
          ('t127ab -> clean40', np.isin(SET, ['t127a', 't127b']), np.isin(SET, ['hold36', 'prev4'])),
          ('clean40 -> t127ab', np.isin(SET, ['hold36', 'prev4']), np.isin(SET, ['t127a', 't127b']))]
ALL = np.ones(len(MS), bool)
out = {'splits': []}
print('\n== default (P19-R) deltas: all %+.6f | 44b6 %+.6f | 6bba %+.6f | clean40 %+.6f | t127ab %+.6f | audit32 %+.6f' % tuple(
    delta(X[:, 1], m) for m in [ALL, EMB == '44b6', EMB == '6bba', SPLITS[2][2], SPLITS[2][1], SET == 'audit32']))
for lab, T, H in SPLITS:
    sel, tab = select(T)
    Ca = combo_counts(sel)
    r = dict(split=lab, sel=sel, nT=int(T.sum()), nH=int(H.sum()),
             T_sel=delta(Ca, T), T_def=delta(X[:, 1], T), H_sel=delta(Ca, H), H_def=delta(X[:, 1], H))
    selH, _ = select(H); r['H_oracle'] = delta(combo_counts(selH), H); r['H_oracle_sel'] = selH
    # per-parameter: value chosen on T, its OAT delta on T and on H, and the H-best value
    print('\n== %s (T %d movies, H %d movies)' % (lab, T.sum(), H.sum()))
    print('   selected on T: %s' % fmt(sel))
    print('   T: selected %+.6f  default %+.6f | H: selected %+.6f  default %+.6f  (H-oracle %+.6f with %s)' % (r['T_sel'], r['T_def'], r['H_sel'], r['H_def'], r['H_oracle'], fmt(selH)))
    for p in PARAMS:
        ds = tab[p]; dsH = {v: delta(X[:, OAT[p][v]], H) for v in OAT[p]}
        order = sorted(ds, key=lambda v: (v is not None, v if v is not None else 0))
        print('   %-4s ' % p + '  '.join('%s:%s%+.6f/%+.6f' % (v, '*' if v == sel[p] else ('d' if v == DEF[p] else ' '), ds[v], dsH[v]) for v in order))
    out['splits'].append(r)

# random embryo-stratified half splits: honest optimism of the tuning procedure
rng = np.random.default_rng(12345)
res = []
for k in range(NSPLIT):
    T = np.zeros(len(MS), bool)
    for e in ['44b6', '6bba']:
        idx = np.flatnonzero(EMB == e); T[rng.choice(idx, len(idx) // 2, replace=False)] = True
    H = ~T
    sel, _ = select(T); Ca = combo_counts(sel)
    res.append((delta(Ca, T), delta(Ca, H), delta(X[:, 1], T), delta(X[:, 1], H), sum(sel[p] != DEF[p] for p in PARAMS),
                sum(sel[p] is None for p in PARAMS)))
res = np.array(res, dtype=float)
opt = res[:, 0] - res[:, 1]
print('\n== %d random embryo-stratified half splits (tune on T, evaluate on H):' % NSPLIT)
print('   T(selected) mean %+.6f | H(selected) mean %+.6f sd %.6f P(H>0) %.3f q05 %+.6f | optimism T-H mean %+.6f sd %.6f' % (
    res[:, 0].mean(), res[:, 1].mean(), res[:, 1].std(), (res[:, 1] > 0).mean(), np.quantile(res[:, 1], .05), opt.mean(), opt.std()))
print('   default: T mean %+.6f H mean %+.6f P(H>0) %.3f | H(selected)-H(default) mean %+.6f, P(selected better on H) %.3f' % (
    res[:, 2].mean(), res[:, 3].mean(), (res[:, 3] > 0).mean(), (res[:, 1] - res[:, 3]).mean(), (res[:, 1] > res[:, 3]).mean()))
print('   params moved from default: mean %.2f | families switched off: mean %.2f' % (res[:, 4].mean(), res[:, 5].mean()))
out['random'] = dict(T_sel=res[:, 0].mean(), H_sel=res[:, 1].mean(), H_sel_sd=res[:, 1].std(), P_H_pos=(res[:, 1] > 0).mean(), opt=opt.mean(),
                     H_def=res[:, 3].mean(), P_def_pos=(res[:, 3] > 0).mean())
# the in-sample 'max over the whole OAT grid' vs default on all 199 (how much a tuned pick would claim)
alld = {vi: delta(X[:, vi], ALL) for vi in range(1, X.shape[1])}
print('\n== all-199 OAT grid: default %+.6f | max %+.6f (vi %d: %s) | min %+.6f (vi %d: %s) | median %+.6f' % (
    alld[1], max(alld.values()), max(alld, key=alld.get), cfgs[max(alld, key=alld.get) - 1], min(alld.values()), min(alld, key=alld.get),
    cfgs[min(alld, key=alld.get) - 1], np.median(list(alld.values()))))
json.dump(out, open(os.path.join(os.path.dirname(rowsdir.rstrip('/')), 'ana_a_out.json'), 'w'), default=str, indent=1)
