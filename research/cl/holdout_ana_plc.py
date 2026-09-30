"""check2/holdout (b): dup / border rule vs placebo (random same-shape deletions), official metric from counts."""
import json, glob, sys
import numpy as np
d = sys.argv[1]
K = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes']
R = {}; SET = {}; INFO = {}
for f in sorted(glob.glob(d + '/*.json')):
    x = json.load(open(f)); m = x['info']['movie']; SET[m] = x['info']['set']; INFO[m] = x['info']
    R[m] = {r['var']: [r[k] for k in K] + [r['n_total']] for r in x['rows']}
MS = sorted(R); print('movies', len(MS))
tags = list(R[MS[0]])
X = {t: np.array([R[m][t] for m in MS], dtype=float) for t in tags}
EMB = np.array([m[:4] for m in MS]); ST = np.array([SET[m] for m in MS])


def score(C, mask, nodes_from=None):
    C = C[mask]; tp, fp, fn, dtp, dfp, dfn, nod, nt = C.T
    if nodes_from is not None: nod = nodes_from[mask][:, 6]
    w = tp + fp + fn; J = np.where(w > 0, tp / np.maximum(w, 1), 0)
    adj = np.maximum(0, J * (1 - 0.1 * (nod - nt) / nt)); den = dtp.sum() + dfp.sum() + dfn.sum()
    return (w * adj).sum() / w.sum() + 0.1 * (dtp.sum() / den if den > 0 else 0)


SUB = {'all': np.ones(len(MS), bool), '44b6': EMB == '44b6', '6bba': EMB == '6bba', 'clean40': np.isin(ST, ['hold36', 'prev4'])}
B = X['p15']
for fam, pre in [('dup', 'Pdup'), ('bd', 'Pbd')]:
    P = sorted([t for t in tags if t.startswith(pre)], key=lambda t: int(t[len(pre):]))
    print('\n== %s rule vs %d placebo draws (random same-shape deletions)' % (fam, len(P)))
    for lab, mk in SUB.items():
        obs = score(X[fam], mk) - score(B, mk); pl = np.array([score(X[t], mk) - score(B, mk) for t in P])
        obs_e = score(X[fam], mk, B) - score(B, mk); pl_e = np.array([score(X[t], mk, B) - score(B, mk) for t in P])
        print('  %-8s obs %+.6f (evaluable part %+.6f, node part %+.6f) | placebo mean %+.6f sd %.6f max %+.6f, #>=obs %d/%d | placebo evaluable mean %+.6f sd %.6f, #>=obs %d' % (
            lab, obs, obs_e, obs - obs_e, pl.mean(), pl.std(), pl.max(), (pl >= obs).sum(), len(P), pl_e.mean(), pl_e.std(), (pl_e >= obs_e).sum()))
    cnt = lambda t: tuple(int(v) for v in (X[t][:, :7] - B[:, :7]).sum(0)[[0, 1, 3, 4, 6]])
    print('  counts (tp, fp, divtp, divfp, nodes): rule %s | placebo %s' % (cnt(fam), [cnt(t) for t in P]))
short = [sum(p['dup_short'] for p in INFO[m]['placebo']) for m in MS]
print('\nplacebo shortfall (nodes not matched): dup %d total over all draws; bd %d' % (sum(short), sum(sum(p['bd_short'] for p in INFO[m]['placebo']) for m in MS)))
