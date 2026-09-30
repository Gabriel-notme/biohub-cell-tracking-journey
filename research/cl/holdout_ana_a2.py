"""check2/holdout (a): exact scores of the subset-selected combos (rows2) vs the additive approximation (rows1), plus bootstrap CIs."""
import json, glob
import numpy as np
K = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes']


def load(d):
    X = {}; ST = {}
    for f in sorted(glob.glob(d + '/*.json')):
        x = json.load(open(f)); ST[x['movie']] = x['set']
        X[x['movie']] = [[r[k] for k in K] + [r['n_total']] for r in sorted(x['rows'], key=lambda r: r['vi'])]
    ms = sorted(X); return ms, np.array([X[m] for m in ms], float), np.array([ST[m] for m in ms])


ms, X2, ST = load('/workspace/cl/p16/check2/holdout/rows2')
ms1, X1, _ = load('/workspace/cl/p16/check2/holdout/rows1')
assert ms == ms1
c1 = json.load(open('/workspace/cl/p16/check2/holdout/cfgs1.json')); c2 = json.load(open('/workspace/cl/p16/check2/holdout/cfgs2.json'))
EMB = np.array([m[:4] for m in ms])


def score(C, mask):
    C = C[mask]; tp, fp, fn, dtp, dfp, dfn, nod, nt = C.T
    w = tp + fp + fn; J = np.where(w > 0, tp / np.maximum(w, 1), 0)
    adj = np.maximum(0, J * (1 - 0.1 * (nod - nt) / nt)); den = dtp.sum() + dfp.sum() + dfn.sum()
    return (w * adj).sum() / w.sum() + 0.1 * (dtp.sum() / den if den > 0 else 0)


DEF = c1[0]; P = ['cd', 'ff', 'st', 'tt', 'par', 'bm', 'bl']
oat = {}
for i, c in enumerate(c1, 1):
    diff = [p for p in P if c[p] != DEF[p]]
    if len(diff) == 1: oat[(diff[0], c[diff[0]])] = i


def additive(c):
    C = X1[:, 1].copy()
    for p in P:
        if c[p] != DEF[p]: C[:, :7] += X1[:, oat[(p, c[p])], :7] - X1[:, 1, :7]
    return C


S = {'all': np.ones(len(ms), bool), '44b6': EMB == '44b6', '6bba': EMB == '6bba', 'clean40': np.isin(ST, ['hold36', 'prev4']),
     't127ab': np.isin(ST, ['t127a', 't127b']), 'audit32': ST == 'audit32'}
lab = ['default', 'sel 44b6->6bba', 'sel 6bba->44b6', 'sel t127ab->clean40', 'sel clean40->t127ab', 'all-199 per-param argmax']
rng = np.random.default_rng(3)
for i, c in enumerate(c2, 1):
    ex = {k: score(X2[:, i], m) - score(X2[:, 0], m) for k, m in S.items()}
    ad = {k: score(additive(c), m) - score(X1[:, 0], m) for k, m in S.items()}
    cnt = (X2[:, i, :7] - X2[:, 0, :7]).sum(0).astype(int)
    print('%-26s %s' % (lab[i - 1], ' '.join('%s=%s' % (p, c[p]) for p in P)))
    print('   exact   : ' + ' '.join('%s %+.6f' % kv for kv in ex.items()) + ' | tp %+d fp %+d nodes %+d' % (cnt[0], cnt[1], cnt[6]))
    print('   additive: ' + ' '.join('%s %+.6f' % kv for kv in ad.items()))
    for k in ['44b6', '6bba', 'clean40', 't127ab']:
        sel = np.flatnonzero(S[k]); bs = []
        for _ in range(400):
            j = rng.choice(sel, len(sel)); mk = np.zeros(len(ms), bool)
            # weighted resample: build counts with multiplicity
            w = np.bincount(j, minlength=len(ms)).astype(float)
            def sc(C):
                tp, fp, fn, dtp, dfp, dfn, nod, nt = C.T
                ww = tp + fp + fn; J = np.where(ww > 0, tp / np.maximum(ww, 1), 0)
                adj = np.maximum(0, J * (1 - 0.1 * (nod - nt) / nt)); den = (w * (dtp + dfp + dfn)).sum()
                return (w * ww * adj).sum() / (w * ww).sum() + 0.1 * ((w * dtp).sum() / den if den > 0 else 0)
            bs.append(sc(X2[:, i]) - sc(X2[:, 0]))
        bs = np.array(bs)
        print('      %-8s exact %+.6f  movie-bootstrap CI [%+.6f, %+.6f] P>0 %.3f' % (k, ex[k], np.quantile(bs, .025), np.quantile(bs, .975), (bs > 0).mean()))
