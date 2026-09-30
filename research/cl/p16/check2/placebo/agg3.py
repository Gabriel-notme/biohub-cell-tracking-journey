"""chk2 placebo: aggregation of the in-context per-family placebo (plc3)."""
import sys, json, glob
from collections import defaultdict
import numpy as np
sys.path.insert(0, '/workspace/official/src')
import warnings; warnings.filterwarnings('ignore')
from tracking_cellmot.metrics import summarise, per_sample_metrics, EvaluationResult

RD = sys.argv[1] if len(sys.argv) > 1 else '/workspace/cl/p16/check2/placebo/rows3'
R = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
X = defaultdict(dict); META = {}
for f in sorted(glob.glob(RD + '/*.json')):
    d = json.load(open(f)); META[d['movie']] = d
    for c, r in d['res'].items(): X[c][d['movie']] = r
CTX = ['seq_cd', 'seq_ff', 'seq_st', 'seq_tt', 'seq_par', 'seq_bd', 'seq_dup', 'loo_cd', 'loo_ff', 'loo_st', 'loo_tt', 'loo_dup']
ms = sorted(META); M = len(ms)
print('movies', M, {c: len(X[c]) for c in CTX})
NT = np.array([META[m]['n_total'] for m in ms], float)
GROUPS = [('all', np.ones(M, bool)), ('44b6', np.array([m.startswith('44b6') for m in ms])), ('6bba', np.array([m.startswith('6bba') for m in ms])),
          ('clean40', np.array([META[m]['set'] in ('hold36', 'prev4') for m in ms]))]


def score(C, sel):
    C = C[..., sel, :]; nt = NT[sel]
    tp, fp, fn, dtp, dfp, dfn, npred = [C[..., i] for i in range(7)]
    den = tp + fp + fn
    J = np.where(den > 0, tp / np.where(den > 0, den, 1), 0.0)
    adj = np.maximum(0.0, J * (1 - 0.1 * (npred - nt) / nt))
    return (den * adj).sum(-1) / den.sum(-1) + 0.1 * dtp.sum(-1) / (dtp.sum(-1) + dfp.sum(-1) + dfn.sum(-1))


rng = np.random.default_rng(777)
out = {}
for c in CTX:
    if len(X[c]) != M: print('skip', c); continue
    K = min(X[c][m]['draws'].__len__() for m in ms)
    B = np.array([X[c][m]['base'] for m in ms], float); O = np.array([X[c][m]['obs'] for m in ms], float)
    DR = np.stack([np.array(X[c][m]['draws'][:K], float) for m in ms], axis=1)
    for g, sel in GROUPS[:1]:
        rows = [per_sample_metrics(EvaluationResult(*[int(x) for x in B[i]]), NT[i], 0.0) for i in np.flatnonzero(sel)]
        assert abs(summarise(rows)['score'] - score(B, sel)) < 1e-12
    sh = np.array([sum(X[c][m]['short'][:K]) for m in ms]).sum() / K
    dO = O.sum(0) - B.sum(0); dP = DR.sum(1) - B.sum(0)
    print('\n== %s (K=%d): obs dTP %+d dFP %+d nodes %+d | placebo mean dTP %+.2f dFP %+.2f nodes %+.1f | shortfall nodes/draw %+.2f' % (
        c, K, dO[0], dO[1], dO[6], dP[:, 0].mean(), dP[:, 1].mean(), dP[:, 6].mean(), sh))
    REC = rng.integers(0, K, (R, M)); Crec = DR[REC, np.arange(M)[None, :], :]
    out[c] = {}
    for g, sel in GROUPS:
        sb = score(B, sel); o = score(O, sel) - sb; pl = score(DR, sel) - sb
        Cn = B.copy(); Cn[:, 6] = O[:, 6]; node_only = score(Cn, sel) - sb
        ge = int((pl >= o - 1e-12).sum()); gt = int((pl > o + 1e-12).sum()); eq = ge - gt
        p = (1 + ge) / (K + 1); pmid = (1 + gt + 0.5 * eq) / (K + 1)
        rec = score(Crec, sel) - sb; prec = (1 + int((rec >= o - 1e-12).sum())) / (R + 1)
        sd = pl.std(ddof=1)
        out[c][g] = dict(obs=o, node_only=node_only, mean=pl.mean(), sd=sd, q05=np.quantile(pl, .05), q95=np.quantile(pl, .95), mx=pl.max(), ge=ge, eq=eq,
                         p=p, pmid=pmid, prec=prec)
        print('   %-8s obs %+.6f | node-only %+.6f | placebo mean %+.6f sd %.6f q05 %+.6f q95 %+.6f max %+.6f | #pl>=obs %d/%d (ties %d) p=%.4f mid-p=%.4f | recomb p=%.5f | obs-mean %+.6f' % (
            g, o, node_only, pl.mean(), sd, out[c][g]['q05'], out[c][g]['q95'], pl.max(), ge, K, eq, p, pmid, prec, o - pl.mean()))
for pre, fams in [('seq', ['cd', 'ff', 'st', 'tt', 'par', 'bd']), ('loo', ['cd', 'ff', 'st', 'tt'])]:
    for g in ['all', '44b6', '6bba', 'clean40']:
        cs = [pre + '_' + f for f in fams if pre + '_' + f in out]
        ps = sorted((out[c][g]['p'], c) for c in cs); adj = {}; run = 0
        for i, (p, c) in enumerate(ps): run = max(run, min(1, (len(ps) - i) * p)); adj[c] = run
        print('Holm-adjusted exact p (%s, %s): %s' % (pre, g, ' '.join('%s %.4f' % (c, adj[c]) for c in cs)))
json.dump({c: {g: {k: float(x) for k, x in r.items()} for g, r in gg.items()} for c, gg in out.items()}, open(RD.rstrip('/') + '_agg.json', 'w'), indent=1)
