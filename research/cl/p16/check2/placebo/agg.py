"""chk2 placebo aggregation: placebo distributions and exact one-sided Monte-Carlo p-values (1 + #{placebo >= obs}) / (K + 1),
plus per-movie recombination (draws are independent across movies, so any per-movie combination is a valid null draw)."""
import sys, json, glob
import numpy as np
sys.path.insert(0, '/workspace/official/src')
import warnings; warnings.filterwarnings('ignore')
from tracking_cellmot.metrics import summarise, per_sample_metrics, EvaluationResult

RD = sys.argv[1] if len(sys.argv) > 1 else '/workspace/cl/p16/check2/placebo/rows'
R = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
D = {}
for f in sorted(glob.glob(RD + '/*.json')):
    d = json.load(open(f)); D[d['info']['movie']] = d
ms = sorted(D); M = len(ms); K = min(D[m]['info']['K'] for m in ms)
print('movies %d K %d | reduced-scorer verification: %d/%d checks equal to evalx.score_movie' % (
    M, K, sum(sum(D[m]['info']['ver']) for m in ms), sum(len(D[m]['info']['ver']) for m in ms)))
NT = np.array([D[m]['info']['n_total'] for m in ms], float)
VARS = ['A', 'cd', 'ff', 'st', 'tt', 'par', 'bd', 'dup']
OBSKEY = {'A': 'P19'}
OBS = {v: np.array([D[m]['obs'][OBSKEY.get(v, v)] for m in ms], float) for v in VARS}   # M x 7
BASE = np.array([D[m]['obs']['p15'] for m in ms], float)
DR = {v: np.stack([np.array(D[m]['draws'][v][:K], float) for m in ms], axis=1) for v in VARS}  # K x M x 7
SH = np.stack([np.array(D[m]['short'][:K], float) for m in ms], axis=1)  # K x M x 6
GROUPS = [('all', np.ones(M, bool)), ('44b6', np.array([m.startswith('44b6') for m in ms])), ('6bba', np.array([m.startswith('6bba') for m in ms])),
          ('clean40', np.array([D[m]['info']['set'] in ('hold36', 'prev4') for m in ms]))]


def score(C, sel):
    """C: (..., M, 7) counts -> score over movies in sel (vectorised summarise)."""
    C = C[..., sel, :]; nt = NT[sel]
    tp, fp, fn, dtp, dfp, dfn, npred = [C[..., i] for i in range(7)]
    den = tp + fp + fn
    with np.errstate(invalid='ignore', divide='ignore'):
        J = np.where(den > 0, tp / np.where(den > 0, den, 1), 0.0)
    adj = np.maximum(0.0, J * (1 - 0.1 * (npred - nt) / nt))
    adjE = (den * adj).sum(-1) / den.sum(-1)
    dd = dtp.sum(-1) + dfp.sum(-1) + dfn.sum(-1)
    divJ = dtp.sum(-1) / dd
    return adjE + 0.1 * divJ


# check vectorised score against the official summarise
for g, sel in GROUPS:
    for C in [BASE, OBS['A']]:
        rows = [dict(per_sample_metrics(EvaluationResult(*[int(x) for x in C[i]]), NT[i], 0.0)) for i in np.flatnonzero(sel)]
        assert abs(summarise(rows)['score'] - score(C, sel)) < 1e-12, g
print('vectorised score == official summarise: ok')
rng = np.random.default_rng(12345)
REC = rng.integers(0, K, (R, M))
out = {}
for v in VARS:
    print('\n== %s  (obs = deployed %s removal applied alone on P15; placebo = matched random deletions, K=%d)' % (v, 'P19-R' if v == 'A' else v, K))
    dobs = {k: OBS[v][:, k].sum() - BASE[:, k].sum() for k in range(7)}
    dpl = {k: (DR[v][:, :, k].sum(1) - BASE[:, k].sum()) for k in range(7)}
    print('   counts obs  : dTP %+d dFP %+d dFN %+d dDivTP %+d dDivFP %+d nodes %+d' % tuple(int(dobs[k]) for k in [0, 1, 2, 3, 4, 6]))
    print('   counts plac.: dTP %+.2f dFP %+.2f dFN %+.2f dDivTP %+.2f dDivFP %+.2f nodes %+.1f (min %+d max %+d) | shortfall nodes/draw %.2f' % (
        tuple(dpl[k].mean() for k in [0, 1, 2, 3, 4, 6]) + (int(dpl[6].min()), int(dpl[6].max()), SH.sum(1).sum(1).mean() if v == 'A' else 0.0)))
    print('   placebo draws with dTP<0: %.3f  dFP<0: %.3f  | P(dTP_pl <= obs dTP) %.3f  P(dFP_pl <= obs dFP) %.3f' % (
        (dpl[0] < 0).mean(), (dpl[1] < 0).mean(), (dpl[0] >= dobs[0]).mean(), (dpl[1] <= dobs[1]).mean()))
    out[v] = {}
    Crec = DR[v][REC, np.arange(M)[None, :], :]  # R x M x 7
    for g, sel in GROUPS:
        sb = score(BASE, sel); o = score(OBS[v], sel) - sb
        pl = score(DR[v], sel) - sb  # K
        # node-count-only counterfactual: base counts with the observed node number
        Cn = BASE.copy(); Cn[:, 6] = OBS[v][:, 6]; node_only = score(Cn, sel) - sb
        ge = int((pl >= o - 1e-12).sum()); gt = int((pl > o + 1e-12).sum()); eq = ge - gt
        p = (1 + ge) / (K + 1); pmid = (1 + gt + 0.5 * eq) / (K + 1)
        rec = score(Crec, sel) - sb  # recombination
        ger = int((rec >= o - 1e-12).sum()); prec = (1 + ger) / (R + 1)
        sd = pl.std(ddof=1)
        out[v][g] = dict(obs=o, node_only=node_only, mean=pl.mean(), sd=sd, q05=np.quantile(pl, .05), q50=np.median(pl), q95=np.quantile(pl, .95),
                         mx=pl.max(), mn=pl.min(), ge=ge, eq=eq, p=p, pmid=pmid, prec=prec, ppos=(pl > 1e-12).mean(), z=(o - pl.mean()) / sd if sd > 0 else float('nan'))
        r = out[v][g]
        print('   %-8s obs %+.6f | node-only %+.6f | placebo mean %+.6f sd %.6f q05 %+.6f q50 %+.6f q95 %+.6f max %+.6f P(pl>0) %.3f | '
              '#pl>=obs %d/%d (ties %d) p=%.4f mid-p=%.4f | recomb R=%d p=%.5f | obs-mean %+.6f z %.2f' % (
                  g, o, node_only, r['mean'], sd, r['q05'], r['q50'], r['q95'], r['mx'], r['ppos'], ge, K, eq, p, pmid, R, prec, o - r['mean'], r['z']))
# Holm over the six families (all-199, exact p)
fams = ['cd', 'ff', 'st', 'tt', 'par', 'bd']
for g in ['all', '44b6', '6bba', 'clean40']:
    ps = sorted((out[f][g]['p'], f) for f in fams); adj = {}; run = 0
    for i, (p, f) in enumerate(ps): run = max(run, min(1, (len(ps) - i) * p)); adj[f] = run
    print('Holm-adjusted exact p (%s): %s' % (g, ' '.join('%s %.4f' % (f, adj[f]) for f in fams)))
json.dump({v: {g: {k: float(x) for k, x in r.items()} for g, r in gg.items()} for v, gg in out.items()}, open(RD.rstrip('/') + '_agg.json', 'w'), indent=1)
