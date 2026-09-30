"""check2/holdout (b): inventory of every per-movie rows file (rule_eval format) on the 199 movies, with base identification,
per-variant delta, fast movie-bootstrap SD, node-only vs evaluable split, dedup by per-movie signature."""
import json, glob, os, re, time, sys
import numpy as np
from collections import defaultdict

ROOTS = sorted(set(glob.glob('/workspace/cl/rule_eval_last_*.json') + glob.glob('/workspace/cl/p16/p19/*/*.json') +
                   glob.glob('/workspace/cl/p16/rows_*.json') + glob.glob('/workspace/cl/p16/check/*/rows*.json') +
                   glob.glob('/workspace/cl/p16/check/*/*/rows*.json')))
K = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes']
p15 = json.load(open('/workspace/cl/p16/check2/holdout/p19r_claim_rows.json'))
P15N = {r['movie']: r['num_pred_nodes'] for r in p15 if r['vi'] == 0}
MS = sorted(P15N)
# known bases: num_pred_nodes per movie
BASES = {'p15': P15N}
for tag, pat in [('p17', '/workspace/cl/p16/ps_p17_%s/graphs'), ('p14', '/workspace/cl/ps_p14_%s/graphs'), ('p13', '/workspace/cl/ps_p13_%s/graphs')]:
    pass


def arr(rows):
    by = {r['movie']: r for r in rows}
    if sorted(by) != MS: return None
    A = np.array([[by[m][k] for k in K] + [by[m]['adj_edge_jaccard'], by[m]['n_total']] for m in MS], dtype=float)
    return A


def score(A, wts=None):
    # A: M x 9 ; wts: B x M multiplicities
    tp, fp, fn, dtp, dfp, dfn, nod, adj, nt = A.T
    w = tp + fp + fn
    if wts is None: wts = np.ones((1, len(tp)))
    adjE = (wts @ (w * adj)) / (wts @ w)
    D = wts @ dtp; den = D + wts @ dfp + wts @ dfn
    return adjE + 0.1 * np.where(den > 0, D / np.maximum(den, 1e-9), 0)


rng = np.random.default_rng(0)
Wb = np.stack([np.bincount(rng.integers(0, len(MS), len(MS)), minlength=len(MS)) for _ in range(400)]).astype(float)
emb = np.array([m[:4] for m in MS]); sets = None
out = []; sigs = {}
for f in ROOTS:
    try:
        R = json.load(open(f))
    except Exception:
        continue
    if not (isinstance(R, list) and R and isinstance(R[0], dict) and 'vi' in R[0] and 'edge_tp' in R[0]): continue
    mt = time.strftime('%m-%d_%H:%M', time.gmtime(os.path.getmtime(f)))
    byvi = defaultdict(list)
    for r in R: byvi[r['vi']].append(r)
    if 0 not in byvi: continue
    B = arr(byvi[0])
    if B is None: continue
    bn = {m: n for m, n in zip(MS, B[:, 6])}
    base = 'p15' if all(bn[m] == P15N[m] for m in MS) else 'other(%+d vs p15)' % int(sum(bn[m] - P15N[m] for m in MS))
    sb = score(B)[0]; sbb = score(B, Wb)
    for vi in sorted(v for v in byvi if v != 0):
        C = arr(byvi[vi])
        if C is None: continue
        d = score(C)[0] - sb; db = score(C, Wb) - sbb
        # evaluable-only: recompute adj with base node count
        Ce = C.copy()
        J = Ce[:, 0] / np.maximum(Ce[:, 0] + Ce[:, 1] + Ce[:, 2], 1)
        Ce[:, 7] = np.maximum(0, J * (1 - 0.1 * (B[:, 6] - Ce[:, 8]) / Ce[:, 8]))
        de = score(Ce)[0] - sb
        sig = hash(C[:, :7].tobytes()) if base == 'p15' else hash((B[:, :7] - C[:, :7]).tobytes())
        dup = sigs.get(sig)
        if dup is None: sigs[sig] = (os.path.basename(f), vi)
        m44 = emb == '44b6'
        d44 = score(C[m44])[0] - score(B[m44])[0]; d6b = score(C[~m44])[0] - score(B[~m44])[0]
        out.append(dict(file=f.replace('/workspace/cl/', ''), mtime=mt, base=base, vi=vi, d=d, sd=float(db.std()), lo=float(np.quantile(db, .025)),
                        d_eval=de, d44=d44, d6b=d6b, dnodes=int((C[:, 6] - B[:, 6]).sum()), dtp=int((C[:, 0] - B[:, 0]).sum()), dfp=int((C[:, 1] - B[:, 1]).sum()),
                        ddtp=int((C[:, 3] - B[:, 3]).sum()), ddfp=int((C[:, 4] - B[:, 4]).sum()), dup_of=dup))
json.dump(out, open('/workspace/cl/p16/check2/holdout/inventory.json', 'w'), indent=0)
for o in sorted(out, key=lambda o: (o['mtime'], o['file'], o['vi'])):
    print('%s %-52s %-18s v%-2d d %+.5f sd %.5f lo %+.5f eval %+.5f 44 %+.5f 6b %+.5f nod %+6d tp %+3d fp %+4d dv %+d/%+d %s' % (
        o['mtime'], o['file'][:52], o['base'][:18], o['vi'], o['d'], o['sd'], o['lo'], o['d_eval'], o['d44'], o['d6b'], o['dnodes'], o['dtp'], o['dfp'], o['ddtp'], o['ddfp'],
        ('DUP ' + '%s:%d' % o['dup_of']) if o['dup_of'] else ''))
