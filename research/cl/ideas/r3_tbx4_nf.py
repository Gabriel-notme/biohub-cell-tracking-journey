"""From rule_eval rows: node-factor robustness (N_total x0.75 / x1.3), per-embryo bootstrap CI, per-embryo clean40, per-set deltas,
edge-only vs node-term decomposition, per-movie win/loss counts, top-10 share, leave-one-set-out."""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '/workspace/cl/rule_eval_last_ideas_r3_tbx4.json'))
A = 0.1


def score(rows, f=1.0, nodefix=None):
    w = 0; s = 0; dt = dfp = dfn = 0
    for r in rows:
        tp, fp, fn = r['edge_tp'], r['edge_fp'], r['edge_fn']
        W = tp + fp + fn
        if W == 0: continue
        J = tp / W
        Nt = r['n_total'] * f
        Np = r['num_pred_nodes'] if nodefix is None else nodefix[r['movie']]
        adj = max(0.0, J * (1 - A * (Np - Nt) / Nt))
        w += W; s += W * adj
        dt += r['division_tp']; dfp += r['division_fp']; dfn += r['division_fn']
    dj = dt / (dt + dfp + dfn) if dt + dfp + dfn else 0.0
    return s / w + 0.1 * dj


vis = sorted({r['vi'] for r in R})
base = {r['movie']: r for r in R if r['vi'] == 0}
ms = sorted(base)
# sanity: my score == official summarise
chk = abs(score([base[m] for m in ms]) - summarise([base[m] for m in ms])['score'])
print('recompute check |diff| = %.2e' % chk)
rng = np.random.default_rng(1)
basen = {m: base[m]['num_pred_nodes'] for m in ms}
for vi in vis[1:]:
    cur = {r['movie']: r for r in R if r['vi'] == vi}
    print('=== variant', vi)
    for f in [0.75, 1.0, 1.3]:
        line = 'Ntot x%.2f:' % f
        for emb in ['44b6', '6bba', '']:
            mm = [m for m in ms if m.startswith(emb)]
            line += ' %s %+.5f' % (emb or 'all', score([cur[m] for m in mm], f) - score([base[m] for m in mm], f))
        print(line)
    line = 'edge-only (base node counts):'
    for emb in ['44b6', '6bba', '']:
        mm = [m for m in ms if m.startswith(emb)]
        line += ' %s %+.5f' % (emb or 'all', score([cur[m] for m in mm], 1.0, basen) - score([base[m] for m in mm]))
    print(line)
    for emb in ['44b6', '6bba']:
        mm = [m for m in ms if m.startswith(emb)]
        bs = []
        for _ in range(1000):
            k = rng.integers(0, len(mm), len(mm))
            bs.append(score([cur[mm[j]] for j in k]) - score([base[mm[j]] for j in k]))
        bs = np.array(bs)
        cm = [m for m in mm if base[m]['set'] in ('hold36', 'prev4')]
        print('%s: CI95 [%+.5f, %+.5f] P>0 %.3f | clean %d movies %+.5f' % (emb, np.quantile(bs, .025), np.quantile(bs, .975), (bs > 0).mean(), len(cm),
                                                                         score([cur[m] for m in cm]) - score([base[m] for m in cm])))
    line = 'per set:'
    for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b']:
        mm = [m for m in ms if base[m]['set'] == s]
        line += ' %s %+.5f' % (s, score([cur[m] for m in mm]) - score([base[m] for m in mm]))
    print(line)
    line = 'leave-one-set-out:'
    for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b']:
        mm = [m for m in ms if base[m]['set'] != s]
        line += ' -%s %+.5f' % (s, score([cur[m] for m in mm]) - score([base[m] for m in mm]))
    print(line)
    d = {m: score([cur[m]]) - score([base[m]]) for m in ms}
    dtp = sum(cur[m]['edge_tp'] - base[m]['edge_tp'] for m in ms); dfp = sum(cur[m]['edge_fp'] - base[m]['edge_fp'] for m in ms)
    dn = sum(cur[m]['num_pred_nodes'] - base[m]['num_pred_nodes'] for m in ms); nt = sum(base[m]['num_pred_nodes'] for m in ms)
    top = sorted(d.values(), reverse=True)
    print('movies up %d down %d eq %d | dTP %+d dFP %+d | nodes %+d (%.2f%%) | top10 sum %+.5f of total-per-movie sum %+.5f | worst3 %s' % (
        sum(v > 1e-9 for v in d.values()), sum(v < -1e-9 for v in d.values()), sum(abs(v) <= 1e-9 for v in d.values()), dtp, dfp, dn, 100. * dn / nt,
        sum(top[:10]), sum(top), [(m, round(d[m], 5), cur[m]['edge_tp'] - base[m]['edge_tp'], cur[m]['edge_fp'] - base[m]['edge_fp'],
                                  cur[m]['num_pred_nodes'] - base[m]['num_pred_nodes']) for m in sorted(d, key=d.get)[:3]]))
