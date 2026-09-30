"""chk_ablate concentration + node-term decomposition on paired rows (vi=0 base, vi=k cand)."""
import sys, json, math
sys.path.insert(0, '/workspace/official/src')
import numpy as np
from tracking_cellmot.metrics import summarise
import warnings; warnings.filterwarnings('ignore')
rows, vi = json.load(open(sys.argv[1])), int(sys.argv[2])
base = {r['movie']: r for r in rows if r['vi'] == 0}; cur = {r['movie']: r for r in rows if r['vi'] == vi}
ms = sorted(base)
S = lambda rs: summarise(rs)['score']
def d(sel, B=base, C=cur): return S([C[m] for m in sel]) - S([B[m] for m in sel])
tot = d(ms); sb = S([base[m] for m in ms])
marg = {m: S([cur[x] if x == m else base[x] for x in ms]) - sb for m in ms}
drop1 = {m: tot - d([x for x in ms if x != m]) for m in ms}
print('total delta %+.6f  sum of marginals %+.6f' % (tot, sum(marg.values())))
K = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes']
top = sorted(ms, key=lambda m: marg[m], reverse=True)
print('top 10 movies by marginal contribution (share of total delta):')
cum = 0
for m in top[:10]:
    cum += marg[m]
    print('  %-16s %-7s marg %+.6f share %5.1f%% cum %5.1f%% drop1 %+.6f | %s' % (m, base[m]['set'], marg[m], 100 * marg[m] / tot, 100 * cum / tot, drop1[m],
          ' '.join('%s%+d' % (k.replace('division_', 'd').replace('edge_', 'e').replace('num_pred_nodes', 'nod'), cur[m][k] - base[m][k]) for k in K if cur[m][k] != base[m][k])))
print('bottom 5:')
for m in top[-5:]:
    print('  %-16s %-7s marg %+.6f | %s' % (m, base[m]['set'], marg[m], ' '.join('%s%+d' % (k, cur[m][k] - base[m][k]) for k in K if cur[m][k] != base[m][k])))
rng = np.random.default_rng(1)
def boot(sel, n=1000):
    o = []
    for _ in range(n):
        k = rng.integers(0, len(sel), len(sel)); s = [sel[i] for i in k]; o.append(d(s))
    o = np.array(o); return np.quantile(o, .025), np.quantile(o, .975), (o > 0).mean()
for k in [0, 1, 2, 3, 5, 10, 20]:
    sel = [m for m in ms if m not in set(top[:k])]
    lo, hi, p = boot(sel)
    s44 = [m for m in sel if m.startswith('44b6')]; s6b = [m for m in sel if m.startswith('6bba')]
    print('drop top%-2d: all %+.6f CI [%+.6f, %+.6f] P>0 %.3f | 44b6 %+.6f | 6bba %+.6f' % (k, d(sel), lo, hi, p, d(s44), d(s6b)))
for e in ['44b6', '6bba']:
    sel = [m for m in ms if m.startswith(e)]; te = sorted(sel, key=lambda m: marg[m], reverse=True)
    for k in [1, 3, 5]:
        s2 = [m for m in sel if m not in set(te[:k])]; lo, hi, p = boot(s2, 600)
        print('%s drop own top%d: %+.6f CI [%+.6f, %+.6f] P>0 %.3f' % (e, k, d(s2), lo, hi, p))
# node-term decomposition
def adj(r, nodes):
    tp, fp, fn = r['edge_tp'], r['edge_fp'], r['edge_fn']; den = tp + fp + fn
    if den == 0 or not (r['n_total'] > 0): return float('nan')
    return max(0.0, tp / den * (1 - 0.1 * (nodes - r['n_total']) / r['n_total']))
def hyb(cnt, nod):  # counts from row cnt, node number from row nod
    r = dict(cnt); r['num_pred_nodes'] = nod['num_pred_nodes']; r['adj_edge_jaccard'] = adj(r, r['num_pred_nodes']); return r
# check the recomputation first
err = max(abs(adj(r, r['num_pred_nodes']) - r['adj_edge_jaccard']) for r in rows if r['adj_edge_jaccard'] == r['adj_edge_jaccard'])
nodeonly = {m: hyb(base[m], cur[m]) for m in ms}; evalonly = {m: hyb(cur[m], base[m]) for m in ms}
print('adj recompute max err %.2e' % err)
for lab, sel in [('all', ms), ('44b6', [m for m in ms if m.startswith('44b6')]), ('6bba', [m for m in ms if m.startswith('6bba')]),
                 ('clean40', [m for m in ms if base[m]['set'] in ('hold36', 'prev4')])]:
    dn = d(sel, base, nodeonly); de = d(sel, base, evalonly); dt = d(sel)
    lo, hi, p = (lambda o: (np.quantile(o, .025), np.quantile(o, .975), (o > 0).mean()))(np.array([d([sel[i] for i in rng.integers(0, len(sel), len(sel))], base, evalonly) for _ in range(600)]))
    print('%-8s total %+.6f = node-count-only %+.6f + evaluable-counts-only %+.6f (inter %+.6f) | eval-only CI [%+.6f, %+.6f] P>0 %.3f' % (lab, dt, dn, de, dt - dn - de, lo, hi, p))
# edge vs division part of evaluable-only
a = summarise([base[m] for m in ms]); b = summarise([evalonly[m] for m in ms]); c = summarise([cur[m] for m in ms])
print('evalonly: adjE %+.6f  divJ %+.6f (x0.1 = %+.6f) | full: adjE %+.6f divJ*0.1 %+.6f' % (b['adj_edge_jaccard'] - a['adj_edge_jaccard'], b['division_jaccard'] - a['division_jaccard'], 0.1 * (b['division_jaccard'] - a['division_jaccard']), c['adj_edge_jaccard'] - a['adj_edge_jaccard'], 0.1 * (c['division_jaccard'] - a['division_jaccard'])))
print('totals delta:', {k: sum(cur[m][k] - base[m][k] for m in ms) for k in K})
