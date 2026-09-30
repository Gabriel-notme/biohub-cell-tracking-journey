import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/official/src')
import numpy as np
from tracking_cellmot.metrics import summarise
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '/workspace/cl/p16/rows_p17_full.json'))
base = {r['movie']: r for r in R if r['vi'] == 0}; cur = {r['movie']: r for r in R if r['vi'] == 1}
ms = sorted(base)
EK = ['edge_tp', 'edge_fp', 'edge_fn']; DK = ['division_tp', 'division_fp', 'division_fn']

def mk(b, e_src, d_src, n_src):
    r = dict(b)
    for k in EK: r[k] = e_src[k]
    for k in DK: r[k] = d_src[k]
    r['num_pred_nodes'] = n_src['num_pred_nodes']
    nt = r['n_total']; ratio = (r['num_pred_nodes'] - nt) / nt
    den = r['edge_tp'] + r['edge_fp'] + r['edge_fn']; J = r['edge_tp'] / den if den else float('nan')
    r['edge_jaccard'] = J; r['total_node_ratio'] = ratio; r['adj_edge_jaccard'] = max(0., J * (1 - 0.1 * ratio)) if J == J else float('nan')
    return r

# sanity: recomputing P17 rows from P17 counts reproduces stored adj
mx = max(abs(mk(cur[m], cur[m], cur[m], cur[m])['adj_edge_jaccard'] - cur[m]['adj_edge_jaccard']) for m in ms if cur[m]['adj_edge_jaccard'] == cur[m]['adj_edge_jaccard'])
print('recompute max abs err', mx)
V = {
 'full (P17)':            lambda m: cur[m],
 'node-only':             lambda m: mk(base[m], base[m], base[m], cur[m]),
 'edge-only':             lambda m: mk(base[m], cur[m], base[m], base[m]),
 'div-only':              lambda m: mk(base[m], base[m], cur[m], base[m]),
 'edge+div (P15 nodes)':  lambda m: mk(base[m], cur[m], cur[m], base[m]),
 'edge+node (P15 div)':   lambda m: mk(base[m], cur[m], base[m], cur[m]),
}
rng = np.random.default_rng(0); K = [rng.integers(0, len(ms), len(ms)) for _ in range(1000)]
def d(fn, sel): return summarise([fn(m) for m in sel])['score'] - summarise([base[m] for m in sel])['score']
out = {}
for name, fn in V.items():
    rows = {m: fn(m) for m in ms}
    f2 = lambda m: rows[m]
    e44 = [m for m in ms if m.startswith('44b6')]; e6 = [m for m in ms if m.startswith('6bba')]
    c40 = [m for m in ms if base[m]['set'] in ('hold36', 'prev4')]
    bs = np.array([summarise([rows[ms[j]] for j in k])['score'] - summarise([base[ms[j]] for j in k])['score'] for k in K])
    sa = summarise([rows[m] for m in ms]); sb = summarise([base[m] for m in ms])
    out[name] = dict(all=d(f2, ms), e44=d(f2, e44), e6=d(f2, e6), c40=d(f2, c40), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), p_le0=float((bs <= 0).mean()),
                     adj=sa['adj_edge_jaccard'] - sb['adj_edge_jaccard'], divj=0.1 * (sa['division_jaccard'] - sb['division_jaccard']))
    print('%-22s all %+.5f CI [%+.5f,%+.5f] P(<=0) %.3f | 44b6 %+.5f 6bba %+.5f clean40 %+.5f | adj %+.5f 0.1*divJ %+.5f' % (name, *[out[name][k] for k in ['all', 'lo', 'hi', 'p_le0', 'e44', 'e6', 'c40', 'adj', 'divj']]))
# totals
for k in EK + DK + ['num_pred_nodes']:
    print(k, sum(cur[m][k] - base[m][k] for m in ms))
sb = summarise([base[m] for m in ms]); print('P15 divJ %.4f div tp/fp/fn %d/%d/%d  edgeJ %.5f adj %.5f score %.5f' % (sb['division_jaccard'], sb['division_tp'], sb['division_fp'], sb['division_fn'], sb['edge_jaccard'], sb['adj_edge_jaccard'], sb['score']))
chg_e = [m for m in ms if any(cur[m][k] != base[m][k] for k in EK)]; chg_d = [m for m in ms if any(cur[m][k] != base[m][k] for k in DK)]
chg_n = [m for m in ms if cur[m]['num_pred_nodes'] != base[m]['num_pred_nodes']]
print('movies with edge-count change', len(chg_e), 'div change', len(chg_d), 'node change', len(chg_n))
for m in chg_d: print('  div', m, base[m]['set'], {k: cur[m][k] - base[m][k] for k in EK + DK + ['num_pred_nodes']})
print('edge-change movies:')
for m in chg_e: print('  ', m, base[m]['set'], {k: cur[m][k] - base[m][k] for k in EK + DK + ['num_pred_nodes']})
json.dump(out, open('decomp_out.json', 'w'), indent=1)
