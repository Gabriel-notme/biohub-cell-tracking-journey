"""Write per-rule incremental row files (vi=0 previous stage, vi=1 this stage) and edge-only / node-only hybrid rows for strict.py.
usage: mkrule.py <src>  -> rule_<src>_<rule>.json, hyb_<src>_edgeonly.json (vi=1 = base nodes + full evaluable counts)"""
import sys, json
D = '/workspace/cl/p16/check2/xgraph/'
src = sys.argv[1]
R = json.load(open(D + 'rows_%s.json' % src))
by = {}
for r in R: by.setdefault(r['stage'], {})[r['movie']] = r
O = ['base', 'cd', 'ff', 'st', 'tt', 'par', 'border']
for i in range(1, len(O)):
    a, c = by[O[i - 1]], by[O[i]]
    rows = [dict(a[m], vi=0) for m in a] + [dict(c[m], vi=1) for m in c]
    json.dump(rows, open(D + 'rule_%s_%s.json' % (src, O[i]), 'w'))
KE = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn']
b, f = by['base'], by['full']
def mk(bb, cc):
    r = dict(bb)
    for k in KE: r[k] = cc[k]
    nt = r['n_total']; ratio = (r['num_pred_nodes'] - nt) / nt
    den = r['edge_tp'] + r['edge_fp'] + r['edge_fn']; J = r['edge_tp'] / den if den else float('nan')
    r['edge_jaccard'] = J; r['total_node_ratio'] = ratio; r['adj_edge_jaccard'] = max(0., J * (1 - 0.1 * ratio)) if J == J else float('nan')
    return r
rows = [dict(b[m], vi=0) for m in b] + [dict(mk(b[m], f[m]), vi=1) for m in b]
json.dump(rows, open(D + 'hyb_%s_edgeonly.json' % src, 'w'))
print('ok', src)
