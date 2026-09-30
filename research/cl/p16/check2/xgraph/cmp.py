"""(1) official rule_eval rows (re_rows_<src>.json) vs xg rows: identical counts per movie for base and full?
(2) selectivity: fraction of base predicted nodes that are matched to a GT node vs fraction among deleted nodes (per rule)."""
import os, sys, json
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
from multiprocessing import Pool
import warnings; warnings.filterwarnings('ignore')
D = '/workspace/cl/p16/check2/xgraph/'
K = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes']
srcs = sys.argv[1].split(',')
for s in srcs:
    p = D + 're_rows_%s.json' % s
    if not os.path.exists(p): print(s, 'no rule_eval rows yet'); continue
    re_ = {(r['movie'], r['vi']): r for r in json.load(open(p))}
    xg = {(r['movie'], r['vi']): r for r in json.load(open(D + 'rows_%s.json' % s)) if r['vi'] in (0, 1)}
    bad = [k for k in xg if any(xg[k][c] != re_[k][c] for c in K) or abs(xg[k]['adj_edge_jaccard'] - re_[k]['adj_edge_jaccard']) > 1e-12]
    print('%s: rule_eval rows %d, xg rows %d, mismatching (movie,vi) %d' % (s, len(re_), len(xg), len(bad)), bad[:5])


def gtn(m):
    import evalx
    gt, _ = evalx.load_gt(m)
    return m, gt.num_nodes()


if len(sys.argv) > 2:
    s = sys.argv[2]
    rows = {r['movie']: r for r in json.load(open(D + 'rows_%s.json' % s)) if r['vi'] == 0}
    with Pool(6) as p: G = dict(p.map(gtn, sorted(rows)))
    det = json.load(open(D + 'det_%s.json' % s))
    matched = sum(rows[m]['node_recall'] * G[m] for m in rows); tot = sum(rows[m]['num_pred_nodes'] for m in rows)
    print('%s base: predicted nodes %d, matched to GT ~%d (%.2f%%), GT nodes %d' % (s, tot, matched, 100 * matched / tot, sum(G.values())))
    for ru in ['cd', 'ff', 'st', 'tt', 'par', 'border']:
        n = sum(d['rules'][ru]['n_rm'] for d in det); k = sum(d['rules'][ru]['n_rm_matched'] for d in det)
        print('  %-6s deleted %5d, matched %3d (%.2f%%)' % (ru, n, k, 100 * k / max(n, 1)))
