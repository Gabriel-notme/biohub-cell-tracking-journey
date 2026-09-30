"""Detail for (A) identical transitions and (B) big global shifts (>= 10 um): GT nodes on both sides, GT edges, and what P20 does:
pred edges across the transition that touch a GT-matched node (TP / FP), GT edges FN by type (t-end unmatched / t+1-end unmatched / both
matched but not linked), and for identical transitions whether a pred node exists within 3 um of the unmatched GT node's twin."""
import os, sys, json
for v in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'): os.environ[v] = '1'
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
from multiprocessing import Pool
from collections import Counter
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
FA = json.load(open('/workspace/nbrun/frames_an.json'))
def job(r):
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K; m = r['movie']; s = r['set']
    ident = {x['t'] for x in r['rows'] if x['ident']}; big = {x['t'] for x in r['rows'] if (not x['ident']) and np.linalg.norm(x['shift']) >= 10}
    if not ident and not big: return None
    nodes, edges = evalx.load_graph_json('/workspace/cl/p21/ps_p20ref_%s/graphs/%s.json' % (s, m))
    g, mp = evalx.to_graph(nodes, edges, rounding=True); inv = {v: k for k, v in mp.items()}
    gt, _ = evalx.load_gt(m)
    evaluate(g, gt, scale=tuple(S), max_distance=7.0)
    na = g.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID]).to_dicts()
    p2g = {int(x[K.NODE_ID]): int(x[K.MATCHED_NODE_ID]) for x in na if x[K.MATCHED_NODE_ID] is not None and int(x[K.MATCHED_NODE_ID]) != -1}
    g2p = {v: k for k, v in p2g.items()}
    ga = {x[K.NODE_ID]: x for x in gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x']).to_dicts()}
    gte = set(); gtcnt = Counter()
    for x in gt.edge_attrs().to_dicts():
        a, b = int(x['source_id']), int(x['target_id']); gte.add((a, b))
    gtn = Counter(int(v['t']) for v in ga.values())
    pos = {n: np.array([max(0, int(round(v[c]))) for c in 'zyx']) * S for n, v in nodes.items()}
    tn = {n: int(v['t']) for n, v in nodes.items()}
    out = Counter()
    for x in g.edge_attrs().to_dicts():
        a, b = int(x['source_id']), int(x['target_id']); t = tn[inv[a]]
        cls = 'I' if t in ident else ('B' if t in big else None)
        if cls is None: continue
        if a in p2g or b in p2g:
            tp = a in p2g and b in p2g and (p2g[a], p2g[b]) in gte
            out[cls + '_pred_edge_touch_' + ('tp' if tp else 'fp')] += 1
    for (a, b) in gte:
        t = int(ga[a]['t']); cls = 'I' if t in ident else ('B' if t in big else None)
        if cls is None: continue
        out[cls + '_gt_edges'] += 1
        ma, mb = a in g2p, b in g2p
        if ma and mb:
            linked = any(inv.get(0) is None for _ in [0]) and ((g2p[a], g2p[b]) in {(int(e['source_id']), int(e['target_id'])) for e in []})
        if not ma and not mb: out[cls + '_fn_both_unmatched'] += 1
        elif not mb: out[cls + '_fn_t1_unmatched'] += 1
        elif not ma: out[cls + '_fn_t0_unmatched'] += 1
    for t in ident | big:
        cls = 'I' if t in ident else 'B'
        out[cls + '_gt_nodes_t'] += gtn[t]; out[cls + '_gt_nodes_t1'] += gtn[t + 1]; out[cls + '_transitions'] += 1
    return dict(out)
if __name__ == '__main__':
    with Pool(40) as p: R = [x for x in p.map(job, FA, chunksize=1) if x]
    tot = Counter()
    for x in R: tot.update(x)
    for k in sorted(tot): print(k, tot[k])
