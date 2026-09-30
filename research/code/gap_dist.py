import sys, json, glob
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
import numpy as np, evalx
from tracking_cellmot.metrics import evaluate
K = evalx.K
res = defaultdict(list)
for path in sorted(glob.glob(sys.argv[1] + '/*.json')):
    name = Path(path).stem
    nodes, edges = evalx.load_graph_json(path)
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: p for p, g in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(n): np.array([z * 1.625, y * .40625, x * .40625]) for n, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    ea = gt.edge_attrs()
    succ = defaultdict(list); par = {}
    for e in edges: s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()):
        s, d = int(s), int(d); gd = float(np.linalg.norm(gpos[d] - gpos[s]))
        ps, pd_ = g2p.get(s), g2p.get(d)
        if ps is None or pd_ is None: cat = 'missing'
        elif pd_ in succ.get(ps, []): cat = 'tp'
        elif not succ.get(ps) and pd_ not in par: cat = 'gapfree'
        else: cat = 'other'
        res[cat].append(gd)
for c, v in res.items():
    v = np.array(v); print(c, len(v), 'gt step um quantiles', np.round(np.quantile(v, [.1, .25, .5, .75, .9, .99]), 2).tolist(), 'frac>8um', round(float((v > 8).mean()), 3))
