import os, sys, json
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
import numpy as np
import evalx
from tracking_cellmot.metrics import evaluate
from concurrent.futures import ProcessPoolExecutor
K = evalx.K
S = np.array([1.625, .40625, .40625])
def one(p):
    name = Path(p).stem
    nodes, edges = evalx.load_graph_json(p)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: q for q, g in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x]) * S for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    gt_t = {int(i): int(t) for i, t in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list())}
    ea = gt.edge_attrs(); gs = defaultdict(list); gp = {}
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(s)].append(int(d)); gp[int(d)] = int(s)
    succ = defaultdict(list); par = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s
    # index pred nodes by frame
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    arr = {t: (np.array(ns), np.array([[nodes[n]['z'], nodes[n]['y'], nodes[n]['x']] for n in ns]) * S) for t, ns in byt.items()}
    cat = Counter(); dists = []
    for g in gpos:
        if g in g2p: continue
        t = gt_t[g]
        # nearest pred node at t
        ns, P = arr.get(t, (np.array([]), np.zeros((0, 3))))
        dmin = float(np.min(np.linalg.norm(P - gpos[g], axis=1))) if len(ns) else 99.
        dists.append(dmin)
        # predecessor / successor in GT matched?
        gpr = gp.get(g); gsu = gs.get(g, [])
        ppr = g2p.get(gpr) if gpr is not None else None
        psu = [g2p.get(x) for x in gsu if g2p.get(x) is not None]
        free_end = ppr is not None and len(succ.get(ppr, [])) == 0
        free_start = any(par.get(x) is None for x in psu)
        key = ('near<10' if dmin < 10 else 'far') + '|' + ('prevEnd' if free_end else ('prevM' if ppr is not None else 'noPrev')) + '|' + ('nextStart' if free_start else ('nextM' if psu else 'noNext'))
        cat[key] += 1
    return name, dict(cat), len(gpos), len(g2p), dists
if __name__ == '__main__':
    files = sorted(Path(sys.argv[1]).glob('*.json'))
    with ProcessPoolExecutor(36) as ex: res = list(ex.map(one, [str(f) for f in files]))
    tot = Counter(); ng = 0; nm = 0; D = []
    for name, c, a, b, d in res: tot.update(c); ng += a; nm += b; D += d
    print('gt nodes', ng, 'matched', nm, 'unmatched', ng - nm)
    for k, v in tot.most_common(): print(k, v)
    D = np.array(D); print('nearest pred dist quantiles', np.round(np.quantile(D, [.1, .25, .5, .75, .9]), 2).tolist())
