"""Potential of longer gap bridging on P11 outputs: for GT lineage paths, find (pred track end matched to g_t) -> (pred track start matched to
g_{t+k}) where GT nodes g_{t+1..t+k-1} are unmatched. Bridging with k-1 interpolated nodes could recover up to k GT edges.
Also measure how far the linear interpolation lands from the unmatched GT nodes (must be < 7 um to count)."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])


def job(f):
    name = Path(f).stem
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    g2p = {v: k for k, v in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x]) * S for i, z, y, x in zip(*[ga[k].to_list() for k in [K.NODE_ID, 'z', 'y', 'x']])}
    ea = gt.edge_attrs(); gch = defaultdict(list)
    for a, b in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gch[int(a)].append(int(b))
    ch = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); ch[a].append(b); par[b] = a
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    out = []
    for g0, pn in g2p.items():
        if ch.get(pn): continue  # pred track does not end here
        # walk GT forward through unmatched nodes (single-child chain)
        path = [g0]; x = g0
        for k in range(1, 8):
            kids = gch.get(x, [])
            if len(kids) != 1: break
            x = kids[0]; path.append(x)
            if x in g2p:
                pe = g2p[x]
                if pe in par: break  # target already has a parent
                # interpolation error for the missing GT nodes
                errs = [float(np.linalg.norm(pos[pn] + (pos[pe] - pos[pn]) * j / k - gpos[path[j]])) for j in range(1, k)]
                out.append((k, max(errs) if errs else 0.0))
                break
    return out


if __name__ == '__main__':
    fs = [f for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b'] for f in sorted(glob.glob('/workspace/cl/ps_p11_%s/graphs/*.json' % s))]
    with Pool(96) as p: R = [x for xs in p.map(job, fs) for x in xs]
    c = Counter(k for k, e in R); ok = Counter(k for k, e in R if e < 7)
    for k in sorted(c): print('gap %d: pairs %d, interpolation within 7um for all missing nodes %d  (recoverable GT edges <= %d)' % (k, c[k], ok[k], ok[k] * k))
