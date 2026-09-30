"""Probe: track-end forward extensions / track-start backward extensions through dropped pre-ILP detections."""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, .40625, .40625])
SET = sys.argv[1] if len(sys.argv) > 1 else 'hold36'
G = {'hold36': ('/workspace/hold36.txt', '/workspace/cl/out_el40/hold36', '/workspace/runs/fullgraph_hold36'),
     't127a': ('/workspace/t127a.txt', '/workspace/sync4/runs/fin_p3_t127a/graphs', '/workspace/sync4/runs/fullgraph_t127a')}


def job(a):
    name, gdir, fdir = a
    import evalx, edge_link
    K = evalx.K
    nodes, edges = evalx.load_graph_json(Path(gdir) / (name + '.json'))
    fids, fT, fV, fE, fprob = edge_link.load_full(Path(fdir) / (name + '.geff'))
    fpos = {int(i): v * S for i, v in zip(fids.tolist(), fV)}
    succ = defaultdict(list); par = {}
    for e in edges: succ[int(e['source_id'])].append(int(e['target_id'])); par[int(e['target_id'])] = int(e['source_id'])
    fsucc = defaultdict(list); fpar = defaultdict(list)
    for (u, v), p in zip(fE.tolist(), fprob.tolist()): fsucc[u].append((v, p)); fpar[v].append((u, p))
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    from tracking_cellmot.metrics import evaluate
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    matched_g = set(p2g.values())
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x]) * S for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    ea = gt.edge_attrs(); gsucc = defaultdict(list); gpar = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(x)].append(int(y)); gpar[int(y)] = int(x)
    c = Counter()
    for u in nodes:
        if succ.get(u): continue
        for v, p in fsucc.get(u, []):
            if v in nodes: continue
            c['fwd'] += 1
            gu = p2g.get(u)
            if gu is None or not gsucc.get(gu): c['fwd_u_unannot'] += 1; continue
            ok = any(g not in matched_g and np.linalg.norm(gpos[g] - fpos[v]) <= 7 for g in gsucc[gu])
            c['fwd_pos' if ok else 'fwd_neg'] += 1
    for v in nodes:
        if v in par: continue
        for u, p in fpar.get(v, []):
            if u in nodes: continue
            c['bwd'] += 1
            gv = p2g.get(v)
            if gv is None or gv not in gpar: c['bwd_v_unannot'] += 1; continue
            g = gpar[gv]
            ok = g not in matched_g and np.linalg.norm(gpos[g] - fpos[u]) <= 7
            c['bwd_pos' if ok else 'bwd_neg'] += 1
    return c


if __name__ == '__main__':
    lst, g, f = G[SET]
    names = [l.strip() for l in open(lst) if l.strip()]
    with Pool(36) as pool: cs = pool.map(job, [(n, g, f) for n in names])
    tot = Counter()
    for c in cs: tot.update(c)
    print(SET, dict(tot))
