import os, sys, json, glob
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
import numpy as np, zarr
from scipy.spatial import cKDTree
S = np.array([1.625, .40625, .40625])
GD = sys.argv[1]; FULL = sys.argv[2]
def job(p):
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    name = Path(p).stem
    nodes, edges = evalx.load_graph_json(p)
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    g2p = {int(b): inv[int(a)] for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): (int(t), np.array([z, y, x]) * S) for i, t, z, y, x in zip(*[ga[c].to_list() for c in [K.NODE_ID, 't', 'z', 'y', 'x']])}
    ea = gt.edge_attrs(); gs = defaultdict(list)
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(s)].append(int(d))
    g = zarr.open_group(FULL + '/%s.geff' % name, mode='r')
    P = {k: np.asarray(g['nodes/props/%s/values' % k][:]) for k in 'tzyx'}
    FT = P['t'].astype(int); FP = np.stack([P['z'], P['y'], P['x']], 1) * S
    out = []
    for m, kids in gs.items():
        if len(kids) != 2: continue
        miss = [c for c in [m] + kids if c not in g2p]
        if not miss: continue
        for c in miss:
            t, q = gpos[c]; sel = FT == t
            d = float(np.min(np.linalg.norm(FP[sel] - q, axis=1))) if sel.any() else 99
            out.append(dict(movie=name, role='mother' if c == m else 'daughter', full_dist=round(d, 2)))
    return out
if __name__ == '__main__':
    ps = sorted(glob.glob(GD + '/*.json'))
    with Pool(36) as pool: res = [r for rr in pool.map(job, ps) for r in rr]
    for r in res: print(r)
