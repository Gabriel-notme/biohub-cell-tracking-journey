import os, sys, json, glob
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
import numpy as np, zarr
from scipy.spatial import cKDTree
S = np.array([1.625, .40625, .40625])
GD, FULL = sys.argv[1], sys.argv[2]
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
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: q for q, g in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): (int(t), np.array([z, y, x]) * S) for i, t, z, y, x in zip(*[ga[c].to_list() for c in [K.NODE_ID, 't', 'z', 'y', 'x']])}
    ea = gt.edge_attrs(); gs = defaultdict(list); gp = {}
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(s)].append(int(d)); gp[int(d)] = int(s)
    succ = defaultdict(list); par = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    g = zarr.open_group(FULL + '/%s.geff' % name, mode='r')
    FT = np.asarray(g['nodes/props/t/values'][:]).astype(int); FP = np.stack([np.asarray(g['nodes/props/%s/values' % k][:]) for k in 'zyx'], 1) * S
    ftree = {t: cKDTree(FP[FT == t]) for t in np.unique(FT)}
    c = Counter()
    for gnode, (t, q) in gpos.items():
        if gnode in g2p: continue
        a = gp.get(gnode); kids = gs.get(gnode, [])
        pu = g2p.get(a) if a is not None else None
        pw = g2p.get(kids[0]) if len(kids) == 1 else None
        dmin = ftree[t].query(q)[0] if t in ftree else 99
        has_d = dmin <= 5
        if pu is None and pw is None: c['no_matched_neighbors'] += 1; continue
        # current pred successor of pu
        x = None
        if pu is not None and len(succ.get(pu, [])) == 1: x = succ[pu][0]
        elif pw is not None and pw in par: x = par[pw]
        if x is None: c['ends_or_starts' + ('_d' if has_d else '')] += 1; continue
        xint = (x in par) and len(succ.get(x, [])) == 1
        same = (pu is None or par.get(x) == pu) and (pw is None or pw in succ.get(x, []))
        key = ('interior_' if xint else 'boundary_') + ('both' if (pu is not None and pw is not None and same) else 'partial') + ('_d' if has_d else '_nod') + ('_xmatched' if x in p2g else '')
        c[key] += 1
    return c
if __name__ == '__main__':
    ps = sorted(glob.glob(GD + '/*.json'))
    with Pool(36) as pool: cs = pool.map(job, ps)
    tot = Counter()
    for c in cs: tot.update(c)
    for k, v in tot.most_common(): print(k, v)
