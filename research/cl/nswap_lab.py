"""Build labelled node-substitution candidates for a set: label +w if n is unmatched and c lies within 7um of an unmatched GT node
(w = its GT degree), -w if n is matched and c is >7um from n's GT node (w = that GT node's degree), 0 otherwise.
usage: nswap_lab.py <tag> <graph_dir> <full_dir>  -> /workspace/cl/ns/<tag>.npz"""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, 0.40625, 0.40625])


def job(args):
    tag, f, fdir = args
    name = Path(f).stem
    fp = Path(fdir) / (name + '.geff')
    if not fp.exists() or not Path('/workspace/data/train/%s.geff/nodes/props' % name).exists(): return None
    import evalx, nswap
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    try: gt, _ = evalx.load_gt(name)
    except Exception: return None
    full = nswap.load_full(fp)
    rows = nswap.candidates(nodes, edges, full)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    matched_g = set(p2g.values())
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gid = np.array([int(x) for x in ga[K.NODE_ID].to_list()]); gT = np.array(ga['t'].to_list()); gP = np.stack([ga[k].to_numpy() for k in 'zyx'], 1) * S
    gidx = {g: i for i, g in enumerate(gid.tolist())}
    ea = gt.edge_attrs(); deg = defaultdict(int)
    for a, b in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): deg[int(a)] += 1; deg[int(b)] += 1
    gtree = {t: (np.flatnonzero(gT == t), cKDTree(gP[gT == t])) for t in np.unique(gT)}
    fids, fT, fV, fE, fprob = full
    out = []
    for n, ci, x in rows:
        t = int(nodes[n]['t']); pc = np.array([max(0, int(round(v))) for v in fV[ci]], float) * S
        lab = 0.
        gn = p2g.get(n)
        if gn is not None:
            if np.linalg.norm(pc - gP[gidx[gn]]) > 7: lab = -max(1, deg[gn])
        elif t in gtree:
            ix, tr = gtree[t]
            for j in tr.query_ball_point(pc, 7.0):
                g = int(gid[ix[j]])
                if g not in matched_g and deg[g] > 0: lab = max(lab, deg[g])
        out.append(x + [lab])
    return tag, name, out


if __name__ == '__main__':
    tag, gdir, fdir = sys.argv[1], sys.argv[2], sys.argv[3]
    jobs = [(tag, f, fdir) for f in sorted(glob.glob(gdir + '/*.json'))]
    with Pool(64) as p: R = [r for r in p.map(job, jobs) if r]
    X = np.array([row for _, _, rs in R for row in rs], float)
    mv = np.array([m for _, m, rs in R for _ in rs])
    Path('/workspace/cl/ns').mkdir(exist_ok=True)
    np.savez_compressed('/workspace/cl/ns/%s.npz' % tag, X=X, movie=mv)
    y = X[:, -1]
    print(tag, 'movies', len(R), 'rows', len(X), 'pos', int((y > 0).sum()), 'pos_w', y[y > 0].sum(), 'neg', int((y < 0).sum()), 'neg_w', -y[y < 0].sum())
