"""READ-ONLY (critic A): spatial-boundary localisation. For GT nodes near the z top/bottom or the xy border (plus a 5% interior
sample): matched?, signed offset (pred - GT, um) to the matched pred node, else to the nearest pred node; whether that nearest pred
node is itself unmatched. Summary printed per embryo."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']


def job(f):
    import evalx
    from tracking_cellmot.metrics import evaluate
    from scipy.spatial import cKDTree
    K = evalx.K
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: p for p, g in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    ppos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx'], float) for n, v in nodes.items()}
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    trees = {t: (cKDTree(np.stack([ppos[n] for n in ns]) * S), ns) for t, ns in byt.items()}
    rng = np.random.default_rng(0)
    rows = []
    for i, t, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list()):
        g = int(i); gp = np.array([z, y, x], float)
        bxy = float(min(y, x, 255 - y, 255 - x) * S[1])
        zone = 'z0' if z <= 2 else ('z63' if z >= 61 else ('xy' if bxy <= 3 else None))
        if zone is None:
            if rng.random() > 0.05: continue
            zone = 'mid'
        if g in g2p:
            q = g2p[g]; m = 1; qm = 1
        else:
            if int(t) not in trees: continue
            d, k = trees[int(t)][0].query(gp * S, k=1); q = trees[int(t)][1][k]; m = 0; qm = int(q in p2g)
        off = (ppos[q] - gp) * S
        rows.append((zone, m, qm, round(float(np.linalg.norm(off)), 2), *[round(float(a), 2) for a in off], round(bxy, 1), round(float(z), 1)))
    return dict(emb=name[:4], rows=rows)


if __name__ == '__main__':
    files = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p14_%s/graphs/*.json' % s))]
    with Pool(24, maxtasksperchild=4) as p: R = p.map(job, files, chunksize=1)
    for emb in ['44b6', '6bba']:
        rr = [x for r in R if r['emb'] == emb for x in r['rows']]
        print('==', emb)
        for zone in ['z0', 'z63', 'xy', 'mid']:
            m = np.array([x[3:7] for x in rr if x[0] == zone and x[1] == 1]); u = [x for x in rr if x[0] == zone and x[1] == 0]
            ua = np.array([x[3:7] for x in u]) if u else np.zeros((0, 4))
            n = len(m) + len(u)
            print('%-4s n %5d unm %.4f | matched: d %.2f dz %+.2f dy %+.2f dx %+.2f |dz| %.2f | unmatched: nearest d med %.2f, dz %+.2f |dz| %.2f |dxy| %.2f, nearest-is-unmatched %d/%d' % (
                zone, n, len(u) / max(n, 1), m[:, 0].mean(), m[:, 1].mean(), m[:, 2].mean(), m[:, 3].mean(), np.abs(m[:, 1]).mean(),
                np.median(ua[:, 0]) if len(ua) else -1, ua[:, 1].mean() if len(ua) else 0, np.abs(ua[:, 1]).mean() if len(ua) else 0,
                np.sqrt(ua[:, 2] ** 2 + ua[:, 3] ** 2).mean() if len(ua) else 0, sum(1 for x in u if x[2] == 0), len(u)))
