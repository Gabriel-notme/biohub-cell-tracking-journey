"""Label every (track end at t) -> (track start at t+1) mutual-nearest pair within 10um on P13 (would the new edge be TP / FP / NE)."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']: os.environ[_k]='1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
def job(f):
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ea = gt.edge_attrs(); gs = defaultdict(list); gp = {}; GE = set()
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(x)].append(int(y)); gp[int(y)] = int(x); GE.add((int(x), int(y)))
    out = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
    pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) * S for n, v in nodes.items()}
    ends = defaultdict(list); starts = defaultdict(list)
    for n, v in nodes.items():
        if not out[n] and int(v['t']) < 99: ends[int(v['t'])].append(n)
        if n not in par and int(v['t']) > 0: starts[int(v['t'])].append(n)
    res = []
    for t, E in ends.items():
        St = starts.get(t + 1)
        if not St: continue
        tr = cKDTree(np.stack([pos[n] for n in St])); te = cKDTree(np.stack([pos[n] for n in E]))
        for a in E:
            d, i = tr.query(pos[a]); b = St[i]
            d2, j = te.query(pos[b])
            if E[j] != a or d > 10: continue
            ga, gb = p2g.get(a), p2g.get(b)
            ev = bool((ga is not None and gs.get(ga)) or (gb is not None and gb in gp))
            lab = 'TP' if (ga is not None and gb is not None and (ga, gb) in GE) else ('FP' if ev else 'NE')
            # length of the ending track and starting track
            L1 = 1; x = a
            while x in par: x = par[x]; L1 += 1
            L2 = 1; x = b
            while len(out[x]) == 1: x = out[x][0]; L2 += 1
            res.append(dict(m=name, d=float(d), lab=lab, L1=L1, L2=L2))
    return res
if __name__ == '__main__':
    fs = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(48) as p: R = p.map(job, fs, chunksize=1)
    R = [r for rs in R for r in rs]
    for emb in ['44b6', '6bba']:
        C = defaultdict(Counter)
        for r in R:
            if r['m'].startswith(emb): C[int(r['d'])][r['lab']] += 1
        print(emb, {k: dict(C[k]) for k in sorted(C)})
    C = defaultdict(Counter)
    for r in R:
        if r['d'] < 5: C[(min(r['L1'], 10) // 5, min(r['L2'], 10) // 5)][r['lab']] += 1
    print('d<5 by track lengths', {k: dict(C[k]) for k in sorted(C)})
