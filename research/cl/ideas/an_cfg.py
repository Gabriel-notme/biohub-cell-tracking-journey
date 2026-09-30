"""Configuration anatomy (analysis only): motion-field residual of each evaluable edge, spike of nodes, fork stubs with division labels."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']: os.environ[_k]='1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
def job(f):
    import evalx
    from tracking_cellmot.metrics import evaluate
    from tracking_cellmot.division_metrics import score_divisions
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
    pred2, mapping2 = evalx.to_graph(nodes, edges); inv2 = {v: k for k, v in mapping2.items()}
    ds = score_divisions(pred2, gt, scale=evalx.SCALE, max_distance=7.)
    tpf = {inv2[i] for i in ds.tp_forks}; fpf = {inv2[i] for i in ds.fp_forks}
    out = defaultdict(list); par = {}
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id']); out[x].append(y); par[y] = x
    pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) * S for n, v in nodes.items()}
    byt = defaultdict(list)
    for n, v in nodes.items():
        if len(out[n]) == 1: byt[int(v['t'])].append(n)
    trees = {t: (cKDTree(np.stack([pos[n] for n in ns])), ns) for t, ns in byt.items() if ns}
    def lab(x, y):
        gx, gy = p2g.get(x), p2g.get(y)
        ev = bool((gx is not None and gs.get(gx)) or (gy is not None and gy in gp))
        if gx is not None and gy is not None and (gx, gy) in GE: return 'TP'
        return 'FP' if ev else 'NE'
    rows = []
    for x in list(out):
        for y in out[x]:
            L = lab(x, y)
            t = int(nodes[x]['t']); tr, ns = trees[t]
            k = min(9, len(ns)); dd, ii = tr.query(pos[x], k=k); dd = np.atleast_1d(dd); ii = np.atleast_1d(ii)
            V = [pos[out[ns[i]][0]] - pos[ns[i]] for d, i in zip(dd, ii) if ns[i] != x and d < 25]
            v = pos[y] - pos[x]
            res = float(np.linalg.norm(v - np.median(V, 0))) if len(V) >= 3 else -1.
            # spike of y: deviation from midpoint of x and y's child
            sp = -1.; ab = -1.
            if len(out[y]) == 1:
                c = out[y][0]; sp = float(np.linalg.norm(pos[y] - (pos[x] + pos[c]) / 2)); ab = float(np.linalg.norm(pos[c] - pos[x]))
            if L == 'NE' and np.random.rand() > 0.03: continue
            rows.append(dict(m=name, lab=L, res=res, nv=len(V), disp=float(np.linalg.norm(v)), sp=sp, ab=ab, fork=len(out[x]) == 2))
    # forks
    def blen(c):
        L = 1
        while len(out[c]) == 1: c = out[c][0]; L += 1
        return L, len(out[c])
    forks = []
    for x in out:
        if len(out[x]) != 2: continue
        bl = [blen(c) for c in out[x]]
        forks.append(dict(m=name, lab='TP' if x in tpf else ('FP' if x in fpf else 'NE'), bl=bl, t=int(nodes[x]['t']),
                          cd=float(np.linalg.norm(pos[out[x][0]] - pos[out[x][1]])), plen=0))
    return rows, forks
if __name__ == '__main__':
    np.random.seed(0)
    fs = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(60) as p: R = p.map(job, fs, chunksize=1)
    json.dump({'rows': [r for a, b in R for r in a], 'forks': [r for a, b in R for r in b]}, open('/workspace/cl/ideas/an/p13_cfg.json', 'w'))
    print('done')
