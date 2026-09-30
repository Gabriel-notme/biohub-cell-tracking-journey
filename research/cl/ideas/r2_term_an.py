"""Round-2 diagnostic (reads GT): terminal edges of P14 tracks. For every track END a (parent p, not a fork daughter) the edge p->a,
and for every track START b (single child c) the edge b->c: official label TP/FP(evaluable)/U, step length, motion residual,
pre-ILP prob, flags, distance to nearest same-frame continuing node. Writes /workspace/cl/ideas/r2_term_rows.json."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
sys.path.insert(0, '/workspace/p56stage')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])


def job(f):
    import numcodecs.blosc
    numcodecs.blosc.use_threads = False
    import evalx
    import rule_eval as RE
    from ideas import combo14
    from edge_link import load_full
    from tracking_cellmot.metrics import evaluate
    from scipy.spatial import cKDTree
    K = evalx.K
    name = Path(f).stem; st = [k for k in RE.FULL if ('_%s/' % k) in f][0]
    fg = RE.FULL[st] + '/' + name + '.geff'
    nodes, edges = evalx.load_graph_json(f)
    nodes, edges, _ = combo14.apply(nodes, edges, fullgeff=fg)
    fids, fT, fV, fE, fprob = load_full(fg)
    fedge = {(int(a), int(b)): float(p) for (a, b), p in zip(fE.tolist(), fprob.tolist())}
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ea = gt.edge_attrs()
    gsucc = defaultdict(list); gpar = {}
    for a, b in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()):
        gsucc[int(a)].append(int(b)); gpar[int(b)] = int(a)
    succ = defaultdict(list); par = {}; eflag = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a
        eflag[(a, b)] = ','.join(sorted(k for k in e if k not in ('source_id', 'target_id', 'edge_prob')))
    pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) * S for n, v in nodes.items()}
    T = {n: int(v['t']) for n, v in nodes.items()}
    tmax = max(T.values())

    def lab(x, y):
        gx, gy = p2g.get(x), p2g.get(y)
        if gx is not None and gy is not None and gy in gsucc.get(gx, []): return 'TP'
        ev = (gx is not None and bool(gsucc.get(gx))) or (gy is not None and gy in gpar)
        return 'FP' if ev else 'U'

    def back(n, k=50):
        c = 0
        while n in par and c < k: n = par[n]; c += 1
        return c

    def fwd(n, k=50):
        c = 0
        while len(succ.get(n, [])) == 1 and c < k: n = succ[n][0]; c += 1
        return c
    by = defaultdict(list)
    for n in nodes:
        if succ.get(n): by[T[n]].append(n)
    ctree = {t: (ns, cKDTree(np.stack([pos[n] for n in ns]))) for t, ns in by.items()}
    byp = defaultdict(list)
    for n in nodes:
        if n in par: byp[T[n]].append(n)
    ptree = {t: (ns, cKDTree(np.stack([pos[n] for n in ns]))) for t, ns in byp.items()}
    rows = []
    for a in nodes:
        if succ.get(a) or a not in par or T[a] >= tmax: continue
        p = par[a]
        if len(succ.get(p, [])) == 2: continue
        v = pos[p] - pos[par[p]] if p in par else None
        dm = 99.
        if T[a] in ctree:
            ns, tr = ctree[T[a]]
            dd, jj = tr.query(pos[a], k=2)
            for d_, j_ in zip(np.atleast_1d(dd), np.atleast_1d(jj)):
                if j_ < len(ns) and ns[int(j_)] != a: dm = float(d_); break
        rows.append(dict(side='end', movie=name, set=st, lab=lab(p, a), step=round(float(np.linalg.norm(pos[a] - pos[p])), 2),
                         res=round(float(np.linalg.norm(pos[a] - pos[p] - v)), 2) if v is not None else -1, hist=back(a),
                         fe=fedge.get((p, a), -1.0), flag=eflag.get((p, a), ''), dcont=round(dm, 2), m=int(a in p2g), mp=int(p in p2g)))
    for b in nodes:
        if b in par or len(succ.get(b, [])) != 1 or T[b] == 0: continue
        c = succ[b][0]
        v = pos[succ[c][0]] - pos[c] if len(succ.get(c, [])) == 1 else None
        dm = 99.
        if T[b] in ptree:
            ns, tr = ptree[T[b]]
            dd, jj = tr.query(pos[b], k=2)
            for d_, j_ in zip(np.atleast_1d(dd), np.atleast_1d(jj)):
                if j_ < len(ns) and ns[int(j_)] != b: dm = float(d_); break
        rows.append(dict(side='start', movie=name, set=st, lab=lab(b, c), step=round(float(np.linalg.norm(pos[c] - pos[b])), 2),
                         res=round(float(np.linalg.norm(pos[b] - (pos[c] - v))), 2) if v is not None else -1, hist=fwd(b),
                         fe=fedge.get((b, c), -1.0), flag=eflag.get((b, c), ''), dcont=round(dm, 2), m=int(b in p2g), mp=int(c in p2g)))
    return rows


if __name__ == '__main__':
    import rule_eval as RE
    fs = [f for s, d in RE.SRC['p13'].items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(40, maxtasksperchild=4) as p: R_ = p.map(job, fs, chunksize=1)
    rows = [r for rs in R_ for r in rs]
    json.dump(rows, open('/workspace/cl/ideas/r2_term_rows.json', 'w'))
    print('done', len(rows))
