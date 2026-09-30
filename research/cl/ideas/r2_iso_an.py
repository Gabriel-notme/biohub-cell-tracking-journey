"""Round-2 diagnostic (reads GT): isolation of gap-1 end->start pairs (d<10um) in P14: distance from the end a to the nearest other
node at t, from the start b to the nearest other node at t+1, and to the nearest CONTINUING track node (a node with parent and child)
at t / t+1. Hypothesis: true breaks are isolated, duplicate fragments sit beside a continuing track. Writes r2_iso_rows.json."""
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
    from tracking_cellmot.metrics import evaluate
    from scipy.spatial import cKDTree
    K = evalx.K
    name = Path(f).stem; st = [k for k in RE.FULL if ('_%s/' % k) in f][0]
    fg = RE.FULL[st] + '/' + name + '.geff'
    nodes, edges = evalx.load_graph_json(f)
    nodes, edges, _ = combo14.apply(nodes, edges, fullgeff=fg)
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ea = gt.edge_attrs()
    gsucc = defaultdict(list); gpar = {}
    for a, b in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()):
        gsucc[int(a)].append(int(b)); gpar[int(b)] = int(a)
    succ = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a
    pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) * S for n, v in nodes.items()}
    T = {n: int(v['t']) for n, v in nodes.items()}
    tmax = max(T.values())
    by = defaultdict(list); cont = defaultdict(list); starts = defaultdict(list)
    for n in nodes:
        by[T[n]].append(n)
        if n in par and succ.get(n): cont[T[n]].append(n)
        if n not in par and succ.get(n) and T[n] > 0: starts[T[n]].append(n)
    at = {t: (ns, cKDTree(np.stack([pos[n] for n in ns]))) for t, ns in by.items()}
    ct = {t: (ns, cKDTree(np.stack([pos[n] for n in ns]))) for t, ns in cont.items()}
    stt = {t: (ns, cKDTree(np.stack([pos[n] for n in ns]))) for t, ns in starts.items()}

    def nn_other(tree, t, n, k=2):
        if t not in tree: return 99.
        ns, tr = tree[t]
        dd, jj = tr.query(pos[n], k=min(k, len(ns)))
        for d_, j_ in zip(np.atleast_1d(dd), np.atleast_1d(jj)):
            if ns[int(j_)] != n: return round(float(d_), 2)
        return 99.

    def blen(n):
        c = 1
        while n in par: n = par[n]; c += 1
        return c

    def flen(n):
        c = 1
        while len(succ.get(n, [])) == 1: n = succ[n][0]; c += 1
        return c
    rows = []
    for a in nodes:
        if succ.get(a) or a not in par or T[a] >= tmax or len(succ.get(par[a], [])) == 2: continue
        t = T[a]
        if t + 1 not in stt: continue
        ns, tr = stt[t + 1]
        for j in tr.query_ball_point(pos[a], 10.0):
            b = ns[j]
            ga, gb = p2g.get(a), p2g.get(b)
            tp = ga is not None and gb is not None and gb in gsucc.get(ga, [])
            ev = (ga is not None and bool(gsucc.get(ga))) or (gb is not None and gb in gpar)
            rows.append(dict(movie=name, set=st, d=round(float(np.linalg.norm(pos[a] - pos[b])), 2), lab='TP' if tp else ('FP' if ev else 'U'),
                             iso_a=nn_other(at, t, a), iso_b=nn_other(at, t + 1, b), cont_a=nn_other(ct, t, a), cont_b=nn_other(ct, t + 1, b),
                             la=blen(a), lb=flen(b)))
    return rows


if __name__ == '__main__':
    import rule_eval as RE
    fs = [f for s, d in RE.SRC['p13'].items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(24, maxtasksperchild=4) as p: R_ = p.map(job, fs, chunksize=1)
    rows = [r for rs in R_ for r in rs]
    json.dump(rows, open('/workspace/cl/ideas/r2_iso_rows.json', 'w'))
    from collections import Counter
    CLEAN = ('hold36', 'prev4')

    def tab(x0, title):
        out = '%-48s n=%6d' % (title, len(x0))
        for emb in ['44b6', '6bba', 'clean']:
            x = [r for r in x0 if (r['set'] in CLEAN if emb == 'clean' else r['movie'][:4] == emb)]
            c = Counter(r['lab'] for r in x)
            out += ' | %s TP %d FP %d U %d' % (emb, c['TP'], c['FP'], c['U'])
        print(out)
    for dmax in [4, 7, 10]:
        x = [r for r in rows if r['d'] < dmax]
        tab(x, 'd<%d all' % dmax)
        for iso in [5, 7, 9]:
            tab([r for r in x if r['cont_a'] >= iso and r['cont_b'] >= iso], '  no continuing node within %d of a (t) and b (t+1)' % iso)
            tab([r for r in x if r['iso_a'] >= iso and r['iso_b'] >= iso], '  no node at all within %d of a and b' % iso)
        tab([r for r in x if r['cont_a'] >= 7 and r['cont_b'] >= 7 and r['la'] >= 5 and r['lb'] >= 5], '  cont>=7 & both tracks >=5')
