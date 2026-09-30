"""Round-2 FREE-END lens (diagnostic only, reads GT): anatomy of P14 track ends / starts and label statistics of every
end->start pair at gap 1 and gap 2 within 30 um. P14 = P13 graph + ideas.combo14 (identical to ps_p14 graphs).
Writes /workspace/cl/ideas/r2_fe_rows.json."""
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
RMAX = 30.0


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
    fnode = set(int(i) for i in fids.tolist())
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: p for p, g in p2g.items()}
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

    def tlen(n):
        L = 1; m = n
        while m in par: m = par[m]; L += 1
        m = n
        while succ.get(m): m = succ[m][0]; L += 1
        return L

    def back(n, k=30):
        c = 0
        while n in par and c < k: n = par[n]; c += 1
        return c

    def fwd(n, k=30):
        c = 0
        while succ.get(n) and c < k: n = succ[n][0]; c += 1
        return c

    def vel_end(n):
        return pos[n] - pos[par[n]] if n in par else None

    def vel_start(n):
        return pos[succ[n][0]] - pos[n] if succ.get(n) else None

    ends = [n for n in nodes if not succ.get(n) and T[n] < tmax]
    starts = [n for n in nodes if n not in par and T[n] > 0]
    sby = defaultdict(list)
    for n in starts: sby[T[n]].append(n)
    stree = {t: (ns, cKDTree(np.stack([pos[n] for n in ns]))) for t, ns in sby.items()}
    allby = defaultdict(list)
    for n in nodes: allby[T[n]].append(n)
    atree = {t: (ns, cKDTree(np.stack([pos[n] for n in ns]))) for t, ns in allby.items()}

    def gstat(n, side):
        g = p2g.get(n)
        if g is None: return 'unm', None
        if side == 'end':
            if not gsucc.get(g): return 'gt_end', g
            return 'gt_cont', g
        if g not in gpar: return 'gt_start', g
        return 'gt_cont', g

    erows = []
    for a in ends:
        cls, g = gstat(a, 'end')
        r = dict(movie=name, set=st, n=a, t=T[a], hist=back(a), fdau=int(a in par and len(succ.get(par[a], [])) == 2), cls=cls)
        if cls == 'gt_cont':
            gc = gsucc[g][0]; pc = g2p.get(gc)
            if pc is None:
                r['sub'] = 'child_unm'
                ggc = gsucc.get(gc, [None])[0] if gsucc.get(gc) else None
                pgc = g2p.get(ggc) if ggc is not None else None
                r['sub2'] = 'none' if pgc is None else ('start' if pgc not in par else 'held')
            else:
                r['sub'] = 'child_start' if pc not in par else 'child_held'
                r['dc'] = float(np.linalg.norm(pos[pc] - pos[a]))
                if pc in par:
                    q = par[pc]; r['held_by_end'] = 0
                    r['held_dpar'] = float(np.linalg.norm(pos[q] - pos[a]))
                    r['held_pmatch'] = 'unm' if q not in p2g else ('gt' if gsucc.get(p2g[q]) else 'gtend')
                r['fe_c'] = fedge.get((a, pc), -1.0)
        erows.append(r)
    srows = []
    for b in starts:
        cls, g = gstat(b, 'start')
        r = dict(movie=name, set=st, n=b, t=T[b], fut=fwd(b), cls=cls)
        if cls == 'gt_cont':
            gp = gpar[g]; pp = g2p.get(gp)
            r['sub'] = 'par_unm' if pp is None else ('par_end' if not succ.get(pp) else 'par_held')
        srows.append(r)
    # pairs
    prows = []
    for a in ends:
        for gap in (1, 2):
            t2 = T[a] + gap
            if t2 not in stree: continue
            ns, tr = stree[t2]
            for j in tr.query_ball_point(pos[a], RMAX):
                b = ns[j]
                d = float(np.linalg.norm(pos[b] - pos[a]))
                ga, gb = p2g.get(a), p2g.get(b)
                if gap == 1:
                    tp = ga is not None and gb is not None and gb in gsucc.get(ga, [])
                else:
                    tp = ga is not None and gb is not None and any(gb in gsucc.get(c, []) for c in gsucc.get(ga, []))
                ev = (ga is not None and bool(gsucc.get(ga))) or (gb is not None and gb in gpar)
                va, vb = vel_end(a), vel_start(b)
                pr = pos[a] + gap * va if va is not None else None
                row = dict(movie=name, set=st, a=a, b=b, t=T[a], gap=gap, d=round(d, 2), fe=fedge.get((a, b), -1.0) if gap == 1 else -1.0,
                           a_in_full=int(a in fnode), b_in_full=int(b in fnode), lab='TP' if tp else ('FP' if ev else 'U'),
                           hist=back(a), fut=fwd(b), dpred=round(float(np.linalg.norm(pos[b] - pr)), 2) if pr is not None else -1,
                           dpred_b=round(float(np.linalg.norm(pos[a] - (pos[b] - gap * vb))), 2) if vb is not None else -1,
                           fdau_a=int(a in par and len(succ.get(par[a], [])) == 2))
                # competing: nearest other node (any, same frame t2) to a, and nearest other end to b in frame T[a]
                nsa, tra = atree[t2]
                dd, jj = tra.query(pos[a], k=min(3, len(nsa)))
                dd = np.atleast_1d(dd); jj = np.atleast_1d(jj)
                row['nn_any_t2'] = round(float(dd[0]), 2)
                row['b_is_nn'] = int(nsa[int(jj[0])] == b)
                prows.append(row)
    # mutual-nearest ranks among free pairs
    by_a = defaultdict(list); by_b = defaultdict(list)
    for i, r in enumerate(prows): by_a[(r['a'], r['gap'])].append(i); by_b[(r['b'], r['gap'])].append(i)
    for k, ix in by_a.items():
        ix = sorted(ix, key=lambda i: prows[i]['d'])
        for rk, i in enumerate(ix):
            prows[i]['rk_a'] = rk; prows[i]['n_a'] = len(ix)
            prows[i]['sep_a'] = round(prows[ix[1]]['d'] - prows[i]['d'], 2) if rk == 0 and len(ix) > 1 else (99. if rk == 0 else -1)
    for k, ix in by_b.items():
        ix = sorted(ix, key=lambda i: prows[i]['d'])
        for rk, i in enumerate(ix):
            prows[i]['rk_b'] = rk; prows[i]['n_b'] = len(ix)
            prows[i]['sep_b'] = round(prows[ix[1]]['d'] - prows[i]['d'], 2) if rk == 0 and len(ix) > 1 else (99. if rk == 0 else -1)
    return dict(movie=name, set=st, ends=erows, starts=srows, pairs=prows, n_nodes=len(nodes), n_edges=len(edges))


if __name__ == '__main__':
    import rule_eval as RE
    fs = [f for s, d in RE.SRC['p13'].items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(40, maxtasksperchild=4) as p: R = p.map(job, fs, chunksize=1)
    json.dump(R, open('/workspace/cl/ideas/r2_fe_rows.json', 'w'))
    print('done', len(R))
