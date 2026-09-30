"""Round-2 diagnostic (reads GT): RE-PARENTING configurations in P14.
END side: track END a at t (has parent, no child, not fork daughter) and a same-frame node q (<= 12 um) with exactly one child pc:
  label the swap q->pc  =>  a->pc (edge-level delta with fixed node matching).
START side: track START b at t+1 (no parent, has child) and a same-frame node c (<= 12 um) whose parent p has exactly one child:
  label the swap p->c  =>  p->b.
Writes /workspace/cl/ideas/r2_rp_rows.json."""
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
R = 12.0


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
        eflag[(a, b)] = ','.join(k for k in e if k not in ('source_id', 'target_id', 'edge_prob'))
    pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) * S for n, v in nodes.items()}
    T = {n: int(v['t']) for n, v in nodes.items()}
    tmax = max(T.values())

    def lab(x, y):
        gx, gy = p2g.get(x), p2g.get(y)
        if gx is not None and gy is not None and gy in gsucc.get(gx, []): return 1, 0
        ev = (gx is not None and bool(gsucc.get(gx))) or (gy is not None and gy in gpar)
        return 0, int(ev)

    def back(n, k=50):
        c = 0
        while n in par and c < k: n = par[n]; c += 1
        return c

    def fwd(n, k=50):
        c = 0
        while len(succ.get(n, [])) == 1 and c < k: n = succ[n][0]; c += 1
        return c

    def nm(n):
        g = p2g.get(n)
        return 'unm' if g is None else 'm'

    by = defaultdict(list)
    for n in nodes: by[T[n]].append(n)
    tree = {t: (ns, cKDTree(np.stack([pos[n] for n in ns]))) for t, ns in by.items()}
    rows = []
    # END side
    for a in nodes:
        if succ.get(a) or a not in par or T[a] >= tmax: continue
        if len(succ.get(par[a], [])) == 2: continue
        ns, tr = tree[T[a]]
        for j in tr.query_ball_point(pos[a], R):
            q = ns[j]
            if q == a or len(succ.get(q, [])) != 1: continue
            if q in par and len(succ.get(par[q], [])) == 2: fq = 1
            else: fq = 0
            pc = succ[q][0]
            if len(succ.get(pc, [])) == 2: pass
            tp_new, fp_new = lab(a, pc); tp_old, fp_old = lab(q, pc)
            va = pos[a] - pos[par[a]]; vq = pos[q] - pos[par[q]] if q in par else None
            rows.append(dict(side='end', movie=name, set=st, t=T[a], d_aq=round(float(np.linalg.norm(pos[a] - pos[q])), 2),
                             d_apc=round(float(np.linalg.norm(pos[a] - pos[pc])), 2), d_qpc=round(float(np.linalg.norm(pos[q] - pos[pc])), 2),
                             h_a=back(a), h_q=back(q), f_pc=fwd(pc), q_start=int(q not in par), q_fdau=fq, pc_fork=int(len(succ.get(pc, [])) == 2),
                             fe_new=fedge.get((a, pc), -1.0), fe_old=fedge.get((q, pc), -1.0),
                             dpa=round(float(np.linalg.norm(pos[a] + va - pos[pc])), 2),
                             dpq=round(float(np.linalg.norm(pos[q] + vq - pos[pc])), 2) if vq is not None else -1,
                             flag_old=eflag.get((q, pc), ''), m_a=nm(a), m_q=nm(q), m_pc=nm(pc),
                             dtp=tp_new - tp_old, dfp=fp_new - fp_old))
    # START side
    for b in nodes:
        if b in par or not succ.get(b) or T[b] == 0: continue
        ns, tr = tree[T[b]]
        for j in tr.query_ball_point(pos[b], R):
            c = ns[j]
            if c == b or c not in par: continue
            p = par[c]
            if len(succ.get(p, [])) != 1: continue
            tp_new, fp_new = lab(p, b); tp_old, fp_old = lab(p, c)
            vp = pos[p] - pos[par[p]] if p in par else None
            rows.append(dict(side='start', movie=name, set=st, t=T[b], d_bc=round(float(np.linalg.norm(pos[b] - pos[c])), 2),
                             d_pb=round(float(np.linalg.norm(pos[p] - pos[b])), 2), d_pc=round(float(np.linalg.norm(pos[p] - pos[c])), 2),
                             f_b=fwd(b), f_c=fwd(c), h_p=back(p), b_len1=int(len(succ.get(b, [])) == 0),
                             fe_new=fedge.get((p, b), -1.0), fe_old=fedge.get((p, c), -1.0),
                             dpb=round(float(np.linalg.norm(pos[p] + vp - pos[b])), 2) if vp is not None else -1,
                             dpc=round(float(np.linalg.norm(pos[p] + vp - pos[c])), 2) if vp is not None else -1,
                             flag_old=eflag.get((p, c), ''), m_b=nm(b), m_c=nm(c), m_p=nm(p),
                             dtp=tp_new - tp_old, dfp=fp_new - fp_old))
    return rows


if __name__ == '__main__':
    import rule_eval as RE
    fs = [f for s, d in RE.SRC['p13'].items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(40, maxtasksperchild=4) as p: R_ = p.map(job, fs, chunksize=1)
    rows = [r for rs in R_ for r in rs]
    json.dump(rows, open('/workspace/cl/ideas/r2_rp_rows.json', 'w'))
    print('done', len(rows))
