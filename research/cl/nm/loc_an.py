"""Anatomy of the localisation-fixable nodes on P15: unmatched pred nodes whose track neighbours say they belong to GT node g
(d(n,g) <= 12 um) = 'U', versus controls: matched nodes with residual 5-7 um ('M57') and a 3% random sample of matched nodes ('M').
Features: displacement vector, spike (pred vs pred-neighbour midpoint), GT spike, GT speed, distance to GT division, raw detection
(fullgraph, same id) distance to GT, nearest dropped detection to GT, other pred nodes near GT, image intensity at pred / GT / midpoint.
usage: loc_an.py <out.json>"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/p56stage')
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def rpos(v):
    return np.array([max(0, int(round(v[k]))) for k in 'zyx'], float) * S


def job(args):
    s, f = args
    import numcodecs.blosc; numcodecs.blosc.use_threads = False
    import zarr, evalx, edge_link
    from tracking_cellmot.metrics import evaluate
    from scipy.spatial import cKDTree
    K = evalx.K
    name = Path(f).stem
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    nodes, edges = evalx.load_graph_json(f)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: p for p, g in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x], float) * S for i, z, y, x in zip(*[ga[k].to_list() for k in [K.NODE_ID, 'z', 'y', 'x']])}
    gt_t = {int(i): int(t) for i, t in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list())}
    ea = gt.edge_attrs(); GE = [(int(a), int(b)) for a, b in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list())]
    gsucc = defaultdict(list); gpar = {}
    for a, b in GE: gsucc[a].append(b); gpar[b] = a
    succ = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a
    pos = {n: rpos(v) for n, v in nodes.items()}
    fpos = {n: np.array([v[k] for k in 'zyx'], float) * S for n, v in nodes.items()}
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    trees = {t: (cKDTree(np.stack([pos[n] for n in ns])), ns) for t, ns in byt.items()}
    gbyt = defaultdict(list)
    for g, t in gt_t.items(): gbyt[t].append(g)
    gtrees = {t: (cKDTree(np.stack([gpos[g] for g in gs])), gs) for t, gs in gbyt.items()}
    fids, fT, fV, fE, fprob = edge_link.load_full(FULL[s] + '/' + name + '.geff')
    fP = fV * S; fidx = {int(i): j for j, i in enumerate(fids.tolist())}
    drop = np.array([int(i) not in nodes for i in fids.tolist()])
    dtrees = {}
    for t in np.unique(fT):
        ix = np.where((fT == t) & drop)[0]
        if len(ix): dtrees[int(t)] = (ix, cKDTree(fP[ix]))

    def gwalk_fwd(g, chain):
        for q in chain:
            ch = gsucc.get(g, [])
            if not ch: return None
            g = ch[0] if len(ch) == 1 else min(ch, key=lambda c: np.linalg.norm(gpos[c] - pos[q]))
        return g

    def gwalk_back(g, k):
        for _ in range(k):
            g = gpar.get(g)
            if g is None: return None
        return g
    votes = {}
    for n in nodes:
        v = Counter()
        if n in p2g: v[p2g[n]] += 1.0
        c = n; back = []
        for k in range(1, 4):
            p = par.get(c)
            if p is None: break
            back.append(p); c = p
            if p in p2g:
                g = gwalk_fwd(p2g[p], list(reversed(back[:-1])) + [n])
                if g is not None: v[g] += 1.0 / k
        c = n
        for k in range(1, 4):
            ch = succ.get(c, [])
            if len(ch) != 1: break
            c = ch[0]
            if c in p2g:
                g = gwalk_back(p2g[c], k)
                if g is not None: v[g] += 1.0 / k
        if v:
            g, w = max(v.items(), key=lambda kv: (kv[1], -np.linalg.norm(gpos[kv[0]] - pos[n])))
            if w >= 1.0 and gt_t.get(g) == int(nodes[n]['t']): votes[n] = (g, w, float(np.linalg.norm(gpos[g] - pos[n])))
    claim = {}
    for n, (g, w, d) in votes.items():
        if g not in claim or (w, -d) > (votes[claim[g]][1], -votes[claim[g]][2]): claim[g] = n
    intended = {n: votes[n] for n in claim.values()}

    def gdiv_dist(g):  # frames to nearest GT division on this lineage (back through parents, forward along unique successors)
        best = 99; c = g; k = 0
        while c is not None and k < 10:
            if len(gsucc.get(c, [])) == 2: best = min(best, k); break
            c = gpar.get(c); k += 1
        c = g; k = 0
        while c is not None and k < 10:
            ch = gsucc.get(c, [])
            if len(ch) == 2: best = min(best, k); break
            c = ch[0] if len(ch) == 1 else None; k += 1
        return best

    sel = []
    for n, (g, w, d) in intended.items():
        m = p2g.get(n)
        if m is None and d <= 12: sel.append(('U', n, g))
        elif m == g and d > 5: sel.append(('M57', n, g))
        elif m == g and rng.random() < 0.03: sel.append(('M', n, g))
    img = zarr.open_group('/workspace/data/train/%s.zarr' % name, mode='r')['0']
    cache = {}

    def frame(t):
        if t not in cache:
            if len(cache) > 6: cache.pop(next(iter(cache)))
            cache[t] = np.asarray(img[t]).astype(np.float32)
        return cache[t]

    def inten(t, p_um):
        a = frame(t); q = np.round(p_um / S).astype(int)
        z0, y0, x0 = [int(np.clip(v, 0, sh - 1)) for v, sh in zip(q, a.shape)]
        box = a[max(0, z0 - 1):z0 + 2, max(0, y0 - 3):y0 + 4, max(0, x0 - 3):x0 + 4]
        return float(box.mean())
    recs = []
    for typ, n, g in sorted(sel, key=lambda r: nodes[r[1]]['t']):
        t = int(nodes[n]['t']); P = pos[n]; G = gpos[g]; dv = G - P
        p, cs = par.get(n), succ.get(n, [])
        mid = (pos[p] + pos[cs[0]]) / 2 if (p is not None and len(cs) == 1) else None
        gp, gc = gpar.get(g), gsucc.get(g, [])
        gmid = (gpos[gp] + gpos[gc[0]]) / 2 if (gp is not None and len(gc) == 1) else None
        raw = fP[fidx[n]] if (n in fidx and int(fT[fidx[n]]) == t) else None
        dd = None
        if t in dtrees:
            ix, tr = dtrees[t]; q, j = tr.query(G); dd = float(q)
        tr, ns = trees[t]; kk = min(4, len(ns)); qd, qj = tr.query(G, k=kk); qd = np.atleast_1d(qd); qj = np.atleast_1d(qj)
        others = [(float(a), ns[int(b)]) for a, b in zip(qd, qj) if ns[int(b)] != n]
        o_near = others[0] if others else (99., None)
        gtr, gs = gtrees[t]; gq, gj = gtr.query(P, k=min(2, len(gs))); gq = np.atleast_1d(gq); gj = np.atleast_1d(gj)
        g_other = [float(a) for a, b in zip(gq, gj) if gs[int(b)] != g]
        I_p = inten(t, P); I_g = inten(t, G); I_m = inten(t, mid) if mid is not None else None
        recs.append(dict(typ=typ, m=name, set=s, t=t, d=float(np.linalg.norm(dv)), dz=float(dv[0]), dy=float(dv[1]), dx=float(dv[2]),
                         spike=float(np.linalg.norm(P - mid)) if mid is not None else None, mid_g=float(np.linalg.norm(mid - G)) if mid is not None else None,
                         gspike=float(np.linalg.norm(G - gmid)) if gmid is not None else None,
                         gspeed=float(np.linalg.norm(G - gpos[gp])) if gp is not None else None,
                         gdiv=gdiv_dist(g), raw_g=float(np.linalg.norm(raw - G)) if raw is not None else None,
                         raw_p=float(np.linalg.norm(raw - fpos[n])) if raw is not None else None, drop_g=dd,
                         other_d=o_near[0], other_matched=(o_near[1] in p2g) if o_near[1] is not None else None,
                         other_is_g=(p2g.get(o_near[1]) == g) if o_near[1] is not None else None, g_matched=g in g2p,
                         g_other=g_other[0] if g_other else 99., I_p=I_p, I_g=I_g, I_m=I_m, fork=len(cs) == 2, has_par=p is not None, n_ch=len(cs),
                         P=P.tolist(), G=G.tolist()))
    return recs


if __name__ == '__main__':
    fs = [(s, f) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s))]
    with Pool(24, maxtasksperchild=4) as p: out = p.map(job, fs, chunksize=1)
    R = [r for rs in out for r in rs]
    json.dump(R, open(sys.argv[1], 'w'))
    print('n', Counter(r['typ'] for r in R))
