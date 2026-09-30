"""READ-ONLY census (critic A): boundary seeds = P14 track STARTs (>=5 nodes) at t in 1..3 or near a spatial border, and track ENDs
(>=5 nodes) at t in T-3..T-1 or near a spatial border. For each seed, GT label of the one-frame extension:
  ev   = seed matched to a GT node that has a GT parent (start side) / GT child (end side)  -> an added edge is evaluable
  rec  = that GT neighbour is currently UNMATCHED (so a correct added node would match it and gain an edge)
For the adjacent frame, dropped pre-ILP detections: nearest to seed (dist), nearest to velocity-extrapolated position,
whether any has a fullgraph edge to the seed, and whether the chosen candidate lies within 7 um of the GT neighbour.
Output /workspace/cl/ideas/r3_ma_seed_rows.json"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p56stage')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625]); SHAPE = np.array([64, 256, 256])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def rp(v):
    return np.array([max(0, int(round(float(x)))) for x in v], float)


def job(f):
    try:
        import numcodecs.blosc
        numcodecs.blosc.use_threads = False
    except Exception:
        pass
    import evalx
    from edge_link import load_full
    from tracking_cellmot.metrics import evaluate
    from scipy.spatial import cKDTree
    K = evalx.K
    name = Path(f).stem; st = [s for s in SETS if '_%s/' % s in f][0]
    nodes, edges = evalx.load_graph_json(f)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: p for p, g in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x], float) for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    ea = gt.edge_attrs()
    gsucc = defaultdict(list); gpar = defaultdict(list)
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(s)].append(int(d)); gpar[int(d)].append(int(s))
    par = {}; succ = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); par[b] = a; succ[a].append(b)
    pos = {n: rp([v['z'], v['y'], v['x']]) for n, v in nodes.items()}
    ts = [int(v['t']) for v in nodes.values()]; T0, T1 = min(ts), max(ts)
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    ktree = {t: cKDTree(np.stack([pos[n] for n in ns]) * S) for t, ns in byt.items()}
    fids, fT, fV, fE, fprob = load_full(FULL[st] + '/' + name + '.geff')
    fidx = {int(i): j for j, i in enumerate(fids.tolist())}
    fpar = defaultdict(list); fch = defaultdict(list)
    for (a, b), p in zip(fE.tolist(), fprob.tolist()): fpar[int(b)].append((float(p), int(a))); fch[int(a)].append((float(p), int(b)))
    dix = defaultdict(list)
    for j, (i, t) in enumerate(zip(fids.tolist(), fT.tolist())):
        if int(i) not in nodes: dix[int(t)].append(j)
    dtree = {t: (ix, cKDTree(np.stack([rp(fV[j]) for j in ix]) * S)) for t, ix in dix.items() if ix}

    def clen(n, down):
        L = 1; c = n
        while L < 5:
            nx = succ.get(c, []) if down else ([par[c]] if c in par else [])
            if len(nx) != 1: break
            c = nx[0]; L += 1
        return L, c

    def border(p):  # um to nearest xy border, and z border
        xy = min(p[1], p[2], SHAPE[1] - 1 - p[1], SHAPE[2] - 1 - p[2]) * S[1]
        z = min(p[0], SHAPE[0] - 1 - p[0]) * S[0]
        return float(xy), float(z)
    rows = []
    for n in nodes:
        t = int(nodes[n]['t'])
        for backward in (True, False):
            if backward and (n in par or t == T0): continue
            if (not backward) and (succ.get(n) or t == T1): continue
            L, far = clen(n, backward is True)
            if L < 5: continue
            bxy, bz = border(pos[n])
            tb = (t - T0) if backward else (T1 - t)
            if not (tb <= 3 or bxy <= 4 or bz <= 3.25): continue
            tn = t - 1 if backward else t + 1
            # GT label
            g = p2g.get(n); ev = 0; rec = 0; gnb = None
            if g is not None:
                nb = gpar.get(g, []) if backward else gsucc.get(g, [])
                if nb:
                    ev = 1; gnb = nb[0]
                    if gnb not in g2p: rec = 1
            # candidates
            vel = pos[n] - pos[far] if far != n else np.zeros(3)
            vstep = vel / max(1, L - 1)
            pred_pos = pos[n] + vstep  # extrapolated position one frame beyond the seed (pixels); vstep points away from the track
            c_near = c_vel = c_edge = None; d_near = d_vel = 99.
            if tn in dtree:
                ix, tr = dtree[tn]
                d, k = tr.query(pos[n] * S, k=1); c_near = ix[int(k)]; d_near = float(d)
                d, k = tr.query(pred_pos * S, k=1); c_vel = ix[int(k)]; d_vel = float(d)
            fe = fpar.get(n, []) if backward else fch.get(n, [])
            fe = sorted([(p, x) for p, x in fe if x in fidx and x not in nodes and int(fT[fidx[x]]) == tn], reverse=True)
            if fe: c_edge = fidx[fe[0][1]]
            # does a kept node already sit near the extrapolated spot?
            dk = float(ktree[tn].query(pred_pos * S, k=1)[0]) if tn in ktree else 99.

            def ok(c):
                if c is None or gnb is None: return -1
                return int(float(np.linalg.norm((rp(fV[c]) - gpos[gnb]) * S)) <= 7.0)
            rows.append(dict(side='s' if backward else 'e', tb=tb, bxy=round(bxy, 1), bz=round(bz, 2), ev=ev, rec=rec,
                             d_near=round(d_near, 2), d_vel=round(d_vel, 2), has_edge=int(c_edge is not None), p_edge=round(fe[0][0], 3) if fe else -1,
                             ok_near=ok(c_near), ok_vel=ok(c_vel), ok_edge=ok(c_edge), same_nv=int(c_near == c_vel), dk=round(dk, 2),
                             step=round(float(np.linalg.norm(vstep * S)), 2)))
    return dict(movie=name, set=st, emb=name[:4], rows=rows)


if __name__ == '__main__':
    files = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p14_%s/graphs/*.json' % s))]
    with Pool(40, maxtasksperchild=4) as p: R = p.map(job, files, chunksize=1)
    json.dump(R, open('/workspace/cl/ideas/r3_ma_seed_rows.json', 'w'))
    print('done', len(R))
