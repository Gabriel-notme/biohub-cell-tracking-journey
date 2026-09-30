"""READ-ONLY labelled census for boundary-chain reinsertion (critic A). Inserts all boundary-touching dropped pre-ILP chains
(r3_tbchain side='boundary', Lmin=2, link='none'), matches with the official matcher, and records per chain: length, touching side,
min/mean pre-ILP edge prob, distance from the inner end to the nearest kept START (start side) / END (end side) in the adjacent
frame, distance to the nearest kept node in its own frames, xy/z border distance, per-movie N_total/w, and the chain's TP / evaluable
FP edges. Also the collateral change on the original edges. Output /workspace/cl/ideas/r3_ma_chainlab_rows.json"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p56stage')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def _r(v):
    return np.array([max(0, int(round(float(x)))) for x in v], float)


def edge_labels(nodes, edges, gt, name):
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    pred, mapping = evalx.to_graph(nodes, edges)
    er = evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ea = gt.edge_attrs()
    gs = defaultdict(set); gp = defaultdict(set)
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(s)].add(int(d)); gp[int(d)].add(int(s))
    lab = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        ga_, gb = p2g.get(a), p2g.get(b)
        tp = ga_ is not None and gb is not None and gb in gs.get(ga_, ())
        ev = (ga_ is not None and len(gs.get(ga_, ())) > 0) or (gb is not None and len(gp.get(gb, ())) > 0)
        lab[(a, b)] = 'tp' if tp else ('fp' if ev else 'u')
    return lab, er


def job(f):
    try:
        import numcodecs.blosc
        numcodecs.blosc.use_threads = False
    except Exception:
        pass
    import evalx
    from edge_link import load_full
    from scipy.spatial import cKDTree
    name = Path(f).stem; st = [s for s in SETS if '_%s/' % s in f][0]
    nodes, edges = evalx.load_graph_json(f)
    gt, n_total = evalx.load_gt(name)
    fids, fT, fV, fE, fprob = load_full(FULL[st] + '/' + name + '.geff')
    fidx = {int(i): j for j, i in enumerate(fids.tolist())}
    par = {}; succ = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); par[b] = a; succ[a].append(b)
    ts = [int(v['t']) for v in nodes.values()]; T0, T1 = min(ts), max(ts)
    drop = {int(i) for i in fids.tolist() if int(i) not in nodes}
    dout = defaultdict(list); din = defaultdict(list); pr = {}
    for (a, b), p in zip(fE.tolist(), fprob.tolist()):
        a, b = int(a), int(b)
        if a in drop and b in drop and a in fidx and b in fidx and int(fT[fidx[b]]) == int(fT[fidx[a]]) + 1 and p >= 0.5:
            dout[a].append(b); din[b].append(a); pr[(a, b)] = float(p)
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    kpos = {n: _r([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    ktree = {t: (cKDTree(np.stack([kpos[n] for n in ns])), ns) for t, ns in byt.items()}
    chains = []
    for a in drop:
        if din.get(a): continue
        ch = [a]; ok = True
        while True:
            nx = dout.get(ch[-1], [])
            if not nx: break
            if len(nx) > 1 or len(din.get(nx[0], [])) > 1: ok = False; break
            ch.append(nx[0])
        if not ok or len(ch) < 2: continue
        t0_, t1_ = int(fT[fidx[ch[0]]]), int(fT[fidx[ch[-1]]])
        if not (t0_ == T0 or t1_ == T1): continue
        bad = any(int(fT[fidx[x]]) in ktree and ktree[int(fT[fidx[x]])][0].query_ball_point(_r(fV[fidx[x]]) * S, 3.5) for x in ch)
        if bad: continue
        chains.append(ch)
    lab0, er0 = edge_labels(nodes, edges, gt, name)
    nn = dict(nodes); ne = list(edges)
    for ch in chains:
        for x in ch:
            j = fidx[x]; nn[x] = {'node_id': x, 't': int(fT[j]), 'z': float(fV[j][0]), 'y': float(fV[j][1]), 'x': float(fV[j][2])}
        for a, b in zip(ch[:-1], ch[1:]): ne.append({'source_id': a, 'target_id': b})
    lab1, er1 = edge_labels(nn, ne, gt, name)
    coll = defaultdict(int)
    for k, v in lab0.items():
        if lab1[k] != v: coll[v + '>' + lab1[k]] += 1
    rows = []
    for ch in chains:
        t0_, t1_ = int(fT[fidx[ch[0]]]), int(fT[fidx[ch[-1]]])
        side = 's' if t0_ == T0 else 'e'
        inner = ch[-1] if side == 's' else ch[0]
        tn = (t1_ + 1) if side == 's' else (t0_ - 1)
        pin = _r(fV[fidx[inner]]) * S
        dse = 99.; dany = 99.
        if tn in ktree:
            tr, ns = ktree[tn]
            for q in tr.query_ball_point(pin, 15.0):
                k = ns[q]; d = float(np.linalg.norm(kpos[k] - pin)); dany = min(dany, d)
                if (side == 's' and k not in par) or (side == 'e' and not succ.get(k)): dse = min(dse, d)
        dk = []
        for x in ch:
            t = int(fT[fidx[x]])
            if t in ktree: dk.append(float(ktree[t][0].query(_r(fV[fidx[x]]) * S, k=1)[0]))
        p = [pr[(a, b)] for a, b in zip(ch[:-1], ch[1:])]
        c0 = _r(fV[fidx[ch[0]]])
        bxy = float(min(c0[1], c0[2], 255 - c0[1], 255 - c0[2]) * S[1]); bz = float(min(c0[0], 63 - c0[0]) * S[0])
        labs = [lab1[(a, b)] for a, b in zip(ch[:-1], ch[1:])]
        rows.append(dict(L=len(ch), side=side, pmin=round(min(p), 3), pmean=round(float(np.mean(p)), 3), dse=round(dse, 2), dany=round(dany, 2),
                         dkmin=round(min(dk), 2) if dk else 99., bxy=round(bxy, 1), bz=round(bz, 1), tp=labs.count('tp'), fp=labs.count('fp')))
    return dict(movie=name, set=st, emb=name[:4], n_total=n_total, w0=er0.edge_tp + er0.edge_fp + er0.edge_fn, J0=er0.edge_tp / max(1, er0.edge_tp + er0.edge_fp + er0.edge_fn),
                npred0=len(nodes), coll=dict(coll), rows=rows,
                d=dict(tp=er1.edge_tp - er0.edge_tp, fp=er1.edge_fp - er0.edge_fp, fn=er1.edge_fn - er0.edge_fn, nodes=len(nn) - len(nodes)))


if __name__ == '__main__':
    files = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p14_%s/graphs/*.json' % s))]
    with Pool(40, maxtasksperchild=4) as p: R = p.map(job, files, chunksize=1)
    json.dump(R, open('/workspace/cl/ideas/r3_ma_chainlab_rows.json', 'w'))
    print('done', len(R))
