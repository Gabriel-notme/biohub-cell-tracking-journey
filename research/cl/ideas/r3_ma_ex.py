"""READ-ONLY: print concrete examples of unmatched GT nodes at frame 0/1 in a few hold36 6bba movies: GT track t=0..5, the pred nodes
near each GT node (kept, with matched flag), and dropped pre-ILP detections within 9 um."""
import os, sys, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p56stage')
from collections import defaultdict
import numpy as np
import evalx
from edge_link import load_full
from tracking_cellmot.metrics import evaluate
from scipy.spatial import cKDTree
K = evalx.K; S = np.array([1.625, 0.40625, 0.40625])
shown = 0
for f in sorted(glob.glob('/workspace/cl/ps_p14_hold36/graphs/6bba*.json'))[:12] + sorted(glob.glob('/workspace/cl/ps_p14_hold36/graphs/44b6*.json'))[:6]:
    name = f.split('/')[-1][:-5]
    nodes, edges = evalx.load_graph_json(f); gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: p for p, g in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x], float) for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    gtt = {int(i): int(t) for i, t in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list())}
    ea = gt.edge_attrs(); gsucc = defaultdict(list)
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(s)].append(int(d))
    par = {}; succ = defaultdict(list)
    for e in edges: par[int(e['target_id'])] = int(e['source_id']); succ[int(e['source_id'])].append(int(e['target_id']))
    ppos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx'], float) for n, v in nodes.items()}
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    fids, fT, fV, fE, fprob = load_full('/workspace/runs/fullgraph_hold36/%s.geff' % name)
    fpar = {int(b): (int(a), float(p)) for (a, b), p in zip(fE.tolist(), fprob.tolist())}
    for g in gpos:
        if gtt[g] != 0 or g in g2p: continue
        print('\n###', name, 'GT', g)
        cur = g
        for step in range(6):
            t = gtt[cur]; gp = gpos[cur]
            near = sorted([(float(np.linalg.norm((ppos[n] - gp) * S)), n) for n in byt[t]])[:2]
            fd = [(float(np.linalg.norm((np.round(fV[j]) - gp) * S)), int(fids[j])) for j in np.where(fT == t)[0]]
            fd = sorted([x for x in fd if x[0] < 9])[:3]
            desc = ' | '.join('pred %d d=%.1f m=%s par=%s' % (n, d, 'Y' if n in p2g else '-', par.get(n, 'START')) for d, n in near)
            fdesc = ' ; '.join('%s%d d=%.1f fpar=%s' % ('K' if i in nodes else 'D', i, d, fpar.get(i, '-')) for d, i in fd)
            print('  t=%d GT %s matched=%s || %s || preILP: %s' % (t, np.round(gp).astype(int), g2p.get(cur, '-'), desc, fdesc))
            nx = gsucc.get(cur, [])
            if len(nx) != 1: break
            cur = nx[0]
        shown += 1
        if shown >= 14: break
    if shown >= 14: break
