"""READ-ONLY (critic A): is the fast-motion FN a local-flow (drift) effect? For every GT edge: length, local flow = median displacement of
P14 edges whose source lies within 25 um of the GT source (GT-free), residual = |GT displacement - local flow|, and TP / FN class
(both endpoints matched but not linked = 'struct'; endpoint unmatched = 'unm'). Summary by length and residual."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']


def job(f):
    import evalx
    from tracking_cellmot.metrics import evaluate
    from scipy.spatial import cKDTree
    K = evalx.K
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: p for p, g in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x], float) * S for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    gtt = {int(i): int(t) for i, t in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list())}
    ea = gt.edge_attrs()
    succ = defaultdict(set)
    for e in edges: succ[int(e['source_id'])].add(int(e['target_id']))
    pos = {n: np.array([v['z'], v['y'], v['x']], float) * S for n, v in nodes.items()}
    flow = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        flow[int(nodes[a]['t'])].append((pos[a], pos[b] - pos[a]))
    trees = {t: (cKDTree(np.stack([p for p, _ in L])), np.stack([d for _, d in L])) for t, L in flow.items() if L}
    rows = []
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()):
        s, d = int(s), int(d); t = gtt[s]
        disp = gpos[d] - gpos[s]; L = float(np.linalg.norm(disp))
        fl = np.zeros(3)
        if t in trees:
            ix = trees[t][0].query_ball_point(gpos[s], 25.0)
            if len(ix) >= 3: fl = np.median(trees[t][1][ix], axis=0)
        res = float(np.linalg.norm(disp - fl))
        ps, pd = g2p.get(s), g2p.get(d)
        c = 'tp' if ps is not None and pd is not None and pd in succ.get(ps, ()) else ('struct' if ps is not None and pd is not None else 'unm')
        rows.append((round(L, 2), round(float(np.linalg.norm(fl)), 2), round(res, 2), c))
    return dict(emb=name[:4], rows=rows)


if __name__ == '__main__':
    files = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p14_%s/graphs/*.json' % s))]
    with Pool(32, maxtasksperchild=4) as p: R = p.map(job, files, chunksize=1)
    for emb in ['44b6', '6bba']:
        rr = [x for r in R if r['emb'] == emb for x in r['rows']]
        print('==', emb, len(rr))
        for lo, hi in [(0, 4), (4, 6), (6, 8), (8, 10), (10, 14), (14, 99)]:
            xs = [x for x in rr if lo <= x[0] < hi]
            if not xs: continue
            n = len(xs); tp = sum(x[3] == 'tp' for x in xs); st = sum(x[3] == 'struct' for x in xs); un = sum(x[3] == 'unm' for x in xs)
            sm = [x for x in xs if x[3] == 'struct']
            print('  GT len %4.0f-%-4.0f n %6d TP %.3f struct-FN %4d unm-FN %4d | flow med %.2f | struct-FN with residual<4um: %d, <6um: %d' % (
                lo, hi, n, tp / n, st, un, float(np.median([x[1] for x in xs])), sum(x[2] < 4 for x in sm), sum(x[2] < 6 for x in sm)))
