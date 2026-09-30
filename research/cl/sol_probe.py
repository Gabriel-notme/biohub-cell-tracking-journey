"""ILP-selected (solution=True) nodes that are absent from the P5 graph (removed by base short-track filtering):
count, chain lengths, and hit rate against GT nodes the P5 graph misses. usage: sol_probe.py <p5_graph_dir> <run_working_dir> <list>"""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np, zarr
from scipy.spatial import cKDTree
S = np.array([1.625, .40625, .40625])
G, RUNW, LST = sys.argv[1], sys.argv[2], sys.argv[3]


def job(name):
    import evalx
    K = evalx.K
    nodes, edges = evalx.load_graph_json(Path(G) / (name + '.json'))
    g = zarr.open_group(RUNW + '/tracking_repo/predictions/unknown/unet_transformer/split_0/%s.geff' % name, mode='r')
    ids = np.asarray(g['nodes/ids'][:]).astype(np.int64); sol = np.asarray(g['nodes/props/solution/values'][:]).astype(bool)
    P = np.stack([np.asarray(g['nodes/props/%s/values' % k][:]) for k in 'zyx'], 1); T = np.asarray(g['nodes/props/t/values'][:]).astype(int)
    E = np.asarray(g['edges/ids'][:]).astype(np.int64); esol = np.asarray(g['edges/props/solution/values'][:]).astype(bool)
    idx = {int(i): j for j, i in enumerate(ids.tolist())}
    miss = np.array([s and int(i) not in nodes for i, s in zip(ids.tolist(), sol.tolist())])
    fs = {}; fpar = {}
    for (a, b), s_ in zip(E.tolist(), esol.tolist()):
        if s_ and miss[idx[a]] and miss[idx[b]]: fs[a] = b; fpar[b] = a
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    from tracking_cellmot.metrics import evaluate
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    matched_g = {int(y) for y in na[K.MATCHED_NODE_ID].to_list() if y is not None and int(y) != -1}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    ug = defaultdict(list)
    for i, t, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list()):
        if int(i) not in matched_g: ug[int(t)].append(np.array([z, y, x]) * S)
    ut = {t: cKDTree(np.array(v)) for t, v in ug.items()}
    c = Counter(); c['sol_nodes'] = int(sol.sum()); c['sol_missing'] = int(miss.sum()); c['pred_nodes'] = len(nodes)
    for j in np.where(miss)[0]:
        n0 = int(ids[j])
        if n0 in fpar: continue
        ch = [n0]
        while ch[-1] in fs: ch.append(fs[ch[-1]])
        L = len(ch); lb = '1' if L == 1 else '2' if L == 2 else '3-5' if L <= 5 else '6+'
        hits = sum(int(T[idx[x]] in ut and ut[T[idx[x]]].query(P[idx[x]] * S)[0] <= 7) for x in ch)
        c[(lb, 'chains')] += 1; c[(lb, 'nodes')] += L; c[(lb, 'hits')] += hits
    return c


if __name__ == '__main__':
    names = [l.strip() for l in open(LST) if l.strip()]
    with Pool(36) as pool: cs = pool.map(job, names)
    tot = Counter()
    for c in cs: tot.update(c)
    print('ILP-selected nodes', tot['sol_nodes'], 'missing from P5 graph', tot['sol_missing'], 'P5 nodes', tot['pred_nodes'])
    for lb in ['1', '2', '3-5', '6+']:
        n = tot[(lb, 'nodes')]
        if n: print('chain len %-4s chains %6d nodes %6d hits %4d hit/node %.4f' % (lb, tot[(lb, 'chains')], n, tot[(lb, 'hits')], tot[(lb, 'hits')] / n))
