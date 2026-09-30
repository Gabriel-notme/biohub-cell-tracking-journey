"""GT hit rate of dropped pre-ILP chains by chain length and duplicate status (hold36, P5 graphs)."""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, .40625, .40625])
G = sys.argv[1]; F = sys.argv[2]; LST = sys.argv[3]


def job(name):
    import evalx, edge_link
    K = evalx.K
    nodes, edges = evalx.load_graph_json(Path(G) / (name + '.json'))
    fids, fT, fV, fE, fprob = edge_link.load_full(Path(F) / (name + '.geff'))
    fP = fV * S; idx = {int(i): j for j, i in enumerate(fids.tolist())}
    drop = np.array([int(i) not in nodes for i in fids.tolist()])
    allf = defaultdict(list)
    for n, v in nodes.items(): allf[int(v['t'])].append(np.array([v['z'], v['y'], v['x']]) * S)
    atree = {t: cKDTree(np.array(p)) for t, p in allf.items()}
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
    utree = {t: cKDTree(np.array(v)) for t, v in ug.items()}
    fs = {}; fpar = {}
    for a, b in fE.tolist():
        if drop[idx[a]] and drop[idx[b]]: fs[a] = b; fpar[b] = a
    c = Counter()
    for j in np.where(drop)[0]:
        n0 = int(fids[j])
        if n0 in fpar: continue
        ch = [n0]
        while ch[-1] in fs: ch.append(fs[ch[-1]])
        L = len(ch)
        dmin = [atree[int(fT[idx[x]])].query(fP[idx[x]])[0] if int(fT[idx[x]]) in atree else 99 for x in ch]
        dup = np.mean(np.array(dmin) < 4.0)
        hits = sum(int(int(fT[idx[x]]) in utree and utree[int(fT[idx[x]])].query(fP[idx[x]])[0] <= 7) for x in ch)
        lb = '1' if L == 1 else '2' if L == 2 else '3-5' if L <= 5 else '6-10' if L <= 10 else '11+'
        db = 'dup' if dup >= 0.5 else ('near' if min(dmin) < 6 else 'clear')
        c[(lb, db, 'chains')] += 1; c[(lb, db, 'nodes')] += L; c[(lb, db, 'hits')] += hits
    return c


if __name__ == '__main__':
    names = [l.strip() for l in open(LST) if l.strip()]
    with Pool(36) as pool: cs = pool.map(job, names)
    tot = Counter()
    for c in cs: tot.update(c)
    for lb in ['1', '2', '3-5', '6-10', '11+']:
        for db in ['clear', 'near', 'dup']:
            n = tot[(lb, db, 'nodes')]
            if n: print('len %-5s %-5s chains %6d nodes %7d hits %4d  hit/node %.4f' % (lb, db, tot[(lb, db, 'chains')], n, tot[(lb, db, 'hits')], tot[(lb, db, 'hits')] / n))
