"""Probe: dropped pre-ILP fragments (chains of dropped detections) that sit between a pred track end and a pred
track start (gap filling), vs one-sided / isolated fragments. Count unmatched-GT hits per category."""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, .40625, .40625])


def job(name):
    import evalx, edge_link
    K = evalx.K
    nodes, edges = evalx.load_graph_json(Path('/workspace/cl/ps_p5_hold36/graphs') / (name + '.json'))
    fids, fT, fV, fE, fprob = edge_link.load_full(Path('/workspace/runs/fullgraph_hold36') / (name + '.geff'))
    fP = fV * S; idx = {int(i): j for j, i in enumerate(fids.tolist())}
    drop = np.array([int(i) not in nodes for i in fids.tolist()])
    succ = defaultdict(list); par = {}
    for e in edges: succ[int(e['source_id'])].append(int(e['target_id'])); par[int(e['target_id'])] = int(e['source_id'])
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    ends = defaultdict(list); starts = defaultdict(list); allf = defaultdict(list)
    for n, v in nodes.items():
        t = int(v['t']); allf[t].append(n)
        if not succ.get(n): ends[t].append(n)
        if n not in par: starts[t].append(n)
    etree = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in ends.items() if ns}
    stree = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in starts.items() if ns}
    atree = {t: cKDTree(np.array([pos[n] for n in ns])) for t, ns in allf.items()}
    # GT unmatched nodes
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    from tracking_cellmot.metrics import evaluate
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    matched_g = {int(y) for y in na[K.MATCHED_NODE_ID].to_list() if y is not None and int(y) != -1}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    ugt = defaultdict(list)
    for i, t, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list()):
        if int(i) not in matched_g: ugt[int(t)].append(np.array([z, y, x]) * S)
    utree = {t: cKDTree(np.array(v)) for t, v in ugt.items()}
    # dropped chains via fullgraph edges among dropped nodes (each node <=1 in/out in fullgraph)
    fs = {}; fpar = {}
    for a, b in fE.tolist():
        if drop[idx[a]] and drop[idx[b]]: fs[a] = b; fpar[b] = a
    c = Counter()
    for j in np.where(drop)[0]:
        n0 = int(fids[j])
        if n0 in fpar: continue
        chain = [n0]
        while chain[-1] in fs: chain.append(fs[chain[-1]])
        t0 = int(fT[idx[chain[0]]]); t1 = int(fT[idx[chain[-1]]])
        p0 = fP[idx[chain[0]]]; p1 = fP[idx[chain[-1]]]
        dup = np.mean([atree[int(fT[idx[x]])].query(fP[idx[x]])[0] < 4.0 if int(fT[idx[x]]) in atree else False for x in chain])
        de = etree[t0 - 1][1].query(p0)[0] if (t0 - 1) in etree else 99
        ds = stree[t1 + 1][1].query(p1)[0] if (t1 + 1) in stree else 99
        cat = ('both' if de <= 10 and ds <= 10 else 'before' if de <= 10 else 'after' if ds <= 10 else 'none') + ('_dup' if dup > 0.5 else '')
        hits = sum(int(int(fT[idx[x]]) in utree and utree[int(fT[idx[x]])].query(fP[idx[x]])[0] <= 7) for x in chain)
        c[(cat, 'frags')] += 1; c[(cat, 'nodes')] += len(chain); c[(cat, 'gt_hits')] += hits; c[(cat, 'frags_with_hit')] += int(hits > 0)
    return c


if __name__ == '__main__':
    names = [l.strip() for l in open('/workspace/hold36.txt') if l.strip()]
    with Pool(36) as pool: cs = pool.map(job, names)
    tot = Counter()
    for c in cs: tot.update(c)
    cats = sorted({k[0] for k in tot})
    for cat in cats:
        print('%-12s frags %6d nodes %7d gt_hits %4d frags_with_hit %4d  hit/node %.4f' % (cat, tot[(cat, 'frags')], tot[(cat, 'nodes')], tot[(cat, 'gt_hits')], tot[(cat, 'frags_with_hit')], tot[(cat, 'gt_hits')] / max(1, tot[(cat, 'nodes')])))
