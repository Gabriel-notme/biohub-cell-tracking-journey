"""Probe: unmatched GT nodes whose GT parent and child ARE matched by consecutive pred track nodes p -> n -> c.
How far is the pred middle node n from the GT node, and is a dropped detection / the p-c midpoint closer?"""
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
    fP = fV * S; drop = np.array([int(i) not in nodes for i in fids.tolist()])
    dtree = {}
    for t in np.unique(fT):
        ix = np.where((fT == t) & drop)[0]
        if len(ix): dtree[int(t)] = (ix, cKDTree(fP[ix]))
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    from tracking_cellmot.metrics import evaluate
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    g2p = {g: p for p, g in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x]) * S for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    ea = gt.edge_attrs(); gsucc = defaultdict(list); gpar = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(x)].append(int(y)); gpar[int(y)] = int(x)
    succ = defaultdict(list); par = {}
    for e in edges: succ[int(e['source_id'])].append(int(e['target_id'])); par[int(e['target_id'])] = int(e['source_id'])
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    c = Counter(); recs = []
    for g in gpos:
        if g in g2p or g not in gpar or not gsucc.get(g): continue
        gp, gc = gpar[g], gsucc[g][0]
        p, cc = g2p.get(gp), g2p.get(gc)
        if p is None or cc is None: c['nbr_unmatched'] += 1; continue
        kids = succ.get(p, [])
        mids = [n for n in kids if cc in succ.get(n, [])]
        if not mids: c['no_pred_path'] += 1; continue
        n = mids[0]; mid = (pos[p] + pos[cc]) / 2
        dn = float(np.linalg.norm(pos[n] - gpos[g])); dm = float(np.linalg.norm(mid - gpos[g]))
        t = int(nodes[n]['t']); dd = 99.
        if t in dtree:
            ix, tr = dtree[t]; q, j = tr.query(mid); dd_g = float(np.linalg.norm(fP[ix[j]] - gpos[g])); dd = float(q)
        else: dd_g = 99.
        c['path_exists'] += 1
        c['n_within7_but_unmatched' if dn <= 7 else 'n_far'] += 1
        c['mid_within7' if dm <= 7 else 'mid_far'] += 1
        c['drop_near_mid_within7_of_g' if (dd <= 4 and dd_g <= 7) else 'no_good_drop'] += 1
        recs.append((round(dn, 1), round(dm, 1), round(dd, 1), round(dd_g, 1), round(float(np.linalg.norm(pos[n] - mid)), 1)))
    return c, recs


if __name__ == '__main__':
    names = [l.strip() for l in open('/workspace/hold36.txt') if l.strip()]
    with Pool(36) as pool: res = pool.map(job, names)
    tot = Counter(); R = []
    for c, r in res: tot.update(c); R += r
    for k, v in sorted(tot.items()): print(k, v)
    print('(dist n->g, dist mid->g, dist drop->mid, dist drop->g, dev n->mid) samples:', R[:40])
