"""Where are the GT nodes that the P5 graph misses? Are they among dropped pre-ILP detections, and how do the
dropped fragments containing them look compared with junk fragments?"""
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
    fP = fV * S
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    from tracking_cellmot.metrics import evaluate
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    matched_g = {int(y) for y in na[K.MATCHED_NODE_ID].to_list() if y is not None and int(y) != -1}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gid = np.array(ga[K.NODE_ID].to_list()); gt_t = np.array(ga['t'].to_list()).astype(int)
    gP = np.stack([np.array(ga[k].to_list()) for k in 'zyx'], 1) * S
    # pred positions per frame
    pos = defaultdict(list)
    for n, v in nodes.items(): pos[int(v['t'])].append(np.array([v['z'], v['y'], v['x']]) * S)
    ptree = {t: cKDTree(np.array(p)) for t, p in pos.items()}
    drop = np.array([int(i) not in nodes for i in fids.tolist()])
    c = Counter()
    # unmatched GT nodes: nearest dropped fullgraph node and nearest pred node
    dtree = {}
    for t in np.unique(fT):
        ix = np.where((fT == t) & drop)[0]
        if len(ix): dtree[int(t)] = (ix, cKDTree(fP[ix]))
    for g, t, p in zip(gid.tolist(), gt_t.tolist(), gP):
        if g in matched_g: continue
        c['gt_unmatched'] += 1
        dp = ptree[t].query(p)[0] if t in ptree else 99
        dd = dtree[t][1].query(p)[0] if t in dtree else 99
        c['dropped<=7' if dd <= 7 else 'no_dropped<=7'] += 1
        if dd <= 7: c['dropped<=4'] += int(dd <= 4)
        c['pred_within_7_to_14' if 7 < dp <= 14 else ('pred_within_7' if dp <= 7 else 'pred_far')] += 1
    # dropped fragments = connected components among dropped nodes via fullgraph edges
    idx = {int(i): j for j, i in enumerate(fids.tolist())}
    adj = defaultdict(list)
    for a, b in fE.tolist():
        if drop[idx[a]] and drop[idx[b]]: adj[a].append(b); adj[b].append(a)
    seen = set(); nfr = Counter()
    for j in np.where(drop)[0]:
        n0 = int(fids[j])
        if n0 in seen: continue
        comp = []; st = [n0]; seen.add(n0)
        while st:
            x = st.pop(); comp.append(x)
            for y in adj[x]:
                if y not in seen: seen.add(y); st.append(y)
        L = len(comp)
        nfr['frag_len_%s' % ('1' if L == 1 else '2-5' if L <= 5 else '6-20' if L <= 20 else '>20')] += 1
        nfr['frag_nodes'] += L
    c.update(nfr)
    c['dropped_nodes'] = int(drop.sum()); c['pred_nodes'] = len(nodes)
    return c


if __name__ == '__main__':
    names = [l.strip() for l in open('/workspace/hold36.txt') if l.strip()]
    with Pool(36) as pool: cs = pool.map(job, names)
    tot = Counter()
    for c in cs: tot.update(c)
    for k, v in sorted(tot.items()): print(k, v)
