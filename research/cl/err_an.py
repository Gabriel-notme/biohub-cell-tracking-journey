"""Edge error anatomy of a graph directory (official validity): FN categories and FP categories, with extra detail on
unmatched endpoints: distance from the unmatched pred node to the nearest GT node, and whether that GT node is matched elsewhere."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, 0.40625, 0.40625])


def job(f):
    name = Path(f).stem
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    g2p = {v: k for k, v in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gid = [int(x) for x in ga[K.NODE_ID].to_list()]; gt_t = np.array(ga['t'].to_list()); gxyz = np.stack([ga[k].to_numpy() for k in 'zyx'], 1) * S
    gidx = {g: i for i, g in enumerate(gid)}
    ea = gt.edge_attrs(); src = [int(x) for x in ea[K.EDGE_SOURCE].to_list()]; dst = [int(x) for x in ea[K.EDGE_TARGET].to_list()]
    ge = set(zip(src, dst)); gout = set(src); gin = set(dst)
    ch = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); ch[a].append(b); par[b] = a
    # pred nodes per frame kd-trees
    fr = defaultdict(list)
    for n, v in nodes.items(): fr[int(v['t'])].append(n)
    trees = {t: (cKDTree(np.array([[nodes[n][k] for k in 'zyx'] for n in ns]) * S), ns) for t, ns in fr.items()}
    c = Counter(); dists = []
    for s_, d_ in ge:
        ps, pd_ = g2p.get(s_), g2p.get(d_)
        if ps is not None and pd_ is not None and pd_ in ch.get(ps, []): c['tp'] += 1; continue
        for side, g, p in [('src', s_, ps), ('dst', d_, pd_)]:
            if p is None:
                t = int(gt_t[gidx[g]]); tr, ns = trees.get(t, (None, None))
                dd = tr.query(gxyz[gidx[g]])[0] if tr is not None else 99
                dists.append(dd)
                c['fn_%s_missing_nearest_%s' % (side, '<7' if dd < 7 else ('7-10' if dd < 10 else '>10'))] += 1
        if ps is not None and pd_ is not None:
            if not ch.get(ps) and pd_ not in par: c['fn_gap_both_free'] += 1
            elif not ch.get(ps): c['fn_src_ends_dst_taken'] += 1
            elif pd_ not in par: c['fn_src_elsewhere_dst_starts'] += 1
            else: c['fn_swap'] += 1
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        valid = (a in p2g and p2g[a] in gout) or (b in p2g and p2g[b] in gin)
        if not valid: continue
        if a in p2g and b in p2g and (p2g[a], p2g[b]) in ge: continue
        if a in p2g and b in p2g: c['fp_both_matched'] += 1
        elif a in p2g: c['fp_dst_unmatched'] += 1
        else: c['fp_src_unmatched'] += 1
    return dict(c)


if __name__ == '__main__':
    gdir = sys.argv[1]
    fs = sorted(glob.glob(gdir + '/*.json'))
    with Pool(48) as p: R = p.map(job, fs)
    tot = Counter()
    for r in R: tot.update(r)
    for k, v in sorted(tot.items()): print('%-40s %6d' % (k, v))
