"""Close same-frame pred pairs (mutual NN, d < 9 um) on B5 graphs: GT status of each pair, restricted to pairs where at least one
member is within 7 um of an annotated GT node. Classes: both_matched (two annotated cells -> merging would destroy one),
one_matched_other_near_same_gt (other member within 7 um of the same GT node: ambiguous/over-seg), one_matched_other_far.
Also: GT same-frame nearest-neighbour distances among annotated nodes, and the midpoint's distance to the GT node vs matched member's."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
SETS = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
        'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
        't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}
BINS = [0, 3, 4, 5, 6, 7, 8, 9]


def job(args):
    s, f = args
    name = Path(f).stem
    import evalx
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    K = evalx.K; S = evalx.SCALE
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * S for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    mp = _match_full(pred, gt, S, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    na = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gid = [int(i) for i in na[K.NODE_ID].to_list()]; gt_t = [int(t) for t in na['t'].to_list()]
    gP = np.stack([np.array(na[k].to_list()) for k in 'zyx'], 1) * S
    gframes = defaultdict(list)
    for i, t in zip(range(len(gid)), gt_t): gframes[t].append(i)
    c = Counter(); gnn = []
    for t, ns in frames.items():
        P = np.array([pos[n] for n in ns]); tr = cKDTree(P)
        gi = gframes.get(t, [])
        if not gi: continue
        gtr = cKDTree(gP[gi])
        if len(gi) > 1:
            dd, _ = gtr.query(gP[gi], k=2); gnn += dd[:, 1].tolist()
        dd, ii = tr.query(P, k=2)
        for a in range(len(ns)):
            b = ii[a, 1]
            if ii[b, 1] != a or a > b or dd[a, 1] >= 9: continue
            na_, nb_ = ns[a], ns[b]
            near = gtr.query_ball_point((P[a] + P[b]) / 2, 9.0)
            if not near: continue  # unannotated region
            bn = BINS[np.searchsorted(BINS, dd[a, 1], side='right') - 1]
            ma_, mb_ = na_ in p2g, nb_ in p2g
            if ma_ and mb_: cls = 'both_matched'
            elif ma_ or mb_:
                m, o = (na_, nb_) if ma_ else (nb_, na_)
                g = gid.index(p2g[m]) if False else None
                gpos = gP[[k for k in gi if gid[k] == p2g[m]][0]]
                do = float(np.linalg.norm(pos[o] - gpos)); dmid = float(np.linalg.norm((P[a] + P[b]) / 2 - gpos)); dm = float(np.linalg.norm(pos[m] - gpos))
                cls = 'one_other_near' if do < 7 else 'one_other_far'
                if do < 7: c['mid_closer_%d' % bn] += int(dmid < dm); c['mid_within7_%d' % bn] += int(dmid < 7)
            else: cls = 'none_matched'
            c['%s_%d' % (cls, bn)] += 1
    return name, dict(c), gnn


if __name__ == '__main__':
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob(SETS[s] + '/*.json'))]
    with Pool(64) as p: R = p.map(job, jobs)
    for emb in ['44b6', '6bba']:
        T = Counter()
        for n, c, g in R:
            if n.startswith(emb): T.update(c)
        print('==', emb)
        for bn in BINS[:-1]:
            print('  d in [%d,..) both_matched %5d one_other_near %5d (mid closer %5d, mid<7 %5d) one_other_far %5d none %5d' % (
                bn, T['both_matched_%d' % bn], T['one_other_near_%d' % bn], T['mid_closer_%d' % bn], T['mid_within7_%d' % bn], T['one_other_far_%d' % bn], T['none_matched_%d' % bn]))
    g = np.array([x for n, c, gg in R for x in gg])
    print('GT same-frame NN distance (annotated only): p1 %.2f p5 %.2f p10 %.2f p25 %.2f p50 %.2f; frac<6 %.4f frac<8 %.4f' % (*np.percentile(g, [1, 5, 10, 25, 50]), (g < 6).mean(), (g < 8).mean()))
