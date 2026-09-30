"""Switch-type FN ('both': mu->x, y->mv with x,y unmatched): GT edge time gap, pred time gap, is mu->x->mv a pred path,
does x lie between? Also overall GT edge dt distribution."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
SETS = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
        'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
        't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def job(args):
    s, f = args
    name = Path(f).stem
    import evalx
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    out, prev = defaultdict(list), defaultdict(list)
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id']); out[x].append(y); prev[y].append(x)
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * evalx.SCALE for n, v in nodes.items()}
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    g2p = {v: k for k, v in p2g.items()}
    na = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gt_t = dict(zip([int(i) for i in na[K.NODE_ID].to_list()], [int(t) for t in na['t'].to_list()]))
    gpos = {int(i): np.array([z, y, x]) * evalx.SCALE for i, z, y, x in zip(na[K.NODE_ID].to_list(), na['z'].to_list(), na['y'].to_list(), na['x'].to_list())}
    ea = gt.edge_attrs(); GE = set()
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): GE.add((int(x), int(y)))
    c = Counter()
    for u, v in GE:
        c['gt_dt=%d' % (gt_t[v] - gt_t[u])] += 1
        mu, mv = g2p.get(u), g2p.get(v)
        if mu is None or mv is None or mv in out.get(mu, []): continue
        so, si = out.get(mu, []), prev.get(mv, [])
        if not (so and si): continue
        x, y = so[0], si[0]
        c['both'] += 1
        c['both_gtdt=%d' % (gt_t[v] - gt_t[u])] += 1
        c['both_pred_dt_mu_mv=%d' % (int(nodes[mv]['t']) - int(nodes[mu]['t']))] += 1
        c['both_x_is_y'] += int(x == y)
        c['both_path_mu_x_mv'] += int(mv in out.get(x, []))
        # x's distance to gt v and mv; mu distance to gt u
        c['both_dist_x_gtv<7'] += int(float(np.linalg.norm(pos[x] - gpos[v])) < 7)
        c['both_dist_y_gtu<7'] += int(float(np.linalg.norm(pos[y] - gpos[u])) < 7)
        c['both_dist_mu_gtu>4'] += int(float(np.linalg.norm(pos[mu] - gpos[u])) > 4)
        c['both_dist_mv_gtv>4'] += int(float(np.linalg.norm(pos[mv] - gpos[v])) > 4)
    return name, dict(c)


if __name__ == '__main__':
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob(SETS[s] + '/*.json'))]
    with Pool(64) as p: R = p.map(job, jobs)
    T = Counter()
    for n, c in R: T.update(c)
    for k in sorted(T): print(k, T[k])
