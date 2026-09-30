"""Anatomy of switch-type FN edges on B5 graphs (GT u->v, both matched to mu, mv; mu->x with x!=mv and/or y->mv with y!=mu).
Is (mu,mv) in the pre-ILP fullgraph candidate edges? distance? are x / y GT-matched? is x ~ a duplicate of mv (dist)? fork involvement?"""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
SETS = {'hold36': ('/workspace/runs/b5f_hold36/working/lineage_graphs', '/workspace/runs/fullgraph_hold36'),
        'prev4': ('/workspace/runs/b5f_prev4/working/lineage_graphs', '/workspace/runs/fullgraph_prev4'),
        'audit32': ('/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', '/workspace/sync3/runs/fullgraph_audit32'),
        't127a': ('/workspace/sync4/runs/b5f_t127a/working/lineage_graphs', '/workspace/sync4/runs/fullgraph_t127a'),
        't127b': ('/workspace/sync3/runs/b5f_t127b/working/lineage_graphs', '/workspace/sync3/runs/fullgraph_t127b')}


def job(args):
    s, f = args
    name = Path(f).stem
    import evalx, edge_link
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    fids, fT, fV, fE, fprob = edge_link.load_full(Path(SETS[s][1]) / (name + '.geff'))
    fedge = {(int(a), int(b)): float(p) for (a, b), p in zip(fE.tolist(), fprob.tolist())}
    out, prev = defaultdict(list), defaultdict(list)
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id']); out[x].append(y); prev[y].append(x)
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * evalx.SCALE for n, v in nodes.items()}
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    g2p = {v: k for k, v in p2g.items()}
    ea = gt.edge_attrs(); GE = set(); gs = defaultdict(list)
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): GE.add((int(x), int(y))); gs[int(x)].append(int(y))
    c = Counter()
    for u, v in GE:
        mu, mv = g2p.get(u), g2p.get(v)
        if mu is None or mv is None or mv in out.get(mu, []): continue
        so, si = out.get(mu, []), prev.get(mv, [])
        if not so and not si: continue
        key = 'both' if (so and si) else ('out' if so else 'in')
        c[key] += 1
        c[key + '_infull'] += int((mu, mv) in fedge)
        d = float(np.linalg.norm(pos[mu] - pos[mv])); c[key + '_d<10'] += int(d < 10)
        if so:
            x = so[0]; c[key + '_x_matched'] += int(x in p2g); c[key + '_mu_fork'] += int(len(so) >= 2)
            c[key + '_x_near_mv<4'] += int(float(np.linalg.norm(pos[x] - pos[mv])) < 4)
        if si:
            y = si[0]; c[key + '_y_matched'] += int(y in p2g); c[key + '_y_fork'] += int(len(out.get(y, [])) >= 2)
        # u divides in GT?
        c[key + '_gt_div'] += int(len(gs.get(u, [])) >= 2)
    return name, dict(c)


if __name__ == '__main__':
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob(SETS[s][0] + '/*.json'))]
    with Pool(64) as p: R = p.map(job, jobs)
    T = Counter()
    for n, c in R: T.update(c)
    for k in sorted(T): print(k, T[k])
