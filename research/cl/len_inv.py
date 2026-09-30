"""GT invariant check: edge displacement (3D, z, xy) distribution in GT vs predicted edges (B5 graphs) by official status."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
SETS = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
        'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
        't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def job(f):
    import evalx
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    K = evalx.K; S = np.array(evalx.SCALE)
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    na = gt.node_attrs(attr_keys=[K.NODE_ID, 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x]) * S for i, z, y, x in zip(na[K.NODE_ID].to_list(), na['z'].to_list(), na['y'].to_list(), na['x'].to_list())}
    ea = gt.edge_attrs(); GE = set(); gs = defaultdict(list); gp = defaultdict(list); G = []
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()):
        x, y = int(x), int(y); GE.add((x, y)); gs[x].append(y); gp[y].append(x); d = gpos[y] - gpos[x]
        G.append((float(np.linalg.norm(d)), float(abs(d[0])), float(np.linalg.norm(d[1:]))))
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    ma = _matched_node_attrs(_match_full(pred, gt, evalx.SCALE, 7.))
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    P = []
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); d = pos[b] - pos[a]
        ga, gb = p2g.get(a), p2g.get(b)
        ev = (ga is not None and gs.get(ga)) or (gb is not None and gp.get(gb))
        st = 0 if not ev else (1 if (ga is not None and gb is not None and (ga, gb) in GE) else 2)
        P.append((float(np.linalg.norm(d)), float(abs(d[0])), float(np.linalg.norm(d[1:])), st))
    return G, P


if __name__ == '__main__':
    fs = [f for s in SETS for f in sorted(glob.glob(SETS[s] + '/*.json'))]
    with Pool(64) as p: R = p.map(job, fs)
    G = np.array([g for r in R for g in r[0]]); P = np.array([q for r in R for q in r[1]])
    for j, nm in enumerate(['3D', 'z', 'xy']):
        q = np.percentile(G[:, j], [50, 99, 99.9, 99.99, 100])
        print('GT %-3s p50 %.2f p99 %.2f p99.9 %.2f p99.99 %.2f max %.2f' % (nm, *q))
        for th in [q[3], q[4]]:
            sel = P[:, j] > th
            print('   pred edges with %s > %.2f: total %d, unevaluable %d, TP %d, FP %d' % (nm, th, sel.sum(), (sel & (P[:, 3] == 0)).sum(), (sel & (P[:, 3] == 1)).sum(), (sel & (P[:, 3] == 2)).sum()))
