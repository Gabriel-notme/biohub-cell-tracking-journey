"""For 'both' switch FNs vs normal TP edges: distance of GT node u to its matched pred node, nearest OTHER lineage node,
and nearest fullgraph (pre-ILP) detection not in the lineage graph. Tells whether the GT cell was detected at all."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
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
    S = evalx.SCALE
    drop = np.array([i not in nodes for i in fids.tolist()])
    dtree = {t: cKDTree(fV[(fT == t) & drop] * S) for t in np.unique(fT) if ((fT == t) & drop).any()}
    out, prev = defaultdict(list), defaultdict(list)
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id']); out[x].append(y); prev[y].append(x)
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * S for n, v in nodes.items()}
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    ltree = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in frames.items()}
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    mp = _match_full(pred, gt, S, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    g2p = {v: k for k, v in p2g.items()}
    na = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gt_t = dict(zip([int(i) for i in na[K.NODE_ID].to_list()], [int(t) for t in na['t'].to_list()]))
    gpos = {int(i): np.array([z, y, x]) * S for i, z, y, x in zip(na[K.NODE_ID].to_list(), na['z'].to_list(), na['y'].to_list(), na['x'].to_list())}
    ea = gt.edge_attrs(); GE = set()
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): GE.add((int(x), int(y)))
    rows = []
    rng = np.random.default_rng(0)
    for u, v in GE:
        mu, mv = g2p.get(u), g2p.get(v)
        if mu is None or mv is None: continue
        if mv in out.get(mu, []): kind = 'tp'
        elif out.get(mu) and prev.get(mv): kind = 'both'
        else: continue
        if kind == 'tp' and rng.random() > 0.05: continue
        t = gt_t[u]; g = gpos[u]
        dm = float(np.linalg.norm(pos[mu] - g))
        ns, tr = ltree[t]; dd, ii = tr.query(g, k=3)
        other = [d for d, i in zip(dd, ii) if ns[i] != mu]
        dl2 = float(other[0]) if other else 99.
        ddrop = float(dtree[t].query(g, k=1)[0]) if t in dtree else 99.
        rows.append((kind, dm, dl2, ddrop))
    return rows


if __name__ == '__main__':
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob(SETS[s][0] + '/*.json'))]
    with Pool(64) as p: R = [r for rs in p.map(job, jobs) for r in rs]
    for kind in ['tp', 'both']:
        A = np.array([r[1:] for r in R if r[0] == kind])
        print(kind, len(A))
        for j, nm in enumerate(['d(u, matched pred)', 'd(u, 2nd lineage node)', 'd(u, dropped fullgraph det)']):
            q = np.percentile(A[:, j], [10, 25, 50, 75, 90])
            print('   %-28s p10 %.2f p25 %.2f p50 %.2f p75 %.2f p90 %.2f' % (nm, *q))
        print('   frac dropped det within 2.5um: %.3f, within 4um: %.3f' % ((A[:, 2] < 2.5).mean(), (A[:, 2] < 4).mean()))
