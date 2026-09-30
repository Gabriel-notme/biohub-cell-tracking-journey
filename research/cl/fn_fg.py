"""For remaining FN GT edges in P5 (both endpoints matched), is the link present in the pre-ILP candidate graph?"""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np


def job(name):
    import evalx, edge_link
    K = evalx.K
    nodes, edges = evalx.load_graph_json(Path('/workspace/cl/ps_p5_hold36/graphs') / (name + '.json'))
    fids, fT, fV, fE, fprob = edge_link.load_full(Path('/workspace/runs/fullgraph_hold36') / (name + '.geff'))
    fset = {(int(a), int(b)): p for (a, b), p in zip(fE.tolist(), fprob.tolist())}
    fout = defaultdict(list)
    for (a, b), p in fset.items(): fout[a].append((b, p))
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    from tracking_cellmot.metrics import evaluate
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    g2p = {g: p for p, g in p2g.items()}
    succ = defaultdict(list); par = {}
    for e in edges: succ[int(e['source_id'])].append(int(e['target_id'])); par[int(e['target_id'])] = int(e['source_id'])
    ea = gt.edge_attrs()
    c = Counter()
    for gs, gd in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()):
        s, d = g2p.get(int(gs)), g2p.get(int(gd))
        if s is None or d is None: c['missing_node'] += 1; continue
        if d in succ.get(s, []): continue
        inf = (s, d) in fset
        typ = ('s_free' if not succ.get(s) else 's_linked') + '/' + ('d_free' if d not in par else 'd_taken')
        # does the pre-ILP graph link s to anything / d from anything?
        c[(typ, 'in_fullgraph' if inf else ('s_has_fg_out' if fout.get(s) else 'no_fg'))] += 1
    return c


if __name__ == '__main__':
    names = [l.strip() for l in open('/workspace/hold36.txt') if l.strip()]
    with Pool(36) as pool: cs = pool.map(job, names)
    tot = Counter()
    for c in cs: tot.update(c)
    for k, v in sorted(tot.items(), key=lambda x: -x[1]): print(k, v)
