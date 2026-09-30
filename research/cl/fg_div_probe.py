"""Does the pre-ILP candidate graph encode divisions (out-degree 2), and do missed GT divisions appear there?"""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np


def job(a):
    s, name, g, f = a
    import evalx, edge_link
    K = evalx.K
    from tracking_cellmot.division_metrics import score_divisions
    nodes, edges = evalx.load_graph_json(Path(g) / (name + '.json'))
    fids, fT, fV, fE, fprob = edge_link.load_full(Path(f) / (name + '.geff'))
    fout = defaultdict(list)
    for (u, v), p in zip(fE.tolist(), fprob.tolist()): fout[u].append((v, p))
    c = Counter()
    c['fg_nodes'] = len(fids); c['fg_outdeg2'] = sum(1 for v in fout.values() if len(v) >= 2)
    succ = defaultdict(list)
    for e in edges: succ[int(e['source_id'])].append(int(e['target_id']))
    c['pred_forks'] = sum(1 for v in succ.values() if len(v) >= 2)
    c['pred_forks_in_fg'] = sum(1 for p, v in succ.items() if len(v) >= 2 and len(fout.get(p, [])) >= 2)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    tp = {inv[x] for x in res.tp_forks}; fp = {inv[x] for x in res.fp_forks}
    c['tp_forks'] = len(tp); c['tp_forks_fg2'] = sum(1 for p in tp if len(fout.get(p, [])) >= 2)
    c['fp_forks'] = len(fp); c['fp_forks_fg2'] = sum(1 for p in fp if len(fout.get(p, [])) >= 2)
    # missed GT divisions: is the matched parent (or its neighbours) an out-degree-2 node in fg?
    from tracking_cellmot.metrics import evaluate
    pred2, mapping2 = evalx.to_graph(nodes, edges); inv2 = {v: k for k, v in mapping2.items()}
    evaluate(pred2, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred2.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    g2p = {int(y): inv2[int(x)] for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    ea = gt.edge_attrs(); gsucc = defaultdict(list); gpar = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(x)].append(int(y)); gpar[int(y)] = int(x)
    par = {}
    for e in edges: par[int(e['target_id'])] = int(e['source_id'])
    for gd, sc in res.scores.items():
        if sc: continue
        c['fn'] += 1
        p = g2p.get(int(gd))
        if p is None: c['fn_parent_unmatched'] += 1; continue
        cand = [p] + ([par[p]] if p in par else []) + succ.get(p, [])
        if any(len(fout.get(x, [])) >= 2 for x in cand): c['fn_fg_fork_near'] += 1
    return c


if __name__ == '__main__':
    SETS = [('hold36', '/workspace/hold36.txt', '/workspace/cl/ps_p5_hold36/graphs', '/workspace/runs/fullgraph_hold36'),
            ('t127a', '/workspace/t127a.txt', '/workspace/cl/p5tr/t127a', '/workspace/sync4/runs/fullgraph_t127a')]
    for s, lst, g, f in SETS:
        names = [l.strip() for l in open(lst) if l.strip()]
        with Pool(36) as pool: cs = pool.map(job, [(s, n, g, f) for n in names])
        tot = Counter()
        for c in cs: tot.update(c)
        print(s, dict(sorted(tot.items())))
