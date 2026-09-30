"""Sanity check: my edge labelling (matched ids after evaluate) vs official edge TP/FP counts on a few movies."""
import os, sys, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from collections import defaultdict
from pathlib import Path
import evalx
from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall
K = evalx.K
for f in sorted(glob.glob('/workspace/cl/ps_p14_hold36/graphs/*.json'))[:3] + sorted(glob.glob('/workspace/cl/ps_p14_hold36/graphs/6bba*.json'))[:2]:
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    gt, nt = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    er = evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    row = per_sample_metrics(er, nt, node_recall(pred, gt))
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ea = gt.edge_attrs()
    gsucc = defaultdict(list); gpar = {}
    for a, b in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()):
        gsucc[int(a)].append(int(b)); gpar[int(b)] = int(a)
    tp = fp = 0
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id'])
        gx, gy = p2g.get(x), p2g.get(y)
        if gx is not None and gy is not None and gy in gsucc.get(gx, []): tp += 1
        elif (gx is not None and gsucc.get(gx)) or (gy is not None and gy in gpar): fp += 1
    print(name, 'mine TP', tp, 'FP', fp, '| official', {k: row[k] for k in row if 'edge' in k and ('tp' in k or 'fp' in k or 'fn' in k)})
