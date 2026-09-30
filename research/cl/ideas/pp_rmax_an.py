"""Edge-level diff between P13 and the replica with edge_link RMAX[1]=X: which edges appear/disappear, their length, origin, label
(TP / evaluable FP / unevaluable). Also GT edge length distribution. usage: python3 ideas/pp_rmax_an.py X"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import Counter
from multiprocessing import Pool
import numpy as np
import rule_eval as RE
S = np.array([1.625, .40625, .40625])
X = float(sys.argv[1])


def labels(nodes, edges, gt):
    import evalx
    import tracksdata as td
    from tracking_cellmot import metrics as M
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    M._evaluate(pred, gt, 'jaccard', evalx.SCALE, 7.)
    ea = M._evaluate_matched_graph(pred, gt)
    K = td.DEFAULT_ATTR_KEYS
    lab = {}
    for s, d, m, v in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list(), ea[K.MATCHED_EDGE_MASK].to_list(), ea['pred_valid'].to_list()):
        lab[(inv[s], inv[d])] = 'TP' if m else ('FP' if v else 'U')
    return lab


def job(f):
    import evalx
    from ideas import pp_pipe
    name = Path(f).stem; st = [k for k in RE.FULL if ('_%s/' % k) in f][0]
    nodes, edges = evalx.load_graph_json(f)
    n2, e2, _ = pp_pipe.apply(nodes, edges, name=name, set=st, fullgeff=RE.FULL[st] + '/' + name + '.geff', rmax1=X)
    gt, _ = evalx.load_gt(name)
    l1 = labels(nodes, edges, gt); l2 = labels(n2, e2, gt)
    orig = lambda e: 'el' if 'edge_link' in e else ('rl' if 'relink' in e else ('dc' if 'div_complete' in e else 'b5'))
    o1 = {(int(e['source_id']), int(e['target_id'])): orig(e) for e in edges}; o2 = {(int(e['source_id']), int(e['target_id'])): orig(e) for e in e2}
    def ln(nn, s, d): return float(np.linalg.norm((np.array([nn[s][k] for k in 'zyx']) - np.array([nn[d][k] for k in 'zyx'])) * S))
    c = Counter()
    for k in set(l1) - set(l2): c[('removed', o1[k], l1[k], min(int(ln(nodes, *k) // 2) * 2, 20))] += 1
    for k in set(l2) - set(l1): c[('added', o2[k], l2[k], min(int(ln(n2, *k) // 2) * 2, 20))] += 1
    # GT edge lengths
    import tracksdata as td
    K = td.DEFAULT_ATTR_KEYS
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 'z', 'y', 'x'])
    gp = {int(i): np.array([z, y, x]) * S for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    eat = gt.edge_attrs()
    for s, d in zip(eat[K.EDGE_SOURCE].to_list(), eat[K.EDGE_TARGET].to_list()):
        L = float(np.linalg.norm(gp[int(s)] - gp[int(d)]))
        c[('gt_len', '', '', min(int(L // 2) * 2, 20))] += 1
    return name[:4], c


if __name__ == '__main__':
    fs = [f for s, d in RE.SRC['p13'].items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(60) as p: R = p.map(job, fs, chunksize=1)
    C = Counter(); CE = {'44b6': Counter(), '6bba': Counter()}
    for emb, c in R: C.update(c); CE[emb].update(c)
    for k in sorted(C): print(k, C[k], '| 44b6', CE['44b6'][k], '6bba', CE['6bba'][k])
