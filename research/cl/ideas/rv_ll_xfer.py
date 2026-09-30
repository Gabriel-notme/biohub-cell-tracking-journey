"""Transfer review of pp_longlink: for each added long link record label (TP/FP/U), fe, whether B5 left the endpoints free,
local-flow residual (vs displacement of nearby P13 edges at the same frame), frame-global motion; plus a reference sample of
existing P13 edges (label, length, fe, flow residual) to test annotation-enrichment of pre-ILP edge_prob (in-sample memorization).
usage: python3 ideas/rv_ll_xfer.py"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
sys.path.insert(0, '/workspace/p56stage')
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import rule_eval as RE
S = np.array([1.625, .40625, .40625])


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
    from scipy.spatial import cKDTree
    from ideas import pp_longlink
    from edge_link import load_full
    name = Path(f).stem; st = [k for k in RE.FULL if ('_%s/' % k) in f][0]
    fg = RE.FULL[st] + '/' + name + '.geff'
    nodes, edges = evalx.load_graph_json(f)
    n2, e2, _ = pp_longlink.apply(nodes, edges, fullgeff=fg, fe_min=0.5)
    added = [(int(e['source_id']), int(e['target_id'])) for e in e2 if 'long_link' in e]
    gt, _ = evalx.load_gt(name)
    lab = labels(n2, e2, gt)
    fids, fT, fV, fE, fprob = load_full(fg)
    fe = {(int(a), int(b)): float(p) for (a, b), p in zip(fE.tolist(), fprob.tolist())}
    # B5 free-end status
    bn, be = evalx.load_graph_json(RE.SRC['b5'][st] + '/' + name + '.json')
    b_child = {int(e['source_id']) for e in be}; b_par = {int(e['target_id']) for e in be}
    pos = {n: np.array([v['z'], v['y'], v['x']], float) * S for n, v in n2.items()}
    # displacement field from existing P13 edges (dt=1)
    by_t = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        if int(nodes[b]['t']) != int(nodes[a]['t']) + 1: continue
        by_t.setdefault(int(nodes[a]['t']), []).append((a, pos[b] - pos[a]))
    trees = {}
    for t, L in by_t.items():
        P = np.array([pos[a] for a, _ in L]); D = np.array([d for _, d in L])
        trees[t] = (cKDTree(P), D, np.array([a for a, _ in L]), float(np.median(np.linalg.norm(D, axis=1))))

    def flow(a, v, R=25.0):
        t = int(n2[a]['t'])
        if t not in trees: return None, None, None, 0
        tr, D, ids, gmed = trees[t]
        idx = [i for i in tr.query_ball_point(pos[a], R) if ids[i] != a]
        if len(idx) < 3: return None, None, gmed, len(idx)
        mf = np.median(D[idx], axis=0)
        return float(np.linalg.norm(v - mf)), float(np.linalg.norm(mf)), gmed, len(idx)

    rows = []
    for a, b in added:
        v = pos[b] - pos[a]
        res, fm, gmed, nn = flow(a, v)
        rows.append({'kind': 'long', 'lab': lab.get((a, b), 'X'), 'd': float(np.linalg.norm(v)), 'fe': fe.get((a, b), -1),
                     'b5free': int(a not in b_child and b not in b_par), 'res': res, 'fm': fm, 'gmed': gmed, 'nn': nn, 't': int(n2[a]['t'])})
    # reference sample of existing P13 edges: all evaluable + 3% of unevaluable, lengths recorded
    rng = np.random.default_rng(hash(name) % 2**32)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        l = lab.get((a, b), 'X')
        d = float(np.linalg.norm(pos[b] - pos[a]))
        if l == 'U' and d < 8 and rng.random() > 0.03: continue
        res, fm, gmed, nn = flow(a, pos[b] - pos[a])
        rows.append({'kind': 'p13', 'lab': l, 'd': d, 'fe': fe.get((a, b), -1), 'res': res, 'fm': fm, 'gmed': gmed, 'nn': nn})
    for r in rows: r['movie'] = name; r['set'] = st
    return rows


if __name__ == '__main__':
    fs = [f for s, d in RE.SRC['p13'].items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(24, maxtasksperchild=4) as p: res = p.map(job, fs, chunksize=1)
    allr = [r for rs in res for r in rs]
    json.dump(allr, open('/workspace/cl/ideas/rv_ll_xfer_rows.json', 'w'))
    print('rows', len(allr))
