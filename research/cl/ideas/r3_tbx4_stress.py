"""Stress tests for r3_tbx4 (tbext, mode edge, K=100) on the 199 P14 graphs, official matching.
Per added edge: outcome (TP / FP=pred_valid&~matched / unevaluable), kind (ext/join), side, pre-ILP prob, walk depth, frame, per-frame
density ratio, local density; near-duplicate pairs among added nodes (<3.5 um, same frame); added nodes that matched GT."""
import os, sys, json, glob, warnings
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p56stage')
from pathlib import Path
from multiprocessing import Pool
from collections import defaultdict
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
VARS = [{'mode': 'edge', 'K': 100}, {'mode': 'edge', 'K': 100, 'join': True}, {'mode': 'edge', 'K': 100, 'pmin': 0.8}]


def rp(v):
    return np.array([max(0, int(round(float(v[k])))) for k in 'zyx'], float) * S


def job(f):
    warnings.filterwarnings('ignore')
    try:
        import numcodecs.blosc
        numcodecs.blosc.use_threads = False
    except Exception:
        pass
    import evalx, importlib
    import tracksdata as td
    from tracking_cellmot.metrics import evaluate, _evaluate_matched_graph
    from edge_link import load_full
    from scipy.spatial import cKDTree
    m = importlib.import_module('ideas.r3_tbx4')
    name = Path(f).stem
    s = [k for k in FULL if ('ps_p14_%s/' % k) in f][0]
    nodes, edges = evalx.load_graph_json(f)
    fg = FULL[s] + '/' + name + '.geff'
    fids, fT, fV, fE, fprob = load_full(fg)
    fedge = {(int(a), int(b)): float(p) for (a, b), p in zip(fE.tolist(), fprob.tolist())}
    gt, n_total = evalx.load_gt(name)
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    med = float(np.median([len(v) for v in byt.values()]))
    trees = {t: cKDTree(np.stack([rp(nodes[n]) for n in ns])) for t, ns in byt.items()}
    T0, T1 = min(byt), max(byt)
    out = {'movie': name, 'set': s, 'n_total': n_total, 'recs': [], 'dup': []}
    for vi, kw in enumerate(VARS):
        nn, ne, st = m.apply(nodes, edges, name=name, set=s, fullgeff=fg, zarr='', **kw)
        pred, mapping = evalx.to_graph(nn, ne, True)
        inv = {v: k for k, v in mapping.items()}
        evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
        ea = _evaluate_matched_graph(pred, gt)
        K = td.DEFAULT_ATTR_KEYS
        res = {}
        for a, b, mm, pv in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list(), ea[K.MATCHED_EDGE_MASK].to_list(), ea['pred_valid'].to_list()):
            res[(inv[int(a)], inv[int(b)])] = 'TP' if mm else ('FP' if pv else 'UN')
        na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
        matched = {inv[int(a)] for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
        # walk depth of added nodes
        par = {}; ch = {}
        for e in ne:
            a, b = int(e['source_id']), int(e['target_id'])
            if e.get('tb_ext'): par[b] = a; ch[a] = b
        depth = {}; sidem = {}
        for n in nn:
            if n in nodes: continue
            c = n; d = 0; ok = False
            while c in ch:
                c = ch[c]; d += 1
                if c in nodes: ok = True; sidem[n] = 'start'; break
            if not ok:
                c = n; d = 0
                while c in par:
                    c = par[c]; d += 1
                    if c in nodes: ok = True; sidem[n] = 'end'; break
            depth[n] = d if ok else -1
        for e in ne:
            if not (e.get('tb_ext') or e.get('tb_join')): continue
            a, b = int(e['source_id']), int(e['target_id'])
            new = a if depth.get(a, 0) >= depth.get(b, 0) else b
            side = sidem.get(new, '?')
            t = int(nn[new]['t'])
            p = rp(nn[new])
            ld = len(trees[t].query_ball_point(p, 10.0)) if t in trees else 0
            out['recs'].append({'vi': vi, 'kind': 'join' if e.get('tb_join') else 'ext', 'side': side, 'prob': fedge.get((a, b), -1.0),
                                'depth': depth.get(new, 0), 't': t, 'bnd': int(t - T0 <= 2 or T1 - t <= 2), 'fdens': len(byt.get(t, [])) / med,
                                'ldens': ld, 'res': res.get((a, b), 'X'), 'new_matched': int(new in matched)})
        # near-duplicate pairs among added nodes (same frame, < 3.5 um)
        addt = defaultdict(list)
        for n in nn:
            if n not in nodes: addt[int(nn[n]['t'])].append(rp(nn[n]))
        nd = 0
        for t, P in addt.items():
            if len(P) > 1:
                nd += len(cKDTree(np.stack(P)).query_pairs(3.5))
        out['dup'].append({'vi': vi, 'n_add': sum(len(v) for v in addt.values()), 'dup_pairs': nd,
                           'add_matched': sum(1 for n in nn if n not in nodes and n in matched)})
    return out


if __name__ == '__main__':
    fs = [f for s in FULL for f in sorted(glob.glob('/workspace/cl/ps_p14_%s/graphs/*.json' % s))]
    with Pool(int(os.environ.get('RULE_POOL', '48')), maxtasksperchild=4) as p:
        R = p.map(job, fs, chunksize=1)
    json.dump(R, open('/workspace/cl/ideas/r3_tbx4_stress_rows.json', 'w'))
    print('done', len(R))
