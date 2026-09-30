"""Systematic localisation bias? For matched (pred, GT) node pairs in P13 outputs: mean/median displacement per axis (rounded
export coords), per embryo and per set. Then test a global shift (voxels) on all nodes with the official metric."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from multiprocessing import Pool
import numpy as np


def disp(a):
    s, f = a
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(Path(f).stem)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 'z', 'y', 'x']); gp = {int(i): np.array([z, y, x], float) for i, z, y, x in zip(*[ga[k].to_list() for k in [K.NODE_ID, 'z', 'y', 'x']])}
    out = []
    for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()):
        if y is None or int(y) == -1: continue
        v = nodes[inv[int(x)]]; p = np.array([max(0, int(round(v[k]))) for k in 'zyx'], float)
        out.append(p - gp[int(y)])
    return s, Path(f).stem[:4], np.array(out)


def shifted(a):
    s, f, sh = a
    import evalx
    nodes, edges = evalx.load_graph_json(f)
    if sh is not None:
        nodes = {n: dict(v, z=v['z'] + sh[0], y=v['y'] + sh[1], x=v['x'] + sh[2]) for n, v in nodes.items()}
    r = evalx.score_movie(Path(f).stem, nodes, edges); r['set'] = s; r['sh'] = sh; return r


if __name__ == '__main__':
    fs = [(s, f) for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b'] for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(96) as p: D = p.map(disp, fs)
    for emb in ['44b6', '6bba']:
        A = np.concatenate([d for s, e, d in D if e == emb and len(d)])
        print(emb, 'matched pairs', len(A), 'mean dz,dy,dx (voxels) %s  median %s  (um: %s)' % (A.mean(0).round(3), np.median(A, 0), (A.mean(0) * np.array([1.625, .40625, .40625])).round(3)))
    for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b']:
        A = np.concatenate([d for ss, e, d in D if ss == s and len(d)]); print('  ', s, A.mean(0).round(3))
    A = np.concatenate([d for s, e, d in D if len(d)]); m = -A.mean(0)
    shifts = [None, [round(float(m[0]), 2), 0.0, 0.0], [round(float(m[0]), 2), round(float(m[1]), 2), round(float(m[2]), 2)]]
    print('testing shifts (voxels, applied to pred):', shifts[1:])
    with Pool(96) as p: R = p.map(shifted, [(s, f, sh) for sh in shifts for s, f in fs])
    from tracking_cellmot.metrics import summarise
    for sh in shifts:
        line = '%-30s' % (sh,)
        for nm, flt in [('44b6', lambda r: r['movie'].startswith('44b6')), ('6bba', lambda r: r['movie'].startswith('6bba')), ('clean', lambda r: r['set'] in ('hold36', 'prev4'))]:
            line += ' %s %.6f' % (nm, summarise([r for r in R if r['sh'] == sh and flt(r)])['score'])
        print(line)
