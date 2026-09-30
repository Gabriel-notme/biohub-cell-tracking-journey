import os, sys, json, glob
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, .40625, .40625])
DD = sys.argv[1]; GD = sys.argv[2]
def job(name):
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    fs = glob.glob(DD + '/' + name + '.zarr_*.npy') + glob.glob(DD + '/' + name + '_*.npy')
    if not fs: return None
    A = np.concatenate([np.load(f) for f in fs], 0)
    # dedupe by grid location keeping max prob
    key = A[:, :4].astype(np.int64); order = np.lexsort((-A[:, 4], key[:, 3], key[:, 2], key[:, 1], key[:, 0]))
    A = A[order]; key = key[order]; keep = np.ones(len(A), bool); keep[1:] = np.any(key[1:] != key[:-1], axis=1); A = A[keep]
    P = np.stack([A[:, 1], A[:, 2] * 4 + 1.5, A[:, 3] * 4 + 1.5], 1); T = A[:, 0].astype(int); pr = A[:, 4]
    nodes, edges = evalx.load_graph_json(GD + '/' + name + '.json')
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    mg = {int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gid = np.array(ga[K.NODE_ID].to_list()); gT = np.array(ga['t'].to_list()); gP = np.stack([ga[c].to_list() for c in 'zyx'], 1).astype(float)
    nid = list(nodes); nT = np.array([nodes[n]['t'] for n in nid]); nP = np.array([[nodes[n][c] for c in 'zyx'] for n in nid])
    out = dict(movie=name, n_peaks=len(A), n_nodes=len(nid), gt=[], extra=[])
    # offset check: nearest node for high-prob peaks
    offs = []
    for t in np.unique(T):
        m = (T == t) & (pr > 0.965); mn = nT == t
        if m.sum() == 0 or mn.sum() == 0: continue
        tr = cKDTree(nP[mn] * S); d, i = tr.query(P[m] * S)
        offs.append(np.c_[d, (nP[mn][i] - P[m])])
    offs = np.concatenate(offs, 0); out['off_med'] = np.median(offs[offs[:, 0] < 4, 1:], 0).tolist(); out['frac_hi_near_node'] = float((offs[:, 0] < 4).mean())
    for t in np.unique(gT):
        mg_t = gT == t; mp = T == t; mn = nT == t
        if mp.sum() == 0: continue
        trp = cKDTree(P[mp] * S); prt = pr[mp]
        trn = cKDTree(nP[mn] * S) if mn.sum() else None
        d, i = trp.query(gP[mg_t] * S, k=min(5, int(mp.sum())))
        d = np.atleast_2d(d); i = np.atleast_2d(i)
        for j, g in enumerate(gid[mg_t]):
            within = d[j] <= 5.0
            best = float(prt[i[j][within]].max()) if within.any() else 0.0
            dn = float(trn.query(gP[mg_t][j] * S)[0]) if trn is not None else 99
            out['gt'].append((int(g in mg), best, dn))
    # extra peaks: prob in bands, not within 4um of an existing node
    for t in np.unique(T):
        mp = (T == t) & (pr <= 0.965); mn = nT == t
        if mp.sum() == 0 or mn.sum() == 0: continue
        d, _ = cKDTree(nP[mn] * S).query(P[mp] * S)
        out['extra'] += list(pr[mp][d > 4.0])
    return out
if __name__ == '__main__':
    names = [l.strip() for l in open('/workspace/hold36.txt') if l.strip()]
    with Pool(36) as pool: res = [r for r in pool.map(job, names) if r]
    G = np.array([g for r in res for g in r['gt']]); E = np.array([e for r in res for e in r['extra']])
    print('movies', len(res), 'offset med', np.round(np.median([r['off_med'] for r in res], 0), 2).tolist(), 'frac hi near node', np.round(np.mean([r['frac_hi_near_node'] for r in res]), 3))
    for lab, nm in [(1, 'matched'), (0, 'unmatched')]:
        s = G[G[:, 0] == lab]
        print(nm, len(s), 'best-peak-within-5um prob quantiles', np.round(np.quantile(s[:, 1], [.1, .25, .5, .75, .9]), 3).tolist())
        for lo, hi in [(0.965, 1.01), (0.9, 0.965), (0.8, 0.9), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (-1, 0.2)]:
            print('   band [%.2f,%.2f): %d' % (lo, hi, int(((s[:, 1] >= lo) & (s[:, 1] < hi)).sum())))
    print('extra peaks (>4um from nodes) by band:', {b: int(((E >= b[0]) & (E < b[1])).sum()) for b in [(0.9, 0.965), (0.8, 0.9), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4)]})
    print('total nodes', sum(r['n_nodes'] for r in res))
