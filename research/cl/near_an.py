"""Near-miss anatomy: GT nodes (on GT edges) left unmatched while a pred node lies 7-10 um away.
For the nearest pred node n: is n matched to another GT node? displacement components (um); does n's track follow g's lineage
(parent/child of n matched to GT parent/child of g)? distance of the neighbours' midpoint to g."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, 0.40625, 0.40625])


def job(f):
    name = Path(f).stem
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    g2p = {v: k for k, v in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gid = [int(x) for x in ga[K.NODE_ID].to_list()]; gt_t = np.array(ga['t'].to_list()); graw = np.stack([ga[k].to_numpy() for k in 'zyx'], 1)
    gxyz = graw * S; gidx = {g: i for i, g in enumerate(gid)}
    ea = gt.edge_attrs(); src = [int(x) for x in ea[K.EDGE_SOURCE].to_list()]; dst = [int(x) for x in ea[K.EDGE_TARGET].to_list()]
    gch = defaultdict(list); gpar = {}
    for a, b in zip(src, dst): gch[a].append(b); gpar[b] = a
    ch = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); ch[a].append(b); par[b] = a
    # use the same rounded positions as the metric
    P = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx'], float) * S for n, v in nodes.items()}
    fr = defaultdict(list)
    for n, v in nodes.items(): fr[int(v['t'])].append(n)
    trees = {t: (cKDTree(np.array([P[n] for n in ns])), ns) for t, ns in fr.items()}
    ftrees = {}
    fdir = os.environ.get('FULLDIR')
    if fdir and Path(fdir, name + '.geff').exists():
        import zarr
        fg = zarr.open_group(str(Path(fdir, name + '.geff')), mode='r')
        fids = np.asarray(fg['nodes/ids'][:]).astype(np.int64); fT = np.asarray(fg['nodes/props/t/values'][:]).astype(int)
        fV = np.stack([np.asarray(fg['nodes/props/%s/values' % k][:]) for k in 'zyx'], 1) * S
        for t in np.unique(fT):
            ix = np.flatnonzero(fT == t); ftrees[int(t)] = (cKDTree(fV[ix]), fids[ix])
    rows = []
    for g in gid:
        if g in g2p or (g not in gpar and not gch.get(g)): continue
        t = int(gt_t[gidx[g]])
        if t not in trees: continue
        tr, ns = trees[t]
        d, j = tr.query(gxyz[gidx[g]])
        if not (7 <= d < 10): continue
        n = ns[j]
        disp = P[n] - gxyz[gidx[g]]
        follows_prev = n in par and par[n] in p2g and g in gpar and p2g[par[n]] == gpar[g]
        follows_next = any(c in p2g and p2g[c] in gch.get(g, []) for c in ch.get(n, []))
        mid = None
        if n in par and len(ch.get(n, [])) == 1:
            mid = float(np.linalg.norm((P[par[n]] + P[ch[n][0]]) / 2 - gxyz[gidx[g]]))
        # second nearest pred node distance
        dd2 = tr.query(gxyz[gidx[g]], k=2)[0][1] if len(ns) > 1 else 99
        fd, fin = 99., None
        if t in ftrees:
            fdd, fj = ftrees[t][0].query(gxyz[gidx[g]]); fd = float(fdd); fin = int(ftrees[t][1][fj]) in nodes
        rows.append(dict(movie=name, d=float(d), dz=float(disp[0]), dy=float(disp[1]), dx=float(disp[2]), n_matched=n in p2g,
                         fprev=bool(follows_prev), fnext=bool(follows_next), mid=mid, d2=float(dd2), zraw=float(graw[gidx[g], 0]), fd=fd, fin=fin))
    return rows


if __name__ == '__main__':
    gdir = sys.argv[1]
    with Pool(48) as p: R = [r for rs in p.map(job, sorted(glob.glob(gdir + '/*.json'))) for r in rs]
    json.dump(R, open('/workspace/cl/near_%s.json' % sys.argv[2], 'w'))
    print('near-miss GT nodes', len(R))
    print('nearest pred node matched to another GT node:', Counter(r['n_matched'] for r in R))
    print('follows prev / next:', Counter((r['fprev'], r['fnext']) for r in R))
    A = np.array([[r['dz'], r['dy'], r['dx']] for r in R])
    print('|disp| components median (z,y,x um):', np.median(np.abs(A), 0).round(2), ' mean signed:', A.mean(0).round(2))
    print('frac where |dz| is the largest component:', np.mean(np.argmax(np.abs(A), 1) == 0).round(3))
    m = [r['mid'] for r in R if r['mid'] is not None]
    print('neighbour-midpoint distance to GT: n %d, <7: %d, quantiles' % (len(m), sum(x < 7 for x in m)), np.percentile(m, [10, 50, 90]).round(2) if m else None)
    print('second-nearest pred node <7um:', sum(r['d2'] < 7 for r in R))
    print('nearest pre-ILP candidate detection distance: <4 %d, 4-7 %d, >=7 %d; of those <7 already in graph: %d' % (
        sum(r['fd'] < 4 for r in R), sum(4 <= r['fd'] < 7 for r in R), sum(r['fd'] >= 7 for r in R), sum(1 for r in R if r['fd'] < 7 and r['fin'])))
    print('GT z (slices) of near-misses quantiles', np.percentile([r['zraw'] for r in R], [10, 50, 90]))
