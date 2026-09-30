"""GT invariants for edges: per-frame displacement (um), per-axis displacement, turning (acceleration) — vs predicted evaluable edges
(official validity) on a config's outputs, split TP / FP. Looks for predicted edges that violate the GT envelope."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])


def gt_job(g):
    import zarr
    z = zarr.open_group(g, mode='r')
    ids = np.asarray(z['nodes/ids']).tolist(); P = np.stack([np.asarray(z['nodes/props/%s/values' % k]) for k in 'zyx'], 1) * S
    pos = dict(zip(ids, P)); E = np.asarray(z['edges/ids']).tolist(); par = {b: a for a, b in E}
    out = []
    for a, b in E:
        d = pos[b] - pos[a]; acc = None
        if a in par: acc = float(np.linalg.norm((pos[b] - pos[a]) - (pos[a] - pos[par[a]])))
        out.append((float(np.linalg.norm(d)), float(abs(d[0])), float(np.linalg.norm(d[1:])), acc))
    return out


def pred_job(f):
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
    ea = gt.edge_attrs(); src = ea[K.EDGE_SOURCE].to_list(); dst = ea[K.EDGE_TARGET].to_list()
    ge = set(zip(src, dst)); gout = set(src); gin = set(dst)
    par = {}
    for e in edges: par[int(e['target_id'])] = int(e['source_id'])
    pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx'], float) * S for n, v in nodes.items()}
    out = []
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        valid = (a in p2g and p2g[a] in gout) or (b in p2g and p2g[b] in gin)
        d = pos[b] - pos[a]; acc = float(np.linalg.norm((pos[b] - pos[a]) - (pos[a] - pos[par[a]]))) if a in par else None
        lab = 'U' if not valid else ('TP' if (a in p2g and b in p2g and (p2g[a], p2g[b]) in ge) else 'FP')
        out.append((float(np.linalg.norm(d)), float(abs(d[0])), float(np.linalg.norm(d[1:])), acc, lab))
    return out


if __name__ == '__main__':
    cfg = sys.argv[1]
    with Pool(64) as p:
        G = [r for rs in p.map(gt_job, sorted(glob.glob('/workspace/data/train/*.geff'))) for r in rs]
        fs = [f for T in sys.argv[2:] for f in sorted(glob.glob('/workspace/cl/ps_%s_%s/graphs/*.json' % (cfg, T)))]
        Pd = [r for rs in p.map(pred_job, fs) for r in rs]
    G = np.array([(a, b, c, -1 if d is None else d) for a, b, c, d in G])
    print('GT edges', len(G))
    for i, nm in enumerate(['disp', 'dz', 'dyx', 'acc']):
        v = G[:, i][G[:, i] >= 0]
        print('  GT %-5s q50 %.2f q99 %.2f q999 %.2f max %.2f' % (nm, np.percentile(v, 50), np.percentile(v, 99), np.percentile(v, 99.9), v.max()))
    gmax = {nm: G[:, i][G[:, i] >= 0].max() for i, nm in enumerate(['disp', 'dz', 'dyx', 'acc'])}
    for i, nm in enumerate(['disp', 'dz', 'dyx', 'acc']):
        for thr_name, thr in [('>GTmax', gmax[nm]), ('>GTq999', np.percentile(G[:, i][G[:, i] >= 0], 99.9))]:
            sel = [r[4] for r in Pd if r[i] is not None and r[i] > thr]
            from collections import Counter
            print('  pred %-5s %-8s (%.2f): %s' % (nm, thr_name, thr, dict(Counter(sel))))
