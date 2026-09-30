"""Are annotated cells rarer near the image borders (xy edge, z top/bottom)? Per embryo: predicted nodes (P11) binned by distance to the
nearest xy border and to the z border; TP edges per predicted node in each bin vs the per-movie break-even (0.1*TP_movie/N_total/m)."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
SHAPE = np.array([64, 256, 256])
XYB = [0, 3, 6, 10, 15, 1e9]; ZB = [0, 2, 4, 8, 1e9]  # um / slices


def job(f):
    name = Path(f).stem
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    gt, ntot = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    er = evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    ea = gt.edge_attrs(); ge = set(zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()))
    tpn = defaultdict(float)  # TP edges attributed half to each endpoint
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        if a in p2g and b in p2g and (p2g[a], p2g[b]) in ge: tpn[a] += .5; tpn[b] += .5
    TP = er.edge_tp; m = 1 - 0.1 * (len(nodes) - ntot) / ntot; be = 0.1 * TP / ntot / m
    cnt = defaultdict(lambda: [0, 0.])
    for n, v in nodes.items():
        dxy = min(v['y'], SHAPE[1] - 1 - v['y'], v['x'], SHAPE[2] - 1 - v['x']) * S[1]
        dz = min(v['z'], SHAPE[0] - 1 - v['z'])
        bx = np.searchsorted(XYB, dxy, side='right') - 1; bz = np.searchsorted(ZB, dz, side='right') - 1
        c = cnt[(int(bx), int(bz))]; c[0] += 1; c[1] += tpn.get(n, 0.)
    return name, be, {'%d_%d' % k: v for k, v in cnt.items()}


if __name__ == '__main__':
    fs = [f for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b'] for f in sorted(glob.glob('/workspace/cl/ps_p11_%s/graphs/*.json' % s))]
    with Pool(96) as p: R = p.map(job, fs)
    for emb in ['44b6', '6bba']:
        rs = [r for r in R if r[0].startswith(emb)]
        print(emb, 'movies', len(rs), 'mean break-even TP/node %.4f' % np.mean([r[1] for r in rs]))
        for kind, B, lab in [('xy', XYB, 'um from xy border'), ('z', ZB, 'slices from z border')]:
            for i in range(len(B) - 1):
                n = tp = bew = 0.
                for name, be, cnt in rs:
                    for k, (nn, tt) in cnt.items():
                        bx, bz = map(int, k.split('_'))
                        if (bx if kind == 'xy' else bz) == i: n += nn; tp += tt; bew += nn * be
                print('   %s [%g,%g) %s: nodes %8d  TP/node %.4f  break-even %.4f  ratio %.2f' % (kind, B[i], B[i + 1], lab, n, tp / max(1, n), bew / max(1, n), (tp / max(1, n)) / max(1e-9, bew / max(1, n))))
