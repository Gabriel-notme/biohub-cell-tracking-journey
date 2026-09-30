"""Node-substitution candidates with image features (intensity / centredness of n and c) + labels.
usage: nswap_img.py <tag> <graph_dir> <full_dir> [label|nolabel]  -> /workspace/cl/ns/<tag>_img.npz"""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, 0.40625, 0.40625])
IMG = ['I_n', 'I_c', 'off_n', 'off_c', 'offz_n', 'offz_c', 'pk_n', 'pk_c', 'I_ratio', 'off_diff']


class Vol:
    def __init__(self, path):
        import zarr
        g = zarr.open_group(str(path), mode='r'); self.a = g['0']; self.shape = self.a.shape
        qs = g.attrs.get('image_statistics', {}).get('quantiles', {})
        self.lo = float(qs.get('0.001', 0)); self.hi = float(qs.get('0.999', 1))
        if self.hi <= self.lo:
            v = np.asarray(self.a[0])[::2, ::4, ::4]; self.lo, self.hi = np.percentile(v, [.1, 99.9])
        self.cache = {}

    def frame(self, t):
        if t not in self.cache:
            if len(self.cache) > 3: self.cache.pop(next(iter(self.cache)))
            self.cache[t] = (np.asarray(self.a[t], np.float32) - self.lo) / (self.hi - self.lo + 1e-6)
        return self.cache[t]

    def feats(self, t, p):
        F = self.frame(t); Z, Y, X = F.shape
        c = np.array([int(round(p[0])), int(round(p[1])), int(round(p[2]))])
        c = np.clip(c, 0, [Z - 1, Y - 1, X - 1])

        def box(cc, rz, ry):
            z0, z1 = max(0, cc[0] - rz), min(Z, cc[0] + rz + 1); y0, y1 = max(0, cc[1] - ry), min(Y, cc[1] + ry + 1); x0, x1 = max(0, cc[2] - ry), min(X, cc[2] + ry + 1)
            return F[z0:z1, y0:y1, x0:x1], (z0, y0, x0)
        b, _ = box(c, 1, 3); I = float(b.mean())
        w, o = box(c, 3, 12)
        ww = np.clip(w - np.median(w), 0, None)
        if ww.sum() > 0:
            zz, yy, xx = np.indices(w.shape)
            cen = np.array([(zz * ww).sum(), (yy * ww).sum(), (xx * ww).sum()]) / ww.sum() + np.array(o)
            d = (cen - c) * S; off = float(np.linalg.norm(d)); offz = float(abs(d[0]))
        else:
            off = offz = -1.
        # peakness: centre box mean vs boxes shifted by ~2.4um along each axis
        sh = [(2, 0, 0), (-2, 0, 0), (0, 6, 0), (0, -6, 0), (0, 0, 6), (0, 0, -6)]
        nb = [box(c + np.array(s), 1, 3)[0].mean() for s in sh]
        pk = float(np.mean([I > v for v in nb]))
        return I, off, offz, pk


def job(args):
    tag, f, fdir, label = args
    name = Path(f).stem
    fp = Path(fdir) / (name + '.geff'); zp = Path('/workspace/data/train/%s.zarr' % name)
    if not fp.exists() or not zp.exists(): return None
    if label and not Path('/workspace/data/train/%s.geff/nodes/props' % name).exists(): return None
    import evalx, nswap
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    full = nswap.load_full(fp)
    rows = nswap.candidates(nodes, edges, full)
    fids, fT, fV, fE, fprob = full
    labs = [0.] * len(rows)
    if label:
        from tracking_cellmot.metrics import evaluate
        gt, _ = evalx.load_gt(name)
        pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
        evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
        na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
        p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
        matched_g = set(p2g.values())
        ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
        gid = np.array([int(x) for x in ga[K.NODE_ID].to_list()]); gT = np.array(ga['t'].to_list()); gP = np.stack([ga[k].to_numpy() for k in 'zyx'], 1) * S
        gidx = {g: i for i, g in enumerate(gid.tolist())}
        ea = gt.edge_attrs(); deg = defaultdict(int)
        for a, b in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): deg[int(a)] += 1; deg[int(b)] += 1
        gtree = {t: (np.flatnonzero(gT == t), cKDTree(gP[gT == t])) for t in np.unique(gT)}
        for i, (n, ci, x) in enumerate(rows):
            t = int(nodes[n]['t']); pc = np.array([max(0, int(round(v))) for v in fV[ci]], float) * S
            gn = p2g.get(n); lab = 0.
            if gn is not None:
                if np.linalg.norm(pc - gP[gidx[gn]]) > 7: lab = -max(1, deg[gn])
            elif t in gtree:
                ix, tr = gtree[t]
                for j in tr.query_ball_point(pc, 7.0):
                    g = int(gid[ix[j]])
                    if g not in matched_g and deg[g] > 0: lab = max(lab, deg[g])
            labs[i] = lab
    vol = Vol(zp)
    order = sorted(range(len(rows)), key=lambda i: int(nodes[rows[i][0]]['t']))
    out = [None] * len(rows)
    for i in order:
        n, ci, x = rows[i]; t = int(nodes[n]['t'])
        In, on, ozn, pn = vol.feats(t, [nodes[n]['z'], nodes[n]['y'], nodes[n]['x']])
        Ic, oc, ozc, pc_ = vol.feats(t, fV[ci])
        out[i] = x + [In, Ic, on, oc, ozn, ozc, pn, pc_, Ic / (In + 1e-3), on - oc] + [labs[i]]
    return tag, name, out, [(int(n), int(ci)) for n, ci, _ in rows]


if __name__ == '__main__':
    tag, gdir, fdir = sys.argv[1], sys.argv[2], sys.argv[3]
    label = (sys.argv[4] if len(sys.argv) > 4 else 'label') == 'label'
    jobs = [(tag, f, fdir, label) for f in sorted(glob.glob(gdir + '/*.json'))]
    with Pool(int(os.environ.get('NPROC', '48'))) as p: R = [r for r in p.map(job, jobs) if r]
    X = np.array([row for _, _, rs, _ in R for row in rs], float)
    mv = np.array([m for _, m, rs, _ in R for _ in rs])
    Path('/workspace/cl/ns').mkdir(exist_ok=True)
    np.savez_compressed('/workspace/cl/ns/%s_img.npz' % tag, X=X, movie=mv)
    y = X[:, -1]
    print(tag, 'movies', len(R), 'rows', len(X), 'pos', int((y > 0).sum()), 'neg', int((y < 0).sum()))
    if label:
        def auc(p, yy):
            o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(len(p)); n1 = yy.sum(); n0 = len(yy) - n1
            return (r[yy == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0)
        e = y != 0; yy = (y[e] > 0).astype(int); base = X.shape[1] - 1 - len(IMG)
        for k, nm in enumerate(IMG):
            v = X[e, base + k]; print('  %-9s AUC %.3f  median pos %.3f neg %.3f' % (nm, auc(v, yy), np.median(v[yy == 1]), np.median(v[yy == 0])))
