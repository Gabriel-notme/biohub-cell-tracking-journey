import os, sys, json, glob
import numpy as np, zarr
from multiprocessing import Pool
from collections import defaultdict
ROOT = '/workspace/data/train'
CZ, CY = 16, 32   # z slices at full res, y/x at half res
FR = (-1, 0, 1, 2)

class Vol:
    def __init__(self, name):
        g = zarr.open_group(ROOT + '/' + name + '.zarr', mode='r'); self.a = g['0']; self.T = self.a.shape[0]
        qs = g.attrs.get('image_statistics', {}).get('quantiles', {})
        self.lo = float(qs.get('0.001', 0)); self.hi = float(qs.get('0.999', 0))
        if self.hi <= self.lo:
            v = np.asarray(self.a[0])[::2, ::4, ::4]; self.lo, self.hi = np.percentile(v, [.1, 99.9])
        self.cache = {}
    def frame(self, t):
        t = int(np.clip(t, 0, self.T - 1))
        if t not in self.cache:
            if len(self.cache) > 8: self.cache.pop(next(iter(self.cache)))
            f = np.asarray(self.a[t], np.float32)[:, ::2, ::2]
            self.cache[t] = np.clip((f - self.lo) / (self.hi - self.lo + 1e-6), 0, 3)
        return self.cache[t]
    def crop(self, t, z, y, x):
        out = np.zeros((len(FR), CZ, CY, CY), np.float16)
        zc, yc, xc = int(round(z)), int(round(y / 2)), int(round(x / 2))
        for i, dt in enumerate(FR):
            f = self.frame(t + dt)
            zi = np.clip(np.arange(zc - CZ // 2, zc + CZ // 2), 0, f.shape[0] - 1)
            yi = np.clip(np.arange(yc - CY // 2, yc + CY // 2), 0, f.shape[1] - 1)
            xi = np.clip(np.arange(xc - CY // 2, xc + CY // 2), 0, f.shape[2] - 1)
            out[i] = f[np.ix_(zi, yi, xi)]
        return out

def gt(name):
    g = zarr.open_group(ROOT + '/' + name + '.geff', mode='r')
    ids = np.asarray(g['nodes/ids'][:]); P = {k: np.asarray(g['nodes/props/%s/values' % k][:]) for k in 'tzyx'}
    e = np.asarray(g['edges/ids'][:]) if 'edges/ids' in g else np.zeros((0, 2), int)
    succ = defaultdict(list); par = {}
    for s, d in e: succ[int(s)].append(int(d)); par[int(d)] = int(s)
    nodes = {int(n): (int(P['t'][i]), float(P['z'][i]), float(P['y'][i]), float(P['x'][i])) for i, n in enumerate(ids)}
    return nodes, succ, par

def job(args):
    name, neg_per_movie, seed = args
    rng = np.random.default_rng(seed)
    nodes, succ, par = gt(name)
    pos = [n for n in nodes if len(succ.get(n, [])) == 2]
    ignore = set()
    for p in pos:
        if p in par: ignore.add(par[p])
        ignore.update(succ[p])
    cand = [n for n in nodes if n not in pos and n not in ignore and len(succ.get(n, [])) == 1]
    near = [n for n in cand if any(abs(nodes[n][0] - nodes[p][0]) <= 3 for p in pos)]
    rest = [n for n in cand if n not in set(near)]
    k = min(len(rest), neg_per_movie)
    negs = list(rng.choice(rest, size=k, replace=False)) if k else []
    negs = negs + near[:60]
    V = Vol(name)
    sel = [(n, 1) for n in pos] + [(int(n), 0) for n in negs]
    sel.sort(key=lambda r: nodes[r[0]][0])
    X = np.stack([V.crop(*nodes[n]) for n, _ in sel]) if sel else np.zeros((0, len(FR), CZ, CY, CY), np.float16)
    y = np.array([l for _, l in sel], np.int8)
    return name, X, y, [n for n, _ in sel]

if __name__ == '__main__':
    names = open(sys.argv[1]).read().split(); outp = sys.argv[2]; negk = int(sys.argv[3])
    with Pool(int(sys.argv[4])) as pool:
        res = pool.map(job, [(n, negk, i) for i, n in enumerate(names)], chunksize=1)
    X = np.concatenate([r[1] for r in res]); y = np.concatenate([r[2] for r in res])
    mv = np.concatenate([[r[0]] * len(r[2]) for r in res]); nid = np.concatenate([r[3] for r in res])
    np.save(outp + '_X.npy', X); np.save(outp + '_y.npy', y); np.save(outp + '_movie.npy', mv); np.save(outp + '_nid.npy', nid)
    print('saved', X.shape, 'pos', int(y.sum()), flush=True)
