import json, glob, sys, zarr, numpy as np
from pathlib import Path
names = [l.strip() for l in open('/workspace/hold36.txt') if l.strip()]
G = '/workspace/runs/b5f_hold36/working/lineage_graphs'
tb = np.zeros((10, 2)); zb = np.zeros((8, 2)); yb = np.zeros((8,2)); xb=np.zeros((8,2))
per = []
for n in names:
    g = zarr.open_group('/workspace/data/train/%s.geff' % n, mode='r')
    gt = {k: np.asarray(g['nodes/props/%s/values' % k][:]) for k in 'tzyx'}
    d = json.load(open('%s/%s.json' % (G, n)))
    P = np.array([[v['t'], v['z'], v['y'], v['x']] for v in d['nodes'].values()], float)
    T = P[:, 0].max() + 1
    sh = zarr.open_group('/workspace/data/train/%s.zarr' % n, mode='r')['0'].shape
    for arr, key in ((tb, 0), (zb, 1), (yb, 2), (xb, 3)):
        nb = arr.shape[0]; lim = [T, sh[1], sh[2], sh[3]][key]
        gi = np.clip((gt['tzyx'[key]] / lim * nb).astype(int), 0, nb - 1); pi = np.clip((P[:, key] / lim * nb).astype(int), 0, nb - 1)
        arr[:, 0] += np.bincount(gi, minlength=nb); arr[:, 1] += np.bincount(pi, minlength=nb)
    per.append((n, len(gt['t']), len(P), int(T), sh, float(gt['t'].min()), float(gt['t'].max())))
for nm, arr in (('t', tb), ('z', zb), ('y', yb), ('x', xb)):
    r = arr[:, 0] / arr[:, 1]; r = r / (arr[:, 0].sum() / arr[:, 1].sum())
    print(nm, ' '.join('%.2f' % v for v in r), ' predfrac', ' '.join('%.2f' % v for v in arr[:, 1] / arr[:, 1].sum()))
for p in per[:12]: print(p)
