import os, sys, json
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
import numpy as np
S = np.array([1.625, .40625, .40625])
def prof(img, z, y, x, r=3, zr=6):
    Z = img.shape[0]; yi, xi = int(round(y)), int(round(x))
    col = img[:, max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1].astype(np.float32).mean(axis=(1, 2))
    zi = int(round(z)); lo, hi = max(0, zi - zr), min(Z, zi + zr + 1)
    seg = col[lo:hi]; zs = np.arange(lo, hi)
    zmax = float(zs[np.argmax(seg)])
    w = np.clip(seg - np.percentile(col, 20), 0, None)
    zc = float((w * zs).sum() / w.sum()) if w.sum() > 0 else z
    return zmax, zc
def job(path):
    import evalx, zarr
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    name = Path(path).stem
    nodes, edges = evalx.load_graph_json(path)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: q for q, g in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gv = {int(i): np.array([z, y, x], float) for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    gt_t = {int(i): int(t) for i, t in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list())}
    arrz = zarr.open('/workspace/data/train/%s.zarr/0' % name, mode='r')
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    ppos = {n: np.array([v['z'], v['y'], v['x']]) for n, v in nodes.items()}
    rows = []
    rng = np.random.default_rng(0)
    pairs = []
    for g in gv:
        t = gt_t[g]
        if g in g2p: pairs.append(('m', g2p[g], g))
        else:
            ns = byt[t]; P = np.array([ppos[n] for n in ns])
            dd = np.linalg.norm((P - gv[g]) * S, axis=1); i = int(np.argmin(dd))
            if dd[i] < 12: pairs.append(('u', ns[i], g))
    cache = {}
    for kind, p, g in pairs:
        t = gt_t[g]
        if t not in cache: cache[t] = np.asarray(arrz[t])
        img = cache[t]
        z, y, x = ppos[p]
        zmax, zc = prof(img, z, y, x)
        rows.append(dict(movie=name, kind=kind, gz=gv[g][0], pz=z, zmax=zmax, zc=zc, dxy=float(np.linalg.norm((ppos[p][1:] - gv[g][1:]) * S[1:])), d=float(np.linalg.norm((ppos[p] - gv[g]) * S))))
    return rows
if __name__ == '__main__':
    import pandas as pd
    files = sorted(Path(sys.argv[1]).glob('*.json'))
    with Pool(len(files)) as pool: res = pool.map(job, [str(f) for f in files], chunksize=1)
    df = pd.DataFrame([r for a in res for r in a]); df.to_parquet('/workspace/runs/zref_hold36.parquet')
    for k in ['m', 'u']:
        s = df[df.kind == k]
        print(k, len(s), 'abs dz pred %.2f zmax %.2f zc %.2f (slices)' % ((s.pz - s.gz).abs().mean(), (s.zmax - s.gz).abs().mean(), (s.zc - s.gz).abs().mean()),
              'signed pred %.2f zmax %.2f zc %.2f' % ((s.pz - s.gz).mean(), (s.zmax - s.gz).mean(), (s.zc - s.gz).mean()))
        for col in ['pz', 'zmax', 'zc']:
            dd = np.sqrt(((s[col] - s.gz) * 1.625) ** 2 + s.dxy ** 2)
            print('   ', col, 'within7', float((dd <= 7).mean()))
