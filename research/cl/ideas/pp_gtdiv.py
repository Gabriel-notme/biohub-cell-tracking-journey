import os, sys, glob, json
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
def job(f):
    import evalx
    import tracksdata as td
    K = td.DEFAULT_ATTR_KEYS
    name = f.split('/')[-1][:-5]
    gt, _ = evalx.load_gt(name)
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gp = {int(i): np.array([z, y, x]) * S for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    out = []
    for n in gt.node_ids():
        ch = gt.successors(n)
        if len(ch) >= 2:
            a, b = ch[:2]
            out.append((name[:4], float(np.linalg.norm(gp[a] - gp[n])), float(np.linalg.norm(gp[b] - gp[n])), float(np.linalg.norm(gp[a] - gp[b]))))
    return out
if __name__ == '__main__':
    fs = sorted(glob.glob('/workspace/data/train/*.geff'))
    with Pool(48) as p: R = [r for rs in p.map(job, fs) for r in rs]
    R = np.array([(r[1], r[2], r[3]) for r in R]); print('n', len(R))
    mx = R[:, :2].max(1); mn = R[:, :2].min(1)
    print('max(p-child) q', np.round(np.quantile(mx, [0, .5, .9, .95, .99, 1]), 1).tolist(), 'n>13', int((mx > 13).sum()), 'min>13', int((mn > 13).sum()))
    print('d_ab q', np.round(np.quantile(R[:, 2], [0, .5, .9, .99, 1]), 1).tolist(), 'n>20', int((R[:, 2] > 20).sum()))
