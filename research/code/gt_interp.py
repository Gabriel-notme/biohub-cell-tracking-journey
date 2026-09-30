import sys, os, json
sys.path.insert(0, '/workspace/code'); os.environ['POLARS_MAX_THREADS'] = '2'
import numpy as np
from collections import defaultdict, Counter
from multiprocessing import Pool
def one(name):
    from evalx import load_gt, K
    gt, n_total = load_gt(name)
    na = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    pos = {int(i): np.array([t, z, y, x], float) for i, t, z, y, x in zip(*[na[c].to_list() for c in [K.NODE_ID, 't', 'z', 'y', 'x']])}
    ea = gt.edge_attrs(); succ = defaultdict(list); par = {}
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): succ[int(s)].append(int(d)); par[int(d)] = int(s)
    lin = 0; tot = 0; d2 = []; frac_int = []
    for n, p in pos.items():
        frac_int.append(np.all(np.abs(p[1:] - np.round(p[1:])) < 1e-6))
        a = par.get(n); kids = succ.get(n, [])
        if a is None or len(kids) != 1 or len(succ.get(a, [])) != 1: continue
        b = kids[0]
        dd = pos[a][1:] + pos[b][1:] - 2 * p[1:]
        d2.append(np.abs(dd).max()); tot += 1
        if np.abs(dd).max() < 1e-3: lin += 1
    return name, lin, tot, d2, float(np.mean(frac_int))
if __name__ == '__main__':
    names = sorted(p[:-5] for p in os.listdir('/workspace/data/train') if p.endswith('.geff'))
    with Pool(64) as pool: res = pool.map(one, names)
    L = sum(r[1] for r in res); T = sum(r[2] for r in res)
    D = np.concatenate([np.array(r[3]) for r in res if len(r[3])])
    print('exactly-linear interior nodes', L, 'of', T, L / T)
    print('second-diff max abs quantiles (vox)', np.round(np.quantile(D, [.05, .1, .25, .5, .75, .9]), 4).tolist())
    print('frac integer coords mean', np.mean([r[4] for r in res]))
    per = sorted([(r[1] / max(1, r[2]), r[0]) for r in res])
    print('per-movie linear frac quantiles', np.round(np.quantile([p[0] for p in per], [0, .1, .5, .9, 1]), 3).tolist())
    print(per[:3], per[-3:])
