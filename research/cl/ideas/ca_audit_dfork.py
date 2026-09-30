"""Audit: double forks remaining in the final P13 graphs and which stage created the path between them."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])


def gt_divs(name):
    import zarr
    g = zarr.open_group('/workspace/data/train/%s.geff' % name, mode='r')
    ids = np.asarray(g['nodes/ids'][:]).astype(int)
    T = np.asarray(g['nodes/props/t/values'][:]).astype(int)
    V = np.stack([np.asarray(g['nodes/props/%s/values' % k][:]) for k in 'zyx'], 1) * S
    E = np.asarray(g['edges/ids'][:]).astype(int)
    outd = Counter(E[:, 0].tolist())
    div = [i for i in range(len(ids)) if outd.get(int(ids[i]), 0) >= 2]
    return [(int(T[i]), V[i]) for i in div], (ids, T, V, outd)


def src(e):
    for k in ('div_complete', 'relink', 'edge_link'):
        if k in e: return k
    return 'b5'


def job(f):
    name = os.path.basename(f)[:-5]
    d = json.load(open(f))
    nodes = {int(k): v for k, v in d['nodes'].items()}
    succ = defaultdict(list); par = {}; eat = {}
    for e in d['edges']:
        s, t = int(e['source_id']), int(e['target_id']); succ[s].append(t); par[t] = s; eat[(s, t)] = src(e)
    forks = [n for n in nodes if len(succ[n]) >= 2]
    fs = set(forks)
    gdiv, _ = gt_divs(name)

    def is_gt(n):
        t = int(nodes[n]['t']); p = np.array([nodes[n][k] for k in 'zyx']) * S
        return any(abs(gt - t) <= 1 and np.linalg.norm(gp - p) <= 7. for gt, gp in gdiv)
    rows = []
    for f0 in forks:
        # walk down all descendants, record descendant forks and path sources
        st = [(c, 1, {eat[(f0, c)]}) for c in succ[f0]]
        while st:
            x, dd, ss = st.pop()
            if x in fs:
                rows.append(dict(movie=name, f=f0, g=x, dt=dd, srcs=sorted(ss), f_src=sorted({eat[(f0, c)] for c in succ[f0]}),
                                 g_src=sorted({eat[(x, c)] for c in succ[x]}), f_gt=is_gt(f0), g_gt=is_gt(x)))
            for y in succ.get(x, []):
                st.append((y, dd + 1, ss | {eat[(x, y)]}))
    return dict(movie=name, nforks=len(forks), rows=rows)


if __name__ == '__main__':
    files = sorted(glob.glob('/workspace/cl/ps_p13_*/graphs/*.json'))
    with Pool(48) as p: R = p.map(job, files, chunksize=1)
    rows = [r for x in R for r in x['rows']]
    print('movies', len(R), 'forks', sum(x['nforks'] for x in R), 'double-fork pairs', len(rows))
    for r in sorted(rows, key=lambda r: r['dt']):
        print(r['movie'], 'dt', r['dt'], 'path', r['srcs'], 'f', r['f_src'], r['f_gt'], 'g', r['g_src'], r['g_gt'])
