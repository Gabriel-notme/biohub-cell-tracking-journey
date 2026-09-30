import os, sys, json, glob, importlib
for _k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']: os.environ[_k]='1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from multiprocessing import Pool
from collections import Counter, defaultdict
import numpy as np
V = json.loads(sys.argv[1])
S = np.array([1.625, 0.40625, 0.40625])
def job(f):
    import evalx
    m = importlib.import_module('ideas.term_trim')
    name = Path(f).stem; st = f.split('ps_p13_')[1].split('/')[0]
    nodes, edges = evalx.load_graph_json(f)
    res = [evalx.score_movie(name, nodes, edges)]
    offs = []
    for kw in V:
        nn, ne, stt = m.apply(nodes, edges, **kw); res.append(evalx.score_movie(name, nn, ne))
    # offsets of trimmed nodes (first iteration) for first variant
    nn, ne, _ = m.apply(nodes, edges, **dict(V[0], iters=1))
    gone = set(nodes) - set(nn)
    from scipy.spatial import cKDTree
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) for n, v in nodes.items()}
    for g in gone:
        t = int(nodes[g]['t']); best = None
        for n in byt[t]:
            if n == g: continue
            d = np.linalg.norm((pos[n] - pos[g]) * S)
            if best is None or d < best[0]: best = (d, pos[n] - pos[g])
        offs.append((int(abs(best[1][0])), float(np.linalg.norm(best[1][1:] * S[1:]))))
    return name, st, res, offs
if __name__ == '__main__':
    fs = [f for s in ['hold36', 'prev4', 'audit32', 't127a', 't127b'] for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(48) as p: R = p.map(job, fs, chunksize=1)
    for i, kw in enumerate(V, 1):
        print(kw)
        for grp, fn in [('44b6', lambda n, s: n.startswith('44b6')), ('6bba', lambda n, s: n.startswith('6bba')), ('clean40', lambda n, s: s in ('hold36', 'prev4')), ('insample', lambda n, s: s not in ('hold36', 'prev4'))]:
            d = Counter()
            up = dn = 0
            for n, s, res, _ in R:
                if not fn(n, s): continue
                for k in ['edge_tp', 'edge_fp', 'edge_fn', 'num_pred_nodes']: d[k] += res[i][k] - res[0][k]
                x = res[i]['adj_edge_jaccard'] - res[0]['adj_edge_jaccard']
                up += x > 1e-9; dn += x < -1e-9
            print('   %-8s dTP %+d dFP %+d dFN %+d dNodes %+d | movies up %d down %d' % (grp, d['edge_tp'], d['edge_fp'], d['edge_fn'], d['num_pred_nodes'], up, dn))
    offs = [o for _, _, _, oo in R for o in oo]
    print('trimmed-node offset to nearest same-frame node: |dz| slices', Counter(o[0] for o in offs).most_common(), 'dxy<1um frac %.2f' % np.mean([o[1] < 1 for o in offs]))
    print('dz=2 with dxy<1:', sum(1 for o in offs if o[0] == 2 and o[1] < 1), 'of', len(offs))
