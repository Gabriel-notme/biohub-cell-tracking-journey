"""P21 probe: offsets between matched pred nodes and GT nodes (official 7 um matching) on P15 graphs, and for UNMATCHED GT nodes
the offset to the nearest pred node in the frame. Does the z offset depend on depth (z / Z), y/x position, or t? Per embryo."""
import os, sys, json, glob
for v in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'): os.environ[v] = '1'
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']

def job(a):
    s, f = a
    import evalx
    from scipy.spatial import cKDTree
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    name = os.path.basename(f)[:-5]
    nodes, edges = evalx.load_graph_json(f)
    g, mp = evalx.to_graph(nodes, edges, rounding=True)
    gt, _ = evalx.load_gt(name)
    evaluate(g, gt, scale=tuple(S), max_distance=7.0)
    na = g.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID, 't', 'z', 'y', 'x']).to_dicts()
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x']).to_dicts()
    gp = {r[K.NODE_ID]: (r['t'], np.array([r['z'], r['y'], r['x']], float)) for r in ga}
    matched = []
    mg = set()
    for r in na:
        m = r[K.MATCHED_NODE_ID]
        if m is None or int(m) == -1: continue
        m = int(m); mg.add(m)
        t, q = gp[m]; p = np.array([r['z'], r['y'], r['x']], float)
        matched.append([t, *p, *(q - p)])
    byt = {}
    for r in na: byt.setdefault(int(r['t']), []).append([r['z'], r['y'], r['x']])
    trees = {t: (cKDTree(np.array(v) * S), np.array(v, float)) for t, v in byt.items()}
    un = []
    for gid, (t, q) in gp.items():
        if gid in mg or t not in trees: continue
        tr, P = trees[t]; d, i = tr.query(q * S)
        un.append([t, *P[i], *(q - P[i]), d])
    return dict(movie=name, set=s, matched=matched, un=un)

if __name__ == '__main__':
    todo = [(s, f) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s))]
    with Pool(90) as p: res = p.map(job, todo, chunksize=1)
    json.dump(res, open('/workspace/cl/p21/zbias.json', 'w'))
    for emb in ('44b6', '6bba'):
        M = np.array([m for r in res if r['movie'].startswith(emb) for m in r['matched']]); U = np.array([u for r in res if r['movie'].startswith(emb) for u in r['un']])
        dz = M[:, 4] * 1.625; dy = M[:, 5] * .40625; dx = M[:, 6] * .40625
        print(emb, 'matched', len(M), 'mean offset um z %.3f y %.3f x %.3f | sd z %.2f y %.2f x %.2f | |d| median %.2f p90 %.2f' % (dz.mean(), dy.mean(), dx.mean(), dz.std(), dy.std(), dx.std(), np.median(np.sqrt(dz**2+dy**2+dx**2)), np.quantile(np.sqrt(dz**2+dy**2+dx**2), .9)))
        zq = np.quantile(M[:, 1], [0, .2, .4, .6, .8, 1.0])
        print('   by pred z (slice) quintile: ' + ' | '.join('z[%.0f,%.0f] dz %+.2f n%d' % (zq[i], zq[i+1], (M[(M[:,1]>=zq[i])&(M[:,1]<=zq[i+1]),4]*1.625).mean(), ((M[:,1]>=zq[i])&(M[:,1]<=zq[i+1])).sum()) for i in range(5)))
        print('   unmatched GT', len(U), 'nearest pred dist quantiles', np.round(np.quantile(U[:, 7], [.1, .25, .5, .75, .9]), 2).tolist(), 'frac 7-10um %.3f' % ((U[:,7]>7)&(U[:,7]<10)).mean(),
              '| among 7-10um: mean |dz| %.2f um vs |dyx| %.2f um' % (np.abs(U[(U[:,7]>7)&(U[:,7]<10),4]*1.625).mean(), np.sqrt((U[(U[:,7]>7)&(U[:,7]<10),5]*.40625)**2+(U[(U[:,7]>7)&(U[:,7]<10),6]*.40625)**2).mean()))
