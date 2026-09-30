"""Residual anatomy: for every GT node, nearest pred node (rounded coords as metric sees them), residual vector, pred velocity, spike."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']: os.environ[_k]='1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
def job(f):
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {v: k for k, v in p2g.items()}
    ea = gt.edge_attrs(); gs = defaultdict(list); gp = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(x)].append(int(y)); gp[int(y)] = int(x)
    gat = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x]) * S for i, z, y, x in zip(gat[K.NODE_ID].to_list(), gat['z'].to_list(), gat['y'].to_list(), gat['x'].to_list())}
    gtt = dict(zip([int(i) for i in gat[K.NODE_ID].to_list()], gat['t'].to_list()))
    out = defaultdict(list); par = {}
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id']); out[x].append(y); par[y] = x
    pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) * S for n, v in nodes.items()}
    fpos = {n: np.array([v[k] for k in 'zyx']) * S for n, v in nodes.items()}
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    trees = {t: (cKDTree(np.stack([pos[n] for n in ns])), ns) for t, ns in byt.items()}
    res = []
    for g, gpo in gpos.items():
        t = int(gtt[g]); tr, ns = trees[t]
        dd, ii = tr.query(gpo, k=2); n = ns[ii[0]]
        m = g2p.get(g)
        # velocity of pred node n
        pv = None; spike = None
        if n in par and len(out[n]) == 1:
            a, b = pos[par[n]], pos[out[n][0]]; pv = ((b - a) / 2).tolist(); spike = float(np.linalg.norm(pos[n] - (a + b) / 2))
            spike_ab = float(np.linalg.norm(b - a))
        else: spike_ab = None
        gv = None; gspike = None
        if g in gp and len(gs.get(g, [])) == 1:
            a, b = gpos[gp[g]], gpos[gs[g][0]]; gv = ((b - a) / 2).tolist(); gspike = float(np.linalg.norm(gpo - (a + b) / 2))
        res.append(dict(m=name, t=t, g=g, n=n, d=float(dd[0]), d2=float(dd[1]), matched=m is not None, m_is_n=(m == n),
                        r=(pos[n] - gpo).tolist(), rf=(fpos[n] - gpo).tolist(), gz=float(gpo[0]), pv=pv, spike=spike, spike_ab=spike_ab, gv=gv, gspike=gspike,
                        n_taken=(n in p2g and p2g[n] != g), npar=n in par, nout=len(out[n]), gin=g in gp, gout=len(gs.get(g, []))))
    return res
if __name__ == '__main__':
    fs = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(60) as p: R = p.map(job, fs, chunksize=1)
    json.dump([r for rs in R for r in rs], open('/workspace/cl/ideas/an/p13_resid.json', 'w'))
    print('done')
