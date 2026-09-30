"""switch_both FN anatomy on P13: GT u->v, mu->x (x!=mv), y->mv (y!=mu). Distances d(mu,y), d(x,mv), is x / y matched,
and lengths of the side-by-side run of T1 and T2 (frames where the two tracks are mutual within 7um)."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']: os.environ[_k]='1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
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
    ea = gt.edge_attrs(); GE = list(zip([int(a) for a in ea[K.EDGE_SOURCE].to_list()], [int(b) for b in ea[K.EDGE_TARGET].to_list()]))
    gat = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x]) * S for i, z, y, x in zip(gat[K.NODE_ID].to_list(), gat['z'].to_list(), gat['y'].to_list(), gat['x'].to_list())}
    out = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
    pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) * S for n, v in nodes.items()}
    def run(a, b):
        # count frames backward/forward where tracks of a and b stay within 7um (a,b same frame)
        L = 1; p, q = a, b
        while p in par and q in par:
            p, q = par[p], par[q]
            if np.linalg.norm(pos[p] - pos[q]) >= 7: break
            L += 1
        p, q = a, b
        while len(out[p]) == 1 and len(out[q]) == 1:
            p, q = out[p][0], out[q][0]
            if np.linalg.norm(pos[p] - pos[q]) >= 7: break
            L += 1
        return L
    res = []
    for u, v in GE:
        mu, mv = g2p.get(u), g2p.get(v)
        if mu is None or mv is None or mv in out[mu]: continue
        if not (out[mu] and mv in par): continue
        x = out[mu][0] if len(out[mu]) == 1 else min(out[mu], key=lambda c: np.linalg.norm(pos[c] - pos[mv]))
        y = par[mv]
        res.append(dict(m=name, dmy=float(np.linalg.norm(pos[mu] - pos[y])), dxv=float(np.linalg.norm(pos[x] - pos[mv])),
                        xm=x in p2g, ym=y in p2g, du_mu=float(np.linalg.norm(pos[mu] - gpos[u])), du_y=float(np.linalg.norm(pos[y] - gpos[u])),
                        dv_mv=float(np.linalg.norm(pos[mv] - gpos[v])), dv_x=float(np.linalg.norm(pos[x] - gpos[v])), run=run(mu, y), fork=len(out[mu]) == 2))
    return res
if __name__ == '__main__':
    fs = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(48) as p: R = p.map(job, fs, chunksize=1)
    R = [r for rs in R for r in rs]
    json.dump(R, open('/workspace/cl/ideas/an/p13_flip.json', 'w'))
    print('n', len(R), 'fork', sum(r['fork'] for r in R))
    print('d(mu,y) bins', Counter(min(int(r['dmy']), 12) for r in R).most_common())
    print('x,y matched', Counter((r['xm'], r['ym']) for r in R))
    print('run len', Counter(min(r['run'], 20) // 2 * 2 for r in R).most_common())
    print('gap du_y - du_mu (how much closer y would be)', np.quantile([r['du_y'] - r['du_mu'] for r in R], [.1, .25, .5, .75, .9]).round(2))
    print('flip margin v: dv_x - dv_mv', np.quantile([r['dv_x'] - r['dv_mv'] for r in R], [.1, .25, .5, .75, .9]).round(2))
