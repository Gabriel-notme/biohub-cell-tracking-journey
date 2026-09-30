"""Division anatomy (analysis only): for each GT division (TP/FN) nearest pred fork in time/space; for each FP fork nearest GT division."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']: os.environ[_k]='1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
def job(f):
    import evalx
    from tracking_cellmot.metrics import evaluate
    from tracking_cellmot.division_metrics import score_divisions, _pred_division_fork_sets
    K = evalx.K
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    ds = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    ev, cc, mf = _pred_division_fork_sets(pred, gt, evalx.SCALE, 7.)
    ev = {inv[i] for i in ev}; cc = {inv[i] for i in cc}; mf = {inv[i] for i in mf}
    tpf = {inv[i] for i in ds.tp_forks}; fpf = {inv[i] for i in ds.fp_forks}
    pred2, _ = evalx.to_graph(nodes, edges)
    evaluate(pred2, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred2.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
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
    pf = [n for n in out if len(out[n]) == 2]
    gdivs = [g for g in gs if len(gs[g]) == 2]
    res = []
    for g in gdivs:
        t = gtt[g]; best = None
        for n in pf:
            dt = int(nodes[n]['t']) - t
            if abs(dt) > 4: continue
            d = float(np.linalg.norm(pos[n] - gpos[g]))
            if d > 15: continue
            if best is None or (abs(dt), d) < (abs(best[0]), best[1]): best = (dt, d, n)
        # daughters' matches
        ch = gs[g]; chm = [g2p.get(c) for c in ch]
        res.append(dict(kind='gtdiv', m=name, tp=bool(ds.scores.get(g)), near=None if best is None else [best[0], best[1], 'TP' if best[2] in tpf else ('FP' if best[2] in fpf else 'NE')],
                        gm=g2p.get(g), gm_out=len(out[g2p[g]]) if g in g2p else -1, chm=[c is not None for c in chm],
                        ch_same_parent=(chm[0] is not None and chm[1] is not None and par.get(chm[0]) == par.get(chm[1]) and par.get(chm[0]) is not None),
                        ch_dist=float(np.linalg.norm(gpos[ch[0]] - gpos[ch[1]])), t=t))
    for n in fpf:
        t = int(nodes[n]['t']); best = None
        for g in gdivs:
            dt = t - gtt[g]
            if abs(dt) > 4: continue
            d = float(np.linalg.norm(pos[n] - gpos[g]))
            if d > 15: continue
            if best is None or (abs(dt), d) < (abs(best[0]), best[1]): best = (dt, d)
        res.append(dict(kind='fpfork', m=name, near=best, ev=n in ev, cc=n in cc, mf=n in mf, t=t, matched=n in p2g))
    return res
if __name__ == '__main__':
    fs = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(60) as p: R = p.map(job, fs, chunksize=1)
    R = [r for rs in R for r in rs]
    json.dump(R, open('/workspace/cl/ideas/an/p13_div.json', 'w'))
    from collections import Counter
    G = [r for r in R if r['kind'] == 'gtdiv']; F = [r for r in R if r['kind'] == 'fpfork']
    print('GT divs', len(G), 'TP', sum(r['tp'] for r in G))
    print('FN near pred fork:', Counter((None if r['near'] is None else (r['near'][0], r['near'][2])) for r in G if not r['tp']).most_common())
    print('FN: g matched', Counter((r['gm'] is not None, r['gm_out'], tuple(r['chm']), r['ch_same_parent']) for r in G if not r['tp']).most_common())
    print('FP forks', len(F), Counter((r['ev'], r['cc'], r['mf'], r['near'] is not None and r['near'][0]) for r in F).most_common())
