import os, sys, json
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
import numpy as np
S = np.array([1.625, .40625, .40625])
def job(path):
    import evalx
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
    gpos = {int(i): np.array([z, y, x]) * S for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    gt_t = {int(i): int(t) for i, t in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list())}
    ea = gt.edge_attrs(); gs = defaultdict(list); gp = {}
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(s)].append(int(d)); gp[int(d)] = int(s)
    succ = defaultdict(list); par = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s
    ppos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    arr = {t: (ns, np.array([ppos[n] for n in ns])) for t, ns in byt.items()}
    allsteps = [float(np.linalg.norm(gpos[d] - gpos[s])) for s, ds in gs.items() for d in ds]
    rows = []
    for g in gpos:
        if g in g2p: continue
        t = gt_t[g]; ns, P = arr[t]
        dd = np.linalg.norm(P - gpos[g], axis=1); i = int(np.argmin(dd)); p = ns[i]
        r = dict(movie=name, d=float(dd[i]))
        a = gp.get(g)
        if a is not None:
            r['gt_in'] = float(np.linalg.norm(gpos[g] - gpos[a]))
            pa = g2p.get(a)
            if pa is not None: r['gprev_to_predprev'] = float(np.linalg.norm(gpos[a] - ppos[pa]))
        if par.get(p) is not None: r['pred_in'] = float(np.linalg.norm(ppos[p] - ppos[par[p]]))
        k = gs.get(g, [])
        if len(k) == 1:
            r['gt_out'] = float(np.linalg.norm(gpos[k[0]] - gpos[g]))
            if a is not None: r['gt_d2'] = float(np.linalg.norm(gpos[k[0]] + gpos[a] - 2 * gpos[g]))
        if len(succ.get(p, [])) == 1 and par.get(p) is not None:
            r['pred_d2'] = float(np.linalg.norm(ppos[succ[p][0]] + ppos[par[p]] - 2 * ppos[p]))
        rows.append(r)
    return rows, allsteps
if __name__ == '__main__':
    import pandas as pd
    files = sorted(Path(sys.argv[1]).glob('*.json'))
    with Pool(len(files)) as pool: res = pool.map(job, [str(f) for f in files], chunksize=1)
    df = pd.DataFrame([r for a, b in res for r in a]); st = np.concatenate([b for a, b in res])
    print('all GT steps quantiles', np.round(np.quantile(st, [.25, .5, .75, .9, .95]), 2).tolist())
    print(df.describe().round(2).to_string())
