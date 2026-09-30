import os, sys, json
from pathlib import Path
from collections import defaultdict, Counter
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
    gv = {int(i): np.array([z, y, x], float) * S for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    ea = gt.edge_attrs(); gs = defaultdict(list); gp = {}
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(s)].append(int(d)); gp[int(d)] = int(s)
    succ = defaultdict(list); par = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s
    pv = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    out = []
    seen = set()
    for g in gv:
        if g in g2p or g in seen: continue
        s = g
        while gp.get(s) is not None and gp[s] not in g2p: s = gp[s]
        run = []; c = s
        while c is not None and c not in g2p and c not in seen:
            seen.add(c); run.append(c); k = gs.get(c, []); c = k[0] if len(k) == 1 else None
        a = gp.get(s); b = c
        if a is None or b is None or a not in g2p or b not in g2p: continue
        pa, pb = g2p[a], g2p[b]
        # follow pred track from pa for len(run)+1 steps
        path = [pa]; cur = pa; ok = True
        for i in range(len(run) + 1):
            k = succ.get(cur, [])
            if len(k) == 0: ok = False; break
            # prefer child closest to GT
            cur = min(k, key=lambda q: np.linalg.norm(pv[q] - (gv[run[i]] if i < len(run) else gv[b])))
            path.append(cur)
        if ok and path[-1] == pb:
            cat = 'same_track'
            # interpolation test
            L = len(run) + 1
            fixed = 0
            for i, gnode in enumerate(run):
                w = (i + 1) / L; ip = pv[pa] * (1 - w) + pv[pb] * w
                fixed += np.linalg.norm(ip - gv[gnode]) <= 7
            devs = [float(np.linalg.norm(pv[path[i + 1]] - gv[run[i]])) for i in range(len(run))]
            out.append(dict(movie=name, cat=cat, L=len(run), interp_fix=int(fixed), dev=float(np.mean(devs))))
        elif not ok:
            out.append(dict(movie=name, cat='pa_track_ends', L=len(run), pb_start=par.get(pb) is None))
        else:
            out.append(dict(movie=name, cat='pa_goes_elsewhere', L=len(run), pb_start=par.get(pb) is None))
    return out
if __name__ == '__main__':
    import pandas as pd
    files = sorted(Path(sys.argv[1]).glob('*.json'))
    with Pool(len(files)) as pool: res = pool.map(job, [str(f) for f in files], chunksize=1)
    df = pd.DataFrame([r for a in res for r in a])
    print(df.groupby('cat').agg(runs=('L', 'size'), nodes=('L', 'sum')))
    s = df[df.cat == 'same_track']; print('same_track interp_fix nodes', s.interp_fix.sum(), 'of', s.L.sum(), 'mean dev', s.dev.mean())
    print(df[df.cat != 'same_track'].groupby(['cat', 'pb_start']).L.agg(['size', 'sum']))
