import os, sys, json
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
import numpy as np
import evalx
from tracking_cellmot.metrics import evaluate
from concurrent.futures import ProcessPoolExecutor
K = evalx.K
S = np.array([1.625, .40625, .40625])
def one(p):
    name = Path(p).stem
    nodes, edges = evalx.load_graph_json(p)
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
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    arr = {t: (ns, np.array([[nodes[n]['z'], nodes[n]['y'], nodes[n]['x']] for n in ns]) * S) for t, ns in byt.items()}
    out = []
    # matched displacement stats too
    md = []
    for q, g in p2g.items():
        md.append((np.array([nodes[q]['z'], nodes[q]['y'], nodes[q]['x']]) * S - gpos[g]).tolist())
    for g in gpos:
        if g in g2p: continue
        t = gt_t[g]; ns, P = arr.get(t, ([], np.zeros((0, 3))))
        if not len(ns): continue
        dd = np.linalg.norm(P - gpos[g], axis=1); i = int(np.argmin(dd))
        v = (P[i] - gpos[g]).tolist()
        out.append(dict(movie=name, d=float(dd[i]), dz=v[0], dy=v[1], dx=v[2], near_matched=ns[i] in p2g))
    return out, md
if __name__ == '__main__':
    files = sorted(Path(sys.argv[1]).glob('*.json'))
    with ProcessPoolExecutor(36) as ex: res = list(ex.map(one, [str(f) for f in files]))
    rows = [r for a, b in res for r in a]; md = np.array([x for a, b in res for x in b])
    import pandas as pd
    df = pd.DataFrame(rows)
    print(len(df)); print(df[['d', 'dz', 'dy', 'dx']].abs().describe().round(2))
    print('near_matched', df.near_matched.mean())
    n = df[df.d < 12]
    print('d<12:', len(n), 'abs dz>5:', int((n.dz.abs() > 5).sum()), 'lateral>5:', int((np.hypot(n.dy, n.dx) > 5).sum()), 'near matched', n.near_matched.mean())
    print('matched displacement abs mean (z,y,x um)', np.round(np.abs(md).mean(0), 2).tolist(), 'mean signed', np.round(md.mean(0), 3).tolist(), 'q90', np.round(np.quantile(np.abs(md), .9, axis=0), 2).tolist())
    print(df.groupby(df.movie.str[:4]).size())
