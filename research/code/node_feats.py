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
    from scipy.spatial import cKDTree
    K = evalx.K
    name = Path(path).stem
    nodes, edges = evalx.load_graph_json(path)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    matched = {inv[int(a)] for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    out = defaultdict(list); par = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); out[s].append(d); par[d] = s
    adj = defaultdict(list)
    for s, ds in out.items():
        for d in ds: adj[s].append(d); adj[d].append(s)
    comp = {}; cid = 0
    for n in nodes:
        if n in comp: continue
        st = [n]; comp[n] = cid; mem = []
        while st:
            x = st.pop(); mem.append(x)
            for y in adj.get(x, []):
                if y not in comp: comp[y] = cid; st.append(y)
        cid += 1
    csize = defaultdict(int)
    for n, c in comp.items(): csize[c] += 1
    ids = list(nodes)
    T = np.array([nodes[n]['t'] for n in ids]); P = np.array([[nodes[n]['z'], nodes[n]['y'], nodes[n]['x']] for n in ids])
    dens = np.zeros(len(ids)); nn1 = np.zeros(len(ids))
    for t in np.unique(T):
        m = np.where(T == t)[0]; tree = cKDTree(P[m] * S)
        cnt = tree.query_ball_point(P[m] * S, r=10., return_length=True); dens[m] = cnt - 1
        dd, _ = tree.query(P[m] * S, k=2); nn1[m] = dd[:, 1]
    rows = dict(movie=[name] * len(ids), node=ids, t=T.tolist(), z=P[:, 0].tolist(), y=P[:, 1].tolist(), x=P[:, 2].tolist(),
                matched=[n in matched for n in ids], clen=[csize[comp[n]] for n in ids], nin=[1 if n in par else 0 for n in ids], nout=[len(out.get(n, [])) for n in ids],
                dens=dens.tolist(), nn1=nn1.tolist())
    return rows, dict(movie=name, n_total=n_total, n_pred=len(ids), n_gt=gt.num_nodes())
if __name__ == '__main__':
    import pandas as pd
    files = sorted(Path(sys.argv[1]).glob('*.json'))
    with Pool(len(files)) as pool: res = pool.map(job, [str(f) for f in files], chunksize=1)
    df = pd.concat([pd.DataFrame(r) for r, m in res]); df.to_parquet(sys.argv[2])
    json.dump([m for r, m in res], open(sys.argv[2] + '.meta.json', 'w'))
    print(len(df), df.matched.mean())
