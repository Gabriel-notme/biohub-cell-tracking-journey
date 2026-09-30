import os, sys, json
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
import numpy as np
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
    matched_g = {int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't'])
    ea = gt.edge_attrs(); gs = defaultdict(list); gp = {}
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(s)].append(int(d)); gp[int(d)] = int(s)
    runs = []
    seen = set()
    for g in [int(i) for i in ga[K.NODE_ID].to_list()]:
        if g in matched_g or g in seen: continue
        # walk back to the start of this unmatched run
        s = g
        while gp.get(s) is not None and gp[s] not in matched_g: s = gp[s]
        # walk forward along single-child chain
        L = 0; c = s; cs = []
        while c is not None and c not in matched_g and c not in seen:
            seen.add(c); L += 1; cs.append(c)
            k = gs.get(c, []); c = k[0] if len(k) == 1 else None
        runs.append((L, gp.get(s) is not None, c is not None))
    # whole-track miss: components with zero matched
    return name, runs
if __name__ == '__main__':
    files = sorted(Path(sys.argv[1]).glob('*.json'))
    with Pool(len(files)) as pool: res = pool.map(job, [str(f) for f in files], chunksize=1)
    L = [r[0] for n, rr in res for r in rr]
    c = Counter(min(l, 11) for l in L)
    print('runs', len(L), 'nodes', sum(L)); print(sorted(c.items()))
    print('nodes in runs >=5:', sum(l for l in L if l >= 5), ' runs len1:', c[1])
    print('bounded both sides (matched before & after):', sum(1 for n, rr in res for r in rr if r[1] and r[2]), 'nodes', sum(r[0] for n, rr in res for r in rr if r[1] and r[2]))
