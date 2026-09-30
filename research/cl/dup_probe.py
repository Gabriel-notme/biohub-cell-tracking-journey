"""Near-duplicate predicted nodes (same frame, < R um): how many, and how often is each member GT-matched?"""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, .40625, .40625])
G = '/workspace/cl/ps_p5_hold36/graphs'


def job(name):
    import evalx
    K = evalx.K
    nodes, edges = evalx.load_graph_json(Path(G) / (name + '.json'))
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    from tracking_cellmot.metrics import evaluate
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    matched = {inv[int(x)] for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    succ = defaultdict(list); par = {}
    for e in edges: succ[int(e['source_id'])].append(int(e['target_id'])); par[int(e['target_id'])] = int(e['source_id'])
    # track length of the linear segment containing a node
    def seglen(n):
        k = 1; x = n
        while x in par and len(succ.get(par[x], [])) == 1: x = par[x]; k += 1
        x = n
        while len(succ.get(x, [])) == 1: x = succ[x][0]; k += 1
        return k
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    c = Counter()
    for t, ns in frames.items():
        P = np.array([[nodes[n][k] for k in 'zyx'] for n in ns]) * S
        tr = cKDTree(P)
        for R in (2.0, 3.0, 4.0):
            for i, j in tr.query_pairs(R):
                a, b = ns[i], ns[j]
                la, lb = seglen(a), seglen(b)
                short, long_ = (a, b) if la <= lb else (b, a)
                c[(R, 'pairs')] += 1
                c[(R, 'short_matched')] += int(short in matched); c[(R, 'long_matched')] += int(long_ in matched)
                c[(R, 'both_unmatched')] += int(short not in matched and long_ not in matched)
    c['nodes'] = len(nodes); c['n_total'] = n_total
    return c


if __name__ == '__main__':
    names = [l.strip() for l in open('/workspace/hold36.txt') if l.strip()]
    with Pool(36) as pool: cs = pool.map(job, names)
    tot = Counter()
    for c in cs: tot.update(c)
    for k, v in sorted(tot.items(), key=lambda kv: str(kv[0])): print(k, v)
