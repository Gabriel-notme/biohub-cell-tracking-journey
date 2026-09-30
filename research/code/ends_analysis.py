import sys, json, glob
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
import numpy as np, evalx
from scipy.spatial import cKDTree
from tracking_cellmot.metrics import evaluate
K = evalx.K
tot = Counter()
for path in sorted(glob.glob(sys.argv[1] + '/*.json')):
    name = Path(path).stem
    nodes, edges = evalx.load_graph_json(path)
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: p for p, g in p2g.items()}
    ea = gt.edge_attrs(); gsucc = defaultdict(list)
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(s)].append(int(d))
    succ = defaultdict(list); par = {}
    for e in edges: s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s
    pos = {n: np.array([v['z'] * 1.625, v['y'] * .40625, v['x'] * .40625]) for n, v in nodes.items()}
    T = max(int(v['t']) for v in nodes.values())
    frames = defaultdict(list)
    for n, v in nodes.items(): frames[int(v['t'])].append(n)
    for t in range(T):
        starts = [n for n in frames.get(t + 1, []) if n not in par]
        if not starts: continue
        tree = cKDTree(np.array([pos[n] for n in starts]))
        for p in frames[t]:
            if succ.get(p): continue
            near = [starts[j] for j in tree.query_ball_point(pos[p], 13.0)]
            k = 'ends_%d_starts' % min(len(near), 3)
            gp = p2g.get(p)
            lab = 'unl'
            if gp is not None and gsucc.get(gp):
                kids = [g2p.get(g) for g in gsucc[gp]]
                hit = [c for c in kids if c in near]
                lab = 'gt%d_hit%d' % (len(gsucc[gp]), len(hit))
            tot[(k, lab)] += 1
for k in sorted(tot): print(k, tot[k])
