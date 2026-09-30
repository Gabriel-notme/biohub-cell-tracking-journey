import os, sys, json
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '4'
import numpy as np
import evalx
from tracking_cellmot.metrics import evaluate
K = evalx.K
S = np.array([1.625, .40625, .40625])
gdir = sys.argv[1]; cases = json.loads(sys.argv[2])
for name, g in cases:
    nodes, edges = evalx.load_graph_json(gdir + '/' + name + '.json')
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {v: k for k, v in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {i: (t, np.array([z, y, x]) * S) for i, t, z, y, x in zip(*[ga[c].to_list() for c in [K.NODE_ID, 't', 'z', 'y', 'x']])}
    ea = gt.edge_attrs(); gsucc = defaultdict(list); gpar = {}
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[s].append(d); gpar[d] = s
    succ = defaultdict(list); par = {}; ep = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s; ep[(s, d)] = e.get('edge_prob')
    ppos = lambda n: np.array([nodes[n]['z'], nodes[n]['y'], nodes[n]['x']]) * S
    print('=====', name, 'GT div', g, 't', gpos[g][0])
    # walk GT: mother chain back 2, and each daughter forward 4
    def show(gn, tag):
        t, P = gpos[gn]; pn = g2p.get(gn)
        s = '%s gt%d t%d' % (tag, gn, t)
        if pn is None:
            # nearest pred node in same frame
            best = min(((np.linalg.norm(ppos(n) - P), n) for n in nodes if nodes[n]['t'] == t), default=None)
            s += ' UNMATCHED nearest pred %s d=%.1f' % (best[1], best[0]) if best else ' UNMATCHED'
        else:
            s += ' -> p%d d=%.1f par=%s nsucc=%d ep_in=%s' % (pn, np.linalg.norm(ppos(pn) - P), par.get(pn), len(succ.get(pn, [])), ep.get((par.get(pn), pn)))
        print(s)
    c = gpar.get(g); chain = []
    for _ in range(2):
        if c is None: break
        chain.append(c); c = gpar.get(c)
    for x in reversed(chain): show(x, 'anc')
    show(g, 'MOTHER')
    for k, d in enumerate(gsucc[g]):
        c = d
        for h in range(5):
            show(c, 'd%d+%d' % (k, h + 1))
            nx = gsucc.get(c, [])
            if len(nx) != 1: break
            c = nx[0]
