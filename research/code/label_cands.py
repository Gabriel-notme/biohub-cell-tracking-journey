import sys, json, glob
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
import numpy as np
import evalx
from tracking_cellmot.metrics import evaluate
K = evalx.K

def labels(name, nodes, edges):
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ea = gt.edge_attrs(); gs = defaultdict(list); gp = {}
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(s)].append(int(d)); gp[int(d)] = int(s)
    return p2g, gs, gp

gdir, cdir = sys.argv[1], sys.argv[2]
rows = []
for cp in sorted(glob.glob(cdir + '/*.cands.json')):
    name = Path(cp).name.split('.')[0]
    nodes, edges = evalx.load_graph_json(gdir + '/' + name + '.json')
    p2g, gs, gp = labels(name, nodes, edges)
    for c in json.loads(open(cp).read()):
        g = p2g.get(c['p'])
        lab = 'unl'
        if g is not None and gs.get(g):
            kids = set(gs[g]); ga, gb = p2g.get(c['a']), p2g.get(c['b'])
            if len(kids) == 2 and ga in kids and gb in kids: lab = 'pos'
            else: lab = 'neg'
        elif c.get('q') is not None and p2g.get(c['q']) is not None and gs.get(p2g[c['q']]):
            lab = 'q_gt'
        c['lab'] = lab; c['movie'] = name; rows.append(c)
print(len(rows), Counter((r['typ'], r['lab']) for r in rows))
for typ in ['start', 'stolen']:
    for lab in ['pos', 'neg']:
        v = np.array([r['fork'] for r in rows if r['typ'] == typ and r['lab'] == lab])
        if len(v): print(typ, lab, len(v), 'fork quantiles', np.round(np.quantile(v, [0, .25, .5, .75, .9, 1]), 3).tolist(), 'n>=.5', int((v >= .5).sum()), 'n>=.2', int((v >= .2).sum()), 'n>=.1', int((v >= .1).sum()))
for r in rows:
    if r['lab'] == 'pos': print('POS', r['movie'], r['typ'], round(r['fork'], 3), r['e_old'] if r['e_old'] is None else round(r['e_old'], 3), round(r['e_pb'], 3), round(r['d_pb'], 1), round(r['d_ab'], 1))
json.dump(rows, open(cdir + '/labeled.json', 'w'))
