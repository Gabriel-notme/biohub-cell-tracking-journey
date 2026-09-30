import os, sys, json, glob
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
import numpy as np, torch, evalx
from tracking_cellmot.metrics import evaluate
from divnet_data import Vol
from divnet_train import DivNet
K = evalx.K
dev = 'cuda'
nets = []
for m in ['/workspace/divnet/divnet_s0.pt', '/workspace/divnet/divnet_s1.pt']:
    ck = torch.load(m, map_location='cpu'); n = DivNet(**ck['config']); n.load_state_dict(ck['state_dict']); nets.append(n.to(dev).eval())
rows = []
for path in sorted(glob.glob(sys.argv[1] + '/*.json')):
    name = Path(path).stem
    nodes, edges = evalx.load_graph_json(path)
    succ = defaultdict(list)
    for e in edges: succ[int(e['source_id'])].append(int(e['target_id']))
    forks = [p for p, k in succ.items() if len(k) == 2]
    if not forks: continue
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ea = gt.edge_attrs(); gs = defaultdict(list)
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(s)].append(int(d))
    V = Vol(name)
    X = np.stack([V.crop(int(nodes[p]['t']), nodes[p]['z'], nodes[p]['y'], nodes[p]['x']) for p in forks])
    x = torch.from_numpy(X.astype(np.float32)).to(dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        lg = sum(n(x).float() + n(x.flip(3)).float() + n(x.flip(4)).float() + n(torch.rot90(x, 1, (3, 4))).float() for n in nets) / 8
    pr = torch.sigmoid(lg).cpu().numpy()
    for p, v in zip(forks, pr):
        g = p2g.get(p); lab = 'unl'
        if g is not None and gs.get(g): lab = 'gt%d' % len(gs[g])
        rows.append((name, p, lab, float(v), bool(succ[p][0] in [] )))
lab = defaultdict(list)
for r in rows: lab[r[2]].append(r[3])
for k, v in lab.items(): print(k, len(v), 'dn quantiles', np.round(np.quantile(v, [0, .1, .25, .5, .75, 1]), 3).tolist(), 'n<0.1', int((np.array(v) < .1).sum()), 'n<0.3', int((np.array(v) < .3).sum()))
json.dump(rows, open(sys.argv[2], 'w'))
