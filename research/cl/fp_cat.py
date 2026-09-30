import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
import evalx
K = evalx.K
R = json.load(open('/workspace/cl/forks2_p3.json'))
G = {'hold36': '/workspace/runs/p3_hold36/graphs', 'prev4': '/workspace/runs/p3_prev4/graphs', 'audit32': '/workspace/sync3/runs/p3_audit32/graphs',
     't127a': '/workspace/sync4/runs/fin_p3_t127a/graphs', 't127b': '/workspace/sync3/runs/fin_p3_t127b/graphs'}
bym = defaultdict(list)
for r in R:
    if r['lab'] == 'FP': bym[(r['set'], r['movie'])].append(r['node'])
def job(key):
    s, name = key
    nodes, edges = evalx.load_graph_json(Path(G[s]) / (name + '.json'))
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    inv = {v: k for k, v in mapping.items()}
    from tracking_cellmot.metrics import evaluate
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ea = gt.edge_attrs(); gsucc = defaultdict(list); gpar = {}
    for a, b in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(a)].append(int(b)); gpar[int(b)] = int(a)
    gat = gt.node_attrs(attr_keys=[K.NODE_ID, 't']); gt_t = dict(zip(gat[K.NODE_ID].to_list(), gat['t'].to_list()))
    succ = defaultdict(list)
    for e in edges: succ[int(e['source_id'])].append(int(e['target_id']))
    out = []
    for p in bym[key]:
        g = p2g.get(p)
        rec = {'set': s, 'movie': name, 'node': p, 't': nodes[p]['t']}
        if g is None: rec['cat'] = 'p_unmatched'; out.append(rec); continue
        # walk GT lineage up/down to find divisions
        up = g; dt_up = None; k = 0
        while up in gpar and k < 60:
            up = gpar[up]; k += 1
            if len(gsucc[up]) == 2: dt_up = k; break
        dn = [(g, 0)]; dt_dn = None; seen = 0
        while dn and dt_dn is None and seen < 500:
            x, k = dn.pop(0); seen += 1
            if len(gsucc[x]) == 2: dt_dn = k; break
            for y in gsucc[x]: dn.append((y, k + 1))
        rec['gt_out'] = len(gsucc[g]); rec['dt_up'] = dt_up; rec['dt_dn'] = dt_dn
        kids = succ[p]; km = [p2g.get(c) for c in kids]
        rec['kids_matched'] = sum(x is not None for x in km)
        rec['kids_are_gt_succ'] = sum(x in gsucc[g] for x in km if x is not None)
        if dt_dn == 0: rec['cat'] = 'gt_div_here_but_wrong_kids'
        elif dt_dn is not None and dt_dn <= 3: rec['cat'] = 'gt_div_later_%d' % dt_dn
        elif dt_up is not None and dt_up <= 3: rec['cat'] = 'gt_div_earlier_%d' % dt_up
        elif rec['gt_out'] == 0: rec['cat'] = 'gt_track_ends'
        else: rec['cat'] = 'no_gt_div_near'
        out.append(rec)
    return out
with Pool(40) as pool:
    res = [r for rs in pool.map(job, list(bym)) for r in rs]
print(Counter(r['cat'] for r in res))
print(Counter((r['cat'], r.get('kids_matched'), r.get('kids_are_gt_succ')) for r in res))
json.dump(res, open('/workspace/cl/fp_forks_cat.json', 'w'))
