import os, sys, json
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
import numpy as np

def job(path):
    import evalx
    import tracking_cellmot.division_metrics as DM
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    name = Path(path).stem
    nodes, edges = evalx.load_graph_json(path)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    inv = {v: k for k, v in mapping.items()}
    res = DM.score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    tp = {inv[f] for f in res.tp_forks}; fp = {inv[f] for f in res.fp_forks}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: p for p, g in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't'])
    gt_t = dict(zip(ga[K.NODE_ID].to_list(), ga['t'].to_list()))
    ea = gt.edge_attrs()
    gsucc = defaultdict(list); gpar = {}
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()):
        gsucc[s].append(d); gpar[d] = s
    succ = defaultdict(list); par = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); par[d] = s
    pforks = {n for n in nodes if len(succ.get(n, [])) >= 2}
    out = []
    for g, sc in res.scores.items():
        rec = dict(movie=name, kind='TP' if sc else 'FN', g=int(g), t=int(gt_t[g]), mother_matched=g in g2p)
        cand = set(); c = g; h = 0
        while c is not None and h <= 4:
            if c in g2p: cand.add(g2p[c])
            c = gpar.get(c); h += 1
        fr = [(d, 1) for d in gsucc[g]]
        while fr:
            c, h = fr.pop()
            if c in g2p: cand.add(g2p[c])
            if h < 4: fr += [(d, h + 1) for d in gsucc.get(c, [])]
        near = []
        for pn in cand:
            c = pn; h = 0
            while c is not None and h <= 6:
                if c in pforks: near.append((int(nodes[c]['t']) - rec['t'], 'tp' if c in tp else ('fp' if c in fp else 'unl'), int(c)))
                c = par.get(c); h += 1
        rec['near_forks'] = sorted(set(near))
        rec['daughters_matched'] = [d in g2p for d in gsucc[g]]
        rec['gt_daughter_len'] = []
        for d in gsucc[g]:
            L = 0; c = d
            while c is not None and L < 50:
                L += 1; nx = gsucc.get(c, []); c = nx[0] if len(nx) == 1 else None
            rec['gt_daughter_len'].append(L)
        out.append(rec)
    for f in fp:
        rec = dict(movie=name, kind='FP', p=int(f), t=int(nodes[f]['t']), matched=f in p2g)
        near = []
        gm = p2g.get(f)
        if gm is not None:
            c = gm; h = 0
            while c is not None and h <= 6:
                if len(gsucc.get(c, [])) >= 2: near.append(int(gt_t[c]) - rec['t'])
                c = gpar.get(c); h += 1
            fr = [(d, 1) for d in gsucc.get(gm, [])]
            while fr:
                c, h = fr.pop()
                if len(gsucc.get(c, [])) >= 2: near.append(int(gt_t[c]) - rec['t'])
                if h < 6: fr += [(d, h + 1) for d in gsucc.get(c, [])]
        rec['near_gt_div'] = sorted(set(near))
        rec['children_matched'] = [c in p2g for c in succ[f]]
        out.append(rec)
    return out

if __name__ == '__main__':
    import glob
    ps = sorted(glob.glob(sys.argv[1] + '/*.json'))
    with Pool(32) as pool: rr = pool.map(job, ps)
    rows = [r for x in rr for r in x]
    json.dump(rows, open(sys.argv[2], 'w'))
    for k in ['TP', 'FN', 'FP']:
        R = [r for r in rows if r['kind'] == k]
        print(k, len(R))
    for r in rows:
        if r['kind'] == 'FN':
            print('FN', r['movie'], 't', r['t'], 'mm', r['mother_matched'], 'dm', r['daughters_matched'], 'dlen', r['gt_daughter_len'], 'near', [(a, b) for a, b, c in r['near_forks']])
    for r in rows:
        if r['kind'] == 'FP':
            print('FP', r['movie'], 't', r['t'], 'matched', r['matched'], 'near_gt_div', r['near_gt_div'], 'cm', r['children_matched'])
