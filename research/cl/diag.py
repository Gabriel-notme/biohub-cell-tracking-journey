"""diagnostic only (reads GT to explain results; never used inside the rule): for each candidate class of p19_edge, how many of the
candidate nodes are matched to a GT node, and how many of the candidate edges are TP / FP (evaluable) / neutral (unannotated)."""
import sys, json, glob, os
from collections import Counter, defaultdict
from multiprocessing import Pool
os.environ.setdefault('POLARS_MAX_THREADS', '1')
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
VARIANTS = json.loads(sys.argv[1])


def one(f):
    import warnings; warnings.filterwarnings('ignore')
    import evalx
    import tracksdata as td
    from tracking_cellmot.metrics import evaluate
    from ideas import p19_edge as m
    K = td.DEFAULT_ATTR_KEYS
    name = os.path.basename(f)[:-5]
    nodes, edges = evalx.load_graph_json(f)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    gids = gt.node_ids(); outd = dict(zip(gids, gt.out_degree(gids))); ind = dict(zip(gids, gt.in_degree(gids)))
    gset = set()
    ea = gt.edge_attrs()
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gset.add((int(s), int(d)))
    shape = m._shape('/workspace/data/train/%s.zarr' % name)
    res = []
    for vi, kw in enumerate(VARIANTS):
        drop, st = m.select(nodes, edges, shape=shape, **kw)
        c = Counter(); c['nodes'] = len(drop); c['matched'] = sum(1 for n in drop if n in p2g)
        for e in edges:
            a, b = int(e['source_id']), int(e['target_id'])
            if a not in drop and b not in drop: continue
            ga, gb = p2g.get(a), p2g.get(b)
            if ga is not None and gb is not None and (ga, gb) in gset: c['tp'] += 1
            elif (ga is not None and outd.get(ga, 0) > 0) or (gb is not None and ind.get(gb, 0) > 0): c['fp'] += 1
            else: c['neutral'] += 1
        res.append((name, vi, dict(c)))
    return res


if __name__ == '__main__':
    fs = sorted(glob.glob('/workspace/cl/p16/ps_p17_*/graphs/*.json'))
    with Pool(int(os.environ.get('RULE_POOL', '8')), maxtasksperchild=4) as p: R = [r for rs in p.map(one, fs, chunksize=1) for r in rs]
    for vi, kw in enumerate(VARIANTS):
        for emb in ['44b6', '6bba']:
            c = Counter()
            for name, v, d in R:
                if v == vi and name.startswith(emb): c.update(d)
            print(json.dumps(kw), emb, dict(c), flush=True)
