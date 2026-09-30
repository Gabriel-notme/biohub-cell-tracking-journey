"""How often are the link changes made by each B5 refinement stage correct? (labels via GT matching of the final graph)
usage: stage_probe.py <stages_dir> <list>"""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
SD, LST = sys.argv[1], sys.argv[2]
ST = ['00_base', '01_fast_division_review', '02_motion', '03_verified_recovery', '04_centroid', '05_visual_edge', '06_track_ensemble']


def job(name):
    import evalx
    K = evalx.K
    G = {}
    for s in ST:
        d = json.load(open('%s/%s/%s.json' % (SD, s, name)))
        G[s] = {(int(e['source_id']), int(e['target_id'])) for e in d['edges']}
    nodes, edges = evalx.load_graph_json('%s/06_track_ensemble/%s.json' % (SD, name))
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    from tracking_cellmot.metrics import evaluate
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    ea = gt.edge_attrs(); gs = defaultdict(set); gp = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(x)].add(int(y)); gp[int(y)] = int(x)
    def lab(e):
        a, b = e; ga, gb = p2g.get(a), p2g.get(b)
        if ga is not None and gb is not None and gb in gs.get(ga, ()): return 'TP'
        if (ga is not None and gs.get(ga)) or (gb is not None and gb in gp): return 'FP'
        return 'U'
    c = Counter()
    for s0, s1 in zip(ST[:-1], ST[1:]):
        for e in G[s0] - G[s1]: c[(s1, 'removed', lab(e))] += 1
        for e in G[s1] - G[s0]: c[(s1, 'added', lab(e))] += 1
    return c


if __name__ == '__main__':
    names = [l.strip() for l in open(LST) if l.strip()]
    with Pool(36) as pool: cs = pool.map(job, names)
    tot = Counter()
    for c in cs: tot.update(c)
    for s in ST[1:]:
        for kind in ['removed', 'added']:
            tp, fp, u = tot[(s, kind, 'TP')], tot[(s, kind, 'FP')], tot[(s, kind, 'U')]
            if tp + fp + u: print('%-26s %-8s TP %4d FP %4d U %6d   (among labelled: %.2f TP)' % (s, kind, tp, fp, u, tp / max(1, tp + fp)))
