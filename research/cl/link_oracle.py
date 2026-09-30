"""Headroom of the link stage: label relink / edge_link candidates (B5 graphs, all 199 movies) with GT and compare
the deployed LightGBM decisions against an oracle. Per candidate: pos = (s->d) is a GT edge between the matched GT nodes;
eval = s or d is in an annotated GT track (only those can change the score). Also marks whether the edges a relink would remove are GT edges.
Writes /workspace/cl/lo2/<set>__<movie>.json"""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
SETS = {'hold36': ('/workspace/runs/b5f_hold36/working/lineage_graphs', '/workspace/runs/fullgraph_hold36'),
        'prev4': ('/workspace/runs/b5f_prev4/working/lineage_graphs', '/workspace/runs/fullgraph_prev4'),
        'audit32': ('/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', '/workspace/sync3/runs/fullgraph_audit32'),
        't127a': ('/workspace/sync4/runs/b5f_t127a/working/lineage_graphs', '/workspace/sync4/runs/fullgraph_t127a'),
        't127b': ('/workspace/sync3/runs/b5f_t127b/working/lineage_graphs', '/workspace/sync3/runs/fullgraph_t127b')}
OUT = Path('/workspace/cl/lo2'); OUT.mkdir(exist_ok=True)


def job(args):
    s, f = args
    name = Path(f).stem; of = OUT / ('%s__%s.json' % (s, name))
    if of.exists(): return 1
    import evalx, relink, edge_link, lgb_np
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    fullp = Path(SETS[s][1]) / (name + '.geff')
    full = edge_link.load_full(fullp)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    ea = gt.edge_attrs(); GE = set(); gs = defaultdict(list); gp = defaultdict(list)
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): GE.add((int(x), int(y))); gs[int(x)].append(int(y)); gp[int(y)].append(int(x))
    isge = lambda a, b: a is not None and b is not None and p2g.get(a) is not None and p2g.get(b) is not None and (p2g[a], p2g[b]) in GE
    evs = lambda a: a is not None and p2g.get(a) is not None and bool(gs.get(p2g[a]))
    evt = lambda b: b is not None and p2g.get(b) is not None and bool(gp.get(p2g[b]))
    out = {}
    rl = relink.features(nodes, edges, full)
    if rl:
        X = np.array([[r.get(k, -1) for k in relink.FEATS] for r in rl], dtype=np.float32); p = lgb_np.load('/workspace/p56stage/relink_lgb.json').predict(X)
        out['rl'] = [dict(p=float(q), pos=int(isge(r['s'], r['d'])), ev=int(evs(r['s']) or evt(r['d'])),
                          cur_s_ge=int(isge(r['s'], r['cur_d'])) if r['cur_d'] is not None else -1,
                          cur_d_ge=int(isge(r['cur_s'], r['d'])) if r['cur_s'] is not None else -1,
                          cur_s_ev=int(evs(r['s']) or evt(r['cur_d'])) if r['cur_d'] is not None else -1,
                          cur_d_ev=int(evs(r['cur_s']) or evt(r['d'])) if r['cur_s'] is not None else -1, typ=r['typ'], s=r['s'], d=r['d'])
                     for r, q in zip(rl, p)]
    el = edge_link.features(nodes, edges, full)
    if el:
        X = np.array([[r.get(k, -1) for k in edge_link.FEATS] for r in el], dtype=np.float32); p = lgb_np.load('/workspace/p56stage/edge_lgb.json').predict(X)
        rows = []
        for r, q in zip(el, p):
            if r['gap'] == 1: pos = isge(r['s'], r['d'])
            else:
                gsn, gdn = p2g.get(r['s']), p2g.get(r['d'])
                pos = gsn is not None and gdn is not None and any((m, gdn) in GE for m in gs.get(gsn, []))
            rows.append(dict(p=float(q), gap=r['gap'], pos=int(pos), ev=int(evs(r['s']) or evt(r['d'])), s=r['s'], d=r['d']))
        out['el'] = rows
    of.write_text(json.dumps(out))
    return 1


if __name__ == '__main__':
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob(SETS[s][0] + '/*.json'))]
    with Pool(64) as p: print('done', sum(p.map(job, jobs)), len(jobs))
