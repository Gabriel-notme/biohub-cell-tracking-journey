"""Audit: anatomy of P13 FN edges (official matching) and, for 'break' FNs (matched src is a track end and matched dst a track start),
whether edge_link sees the pair as a candidate on the final graph and with what probability. Diagnostic only."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p56stage')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def job(f):
    import evalx, edge_link, lgb_np
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    K = evalx.K
    name = Path(f).stem; s = f.split('ps_p13_')[1].split('/')[0]
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    out, prev = defaultdict(list), {}
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id']); out[x].append(y); prev[y] = x
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    g2p = {v: k for k, v in p2g.items()}
    ea = gt.edge_attrs()
    gE = [(int(x), int(y)) for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list())]
    rows = edge_link.features(nodes, edges, edge_link.load_full(FULL[s] + '/' + name + '.geff'))
    pr = {}
    if rows:
        X = np.array([[r.get(k, -1) for k in edge_link.FEATS] for r in rows], dtype=np.float32)
        p = lgb_np.load('/workspace/p56stage/edge_lgb.json').predict(X)
        for r, q in zip(rows, p):
            if r['gap'] == 1: pr[(r['s'], r['d'])] = float(q)
    cat = Counter(); brk = []
    for gs, gd in gE:
        ps, pd_ = g2p.get(gs), g2p.get(gd)
        if ps is None or pd_ is None: cat['fn_unmatched'] += 1; continue
        if pd_ in out.get(ps, []): cat['tp'] += 1; continue
        if not out.get(ps) and pd_ not in prev:
            dist = float(np.linalg.norm((np.array([nodes[pd_][k] for k in 'zyx']) - np.array([nodes[ps][k] for k in 'zyx'])) * S))
            q = pr.get((ps, pd_))
            cat['fn_break'] += 1
            cat['fn_break_' + ('nocand' if q is None else ('p>=0.4' if q >= .4 else ('p0.2-0.4' if q >= .2 else 'p<0.2')))] += 1
            brk.append(dict(movie=name, dist=dist, p=q, t=int(nodes[ps]['t'])))
        elif not out.get(ps): cat['fn_src_end_dst_taken'] += 1
        elif pd_ not in prev: cat['fn_src_elsewhere_dst_start'] += 1
        else: cat['fn_swap'] += 1
    return dict(movie=name, set=s, cat=cat, brk=brk)


if __name__ == '__main__':
    files = sorted(glob.glob('/workspace/cl/ps_p13_*/graphs/*.json'))
    with Pool(48) as p: R = p.map(job, files, chunksize=1)
    cat = Counter()
    for r in R: cat.update(r['cat'])
    for k, v in sorted(cat.items()): print(k, v)
    brk = [b for r in R for b in r['brk']]
    nc = [b for b in brk if b['p'] is None]
    print('nocand dist quantiles', np.round(np.quantile([b['dist'] for b in nc], [.1, .5, .9]), 2).tolist() if nc else None,
          'nocand t==99?', sum(b['t'] >= 99 for b in nc))
    for emb in ['44b6', '6bba']:
        c = Counter()
        for r in R:
            if r['movie'].startswith(emb): c.update(r['cat'])
        print(emb, dict(c))
    json.dump(brk, open('/workspace/cl/ideas/ca_fn_brk.json', 'w'))
