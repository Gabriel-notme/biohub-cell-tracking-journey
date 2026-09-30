"""Edge-error anatomy on B5 graphs (all 199 movies), official node matching (7um).
GT edge u->v missed (FN) because: u_unmatched / v_unmatched (detection miss), break (mu has no child and mv has no parent),
switch_out (mu has a child != mv), switch_in (mv has a parent != mu), other.
FP edges x->y: s_matched_out (source matched GT node with children), t_matched_in; and whether x / y are matched at all."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/art_b56/artifact_bundle')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def job(f):
    import evalx
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    K = evalx.K
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    out, prev = defaultdict(list), defaultdict(list)
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id']); out[x].append(y); prev[y].append(x)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    g2p = {v: k for k, v in p2g.items()}
    ea = gt.edge_attrs(); gs = defaultdict(list); gp = defaultdict(list); GE = set()
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()):
        gs[int(x)].append(int(y)); gp[int(y)].append(int(x)); GE.add((int(x), int(y)))
    c = Counter()
    # detection gaps: is the missed node's GT neighbour matched?
    for u, v in GE:
        mu, mv = g2p.get(u), g2p.get(v)
        if mu is None and mv is None: c['fn_both_unmatched'] += 1; continue
        if mu is None: c['fn_u_unmatched'] += 1; continue
        if mv is None: c['fn_v_unmatched'] += 1; continue
        if mv in out.get(mu, []): c['tp'] += 1; continue
        if out.get(mu) and prev.get(mv): c['fn_switch_both'] += 1
        elif out.get(mu): c['fn_switch_out'] += 1
        elif prev.get(mv): c['fn_switch_in'] += 1
        else: c['fn_break'] += 1
    for x in out:
        for y in out[x]:
            gx, gy = p2g.get(x), p2g.get(y)
            ev = (gx is not None and gs.get(gx)) or (gy is not None and gp.get(gy))
            if not ev: continue
            if gx is not None and gy is not None and (gx, gy) in GE: continue
            c['fp_' + ('both_m' if gx is not None and gy is not None else ('src_m' if gx is not None else 'tgt_m'))] += 1
    c['gt_nodes'] = len(set(gs) | set(gp)); c['gt_nodes_unmatched'] = sum(1 for g in set(gs) | set(gp) if g not in g2p)
    return name, dict(c)


if __name__ == '__main__':
    fs = [f for s in B5 for f in sorted(glob.glob(B5[s] + '/*.json'))]
    with Pool(64) as p: R = p.map(job, fs)
    json.dump(dict(R), open('/workspace/cl/err_cat_b5.json', 'w'))
    for emb in ['44b6', '6bba', '']:
        T = Counter()
        for n, c in R:
            if n.startswith(emb): T.update(c)
        print(emb or 'all', dict(sorted(T.items())))
