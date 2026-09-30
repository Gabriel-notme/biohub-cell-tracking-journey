"""Literal global re-solve: every union candidate (P15 edges, pre-ILP edges, 10 um radius pairs; P15 nodes + optional dropped
detections) is re-decided by an exact per-frame max-weight assignment of learned LOEO costs w = p - theta (+ m for P15 edges),
P15 divisions fixed. p from the all-category LightGBM trained on the other embryo (gr_models/gr_lgb_train_<other>.txt).
Dropped detections pay Cn per endpoint."""
import gr_common as C
import numpy as np
WANTS_META = True
_B = {}


def apply(nodes, edges, theta=0.5, m=0.0, drop=0, Cn=0.2, **meta):
    import gr_feat, lightgbm as lgb
    name, sname = meta['name'], meta['set']
    full = C.load_full(meta['fullgeff'])
    U = C.build_union(nodes, edges, full, radius=10, knn_all=True)
    keys, X = gr_feat.features(U, nodes, edges, full, gr_feat.ilp_sets(sname, name))
    p_ = '/workspace/cl/nm/gr_models/gr_lgb_train_%s.txt' % ('6bba' if name.startswith('44b6') else '44b6')
    if p_ not in _B: _B[p_] = lgb.Booster(model_file=p_)
    p = _B[p_].predict(X)
    inp = U['inp']; ix = U['ix']
    divs = C.division_structure(nodes, edges)
    fsrc = {ix[a] for a in divs}; fdst = {ix[b] for bs in divs.values() for b in bs}
    div_edges = [e for e in edges if int(e['source_id']) in divs]
    W = {}
    for i, (k, q) in enumerate(zip(keys, p)):
        a, b = k
        if not drop and (not inp[a] or not inp[b]): continue
        w = q - theta + (m if X[i, 0] == 1 else 0.0) - Cn * ((not inp[a]) + (not inp[b]))
        if w > 0: W[k] = w
    sel = C.solve_assign(U, W, fsrc, fdst)
    nn, ne = C.to_graph_dict(U, nodes, sel, div_edges)
    p15e = {(ix[int(e['source_id'])], ix[int(e['target_id'])]) for e in edges}
    kept = sum(1 for x in sel if x in p15e)
    return nn, ne, {'kept': kept, 'removed': len(p15e) - len(div_edges) - kept, 'new': len(sel) - kept, 'added_nodes': len(nn) - len(nodes)}
