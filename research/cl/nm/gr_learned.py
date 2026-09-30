"""Learned-cost re-solve on top of P15 (P15 edges and divisions kept; additions only, since learned cuts/swaps have no LOEO signal).
Candidates = union graph (P15 nodes + dropped pre-ILP detections; P15 + pre-ILP + 10 um radius pairs); cost model = gr_train2 LightGBM.
mdl='loeo' uses the model trained on the other embryo; mdl='oof' the inner-fold model of this movie's own embryo (threshold selection).
solver='assign': per-frame-pair max-weight assignment of (p - theta) over free ends / free starts / dropped detections.
solver='scip'  : same edges, plus a start cost A and end cost Dc for every activated dropped detection (global chain coupling)."""
import gr_common as C
import json, time
import numpy as np
WANTS_META = True
_B = {}


def _model(tag, mdl, name):
    import lightgbm as lgb
    D = '/workspace/cl/nm/gr_models'
    meta = json.load(open('%s/gr2_%s_meta.json' % (D, tag)))
    e = name[:4]
    if mdl == 'loeo': p = '%s/gr2_%s_full_%s.txt' % (D, tag, '6bba' if e == '44b6' else '44b6')
    else: p = '%s/gr2_%s_%s_f%d.txt' % (D, tag, e, meta['fold'][name])
    if p not in _B: _B[p] = lgb.Booster(model_file=p)
    return _B[p], meta['use']


def apply(nodes, edges, theta=0.5, tag='v1', mdl='loeo', cats='drop', solver='assign', A=0.0, Dc=0.0, Cn=0.0, attach=0, **meta):
    name, sname, fullgeff = meta['name'], meta['set'], meta['fullgeff']
    import gr_feat
    t0 = time.time()
    full = C.load_full(fullgeff)
    U = C.build_union(nodes, edges, full, radius=10, knn_all=True)
    keys, X = gr_feat.features(U, nodes, edges, full, gr_feat.ilp_sets(sname, name))
    bst, use = _model(tag, mdl, name)
    ui = [gr_feat.FEATS.index(f) for f in use]
    ix = U['ix']; inp = U['inp']
    has_out = set(); has_in = set()
    for e in edges:
        a, b = ix[int(e['source_id'])], ix[int(e['target_id'])]
        has_out.add(a); has_in.add(b)
    sel_rows = [k for k, (a, b) in enumerate(keys) if X[k, 0] == 0 and a not in has_out and b not in has_in
                and (cats != 'drop' or not (inp[a] and inp[b]))]
    st = {'gr_cands': len(sel_rows)}
    if not sel_rows: return nodes, edges, dict(st, gr_added=0)
    p = bst.predict(X[sel_rows][:, ui])
    W = {keys[k]: float(q - theta) for k, q in zip(sel_rows, p) if q > theta}
    if attach:  # only edges with at least one P15 endpoint (extensions / bridges into P15 tracks, no standalone dropped fragments)
        W = {k: w for k, w in W.items() if inp[k[0]] or inp[k[1]]}
    if solver == 'assign':
        W = {(a, b): w - Cn * ((not inp[a]) + (not inp[b])) for (a, b), w in W.items()}
        sel = C.solve_assign(U, W, set(), set())
    else:
        sel = solve_scip(U, W, A, Dc, Cn)
    nn = dict(nodes)
    ne = list(edges)
    added = 0
    for a, b in sel:
        for i in (a, b):
            if not inp[i]:
                n = int(U['ids'][i])
                if n not in nn:
                    z, y, x = U['xyz'][i]; nn[n] = {'node_id': n, 't': int(U['t'][i]), 'z': float(z), 'y': float(y), 'x': float(x), 'gr_added': 1}
        ne.append({'source_id': int(U['ids'][a]), 'target_id': int(U['ids'][b]), 'gr_link': 1}); added += 1
    st.update(gr_added=added, gr_nodes=len(nn) - len(nodes), gr_sec=round(time.time() - t0, 2))
    return nn, ne, st


def solve_scip(U, W, A, Dc, Cn):
    """max sum w x - sum_d [Cn*y_d + A*(y_d - in_d) + Dc*(y_d - out_d)] over dropped detections d; in/out degree <= 1 everywhere."""
    from pyscipopt import Model, quicksum
    from collections import defaultdict
    inp = U['inp']
    m = Model(); m.hideOutput(); m.setParam('limits/time', 60)
    x = {}
    for (a, b), w in W.items():
        x[(a, b)] = m.addVar(vtype='B', obj=w + (0 if inp[b] else A) + (0 if inp[a] else Dc))
    ins = defaultdict(list); outs = defaultdict(list)
    for (a, b), v in x.items(): outs[a].append(v); ins[b].append(v)
    for n, L in ins.items():
        if len(L) > 1: m.addCons(quicksum(L) <= 1)
    for n, L in outs.items():
        if len(L) > 1: m.addCons(quicksum(L) <= 1)
    for n in set(ins) | set(outs):
        if inp[n]: continue
        y = m.addVar(vtype='B', obj=-(Cn + A + Dc))
        for v in ins.get(n, []) + outs.get(n, []): m.addCons(v <= y)
    m.setMaximize(); m.optimize()
    return [e for e, v in x.items() if m.getVal(v) > 0.5]
